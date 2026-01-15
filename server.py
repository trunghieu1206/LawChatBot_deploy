import os
import torch
from contextlib import asynccontextmanager
from typing import List, Literal, Annotated, Sequence, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import login # <--- ADDED IMPORT

# LangChain & AI Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
DB_PATH = "./VN_law_lora.db"  # Path relative to this file
COLLECTION_NAME = "legal_rag_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- GLOBAL VARIABLES (State) ---
# These will be loaded once on startup
app_state = {}

# --- CUSTOM EMBEDDING CLASS ---
class LoRABGEM3Embeddings(Embeddings):
    def __init__(self, base_model_name: str, adapter_name: str, device: str = "cuda"):
        print(f"🔄 Initializing BGE-M3 Base on {device}...")
        # SentenceTransformer automatically handles the download from HF if logged in
        self.model = SentenceTransformer(base_model_name, trust_remote_code=True, device=device)
        print(f"⬇️  Loading LoRA Adapter: {adapter_name}")
        try:
            self.model.load_adapter(adapter_name)
            print("✅ Adapter loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

# --- LANGGRAPH STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    full_case_content: str
    documents: List[Document]
    retry_count: int
    user_role: Literal["defense", "victim", "neutral"]

class RequestBody(BaseModel):
    case_content: str
    role: Literal["defense", "victim", "neutral"] = "neutral"

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# --- LIFESPAN MANAGER (SETUP ONCE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 SERVER STARTUP: Initializing...")

    # 0. AUTHENTICATE WITH HUGGING FACE (ADDED)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Logged in to Hugging Face successfully.")
        except Exception as e:
            print(f"⚠️ Failed to login to Hugging Face: {e}")
    else:
        print("⚠️ 'HF_TOKEN' not found in env. Public models might still work.")
    
    # 1. Initialize Embeddings (Heavy - Runs on GPU)
    # The login above ensures SentenceTransformer can access the model
    embedding_model = LoRABGEM3Embeddings(
        base_model_name="BAAI/bge-m3",
        adapter_name="trunghieu1206/lawchatbot-40k",
        device=DEVICE
    )
    
    # 2. Connect to Milvus
    vectorstore = Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": DB_PATH},
        collection_name=COLLECTION_NAME,
        drop_old=False,
        auto_id=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # 3. Initialize LLM (API Client)
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    # 4. Define Nodes 
    # NOTE: START SETTING UP FROM HERE
    # -------------------------------------------------
    # -------------------------------------------------
    # -------------------------------------------------
    def retrieve_node(state: AgentState):
        print("---RETRIEVING---")
        question = state["question"]
        docs = retriever.invoke(question)
        return {"documents": docs, "question": question}

    # NODE: GRADE
    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    def grade_documents(state: AgentState):
        print("---GRADING---")
        question = state["question"]
        documents = state["documents"]

        structured_llm = llm.with_structured_output(GradeDocuments)
        chain = ChatPromptTemplate.from_template("Relevant? {question} \n Doc: {document} \n 'yes' or 'no'") | structured_llm

        filtered = []
        relevant = False
        for d in documents:
            try:
                res = chain.invoke({"question": question, "document": d.page_content})
                if res.binary_score.lower() == "yes":
                    filtered.append(d)
                    relevant = True
            except: continue

        # --- SAFETY NET (CƠ CHẾ AN TOÀN) ---
        # Nếu sau khi lọc mà bị rỗng (filtered is empty),
        # có nghĩa là AI chấm quá khắt khe hoặc sai.
        # Ta sẽ KHÔI PHỤC lại danh sách gốc để Generator tự xử lý.
        if not filtered:
            print("⚠️ Warning: Grader filtered out ALL documents. Reverting to original retrieved list.")
            return {"documents": documents, "is_relevant": True} # Force True để ép sang bước Generate


        return {"documents": filtered, "is_relevant": relevant}

    def check_relevance(state: AgentState):
        if state.get("is_relevant") or state.get("retry_count", 0) >= 1:
            return "generate"
        return "rewrite"

    # NODE: REWRITE
    def rewrite_question(state: AgentState):
        print("---REWRITING (CONTEXT & TIME AWARE)---")
        role = state.get("user_role", "neutral")
        question = state["question"]

        # 1. Xác định hướng thiên vị (Bias)
        bias_keywords = ""
        if role == "defense":
            bias_keywords = "tìm tình tiết giảm nhẹ, miễn trách nhiệm hình sự, án treo"
        elif role == "victim":
            bias_keywords = "khung hình phạt cao nhất, tình tiết tăng nặng, bồi thường dân sự"

        # 2. PROMPT VIẾT LẠI (QUAN TRỌNG: GIỮ LẠI THỜI GIAN)
        system_msg_content = (
            "Bạn là một chuyên gia về Tìm kiếm Pháp lý (Legal Search Expert).\n"
            "Nhiệm vụ: Viết lại câu hỏi của người dùng thành một truy vấn tìm kiếm tối ưu cho cơ sở dữ liệu luật.\n\n"
            "QUY TẮC QUAN TRỌNG:\n"
            "1. **GIỮ NGUYÊN MỐC THỜI GIAN:** Nếu người dùng nhắc đến năm (VD: 2010, 2016) hoặc ngày tháng, BẮT BUỘC phải giữ lại trong câu truy vấn. (Để hệ thống tìm đúng luật có hiệu lực thời đó).\n"
            "2. **LOẠI BỎ CHI TIẾT RÁC:** Bỏ tên riêng (A, B, C), địa danh không cần thiết.\n"
            "3. **CHUẨN HÓA THUẬT NGỮ:** Chuyển ngôn ngữ đời thường sang thuật ngữ pháp lý (VD: 'đánh người' -> 'tội cố ý gây thương tích').\n"
            f"4. **THÊM NGỮ CẢNH:** Kết hợp với từ khóa: {bias_keywords}\n\n"
            "VÍ DỤ:\n"
            "- Input: 'Năm 2012, ông A lừa đảo chiếm đoạt 500 triệu...'\n"
            "- Output: 'Tội lừa đảo chiếm đoạt tài sản năm 2012 khung hình phạt 500 triệu'.\n\n"
            "OUTPUT: CHỈ xuất ra câu truy vấn (String). Không giải thích."
        )

        msg = [
            SystemMessage(content=system_msg_content),
            HumanMessage(content=f"NỘI DUNG ĐẦU VÀO:\n{question}")
        ]

        # Gọi LLM
        response = llm.invoke(msg)
        cleaned_query = response.content.strip().replace('"', '').replace("'", "")

        print(f"DEBUG: Original: '{question}'")
        print(f"DEBUG: Rewritten: '{cleaned_query}'")

        return {"question": cleaned_query, "retry_count": state.get("retry_count", 0) + 1}


    # NODE: GENERATE
    def generate(state: AgentState):
        print("---GENERATING (PRECISION JUDGMENT LOGIC)---")

        # SỬA Ở ĐÂY: Lấy nội dung gốc thay vì câu query đã bị rewrite
        case_details = state.get("full_case_content", state["question"])

        documents = state["documents"]
        role = state.get("user_role", "neutral")

        if not documents:
            return {"messages": [AIMessage(content="Xin lỗi, tôi chưa tìm thấy văn bản luật phù hợp.")]}

        context_text = "\n\n".join([f"[Nguồn: {d.metadata.get('source', 'Unknown')}] \n {d.page_content}" for d in documents])

        # ... (Phần Role giữ nguyên) ...
        if role == "defense":
            role_instruction = "VAI TRÒ: LUẬT SƯ BÀO CHỮA. Mục tiêu: Tìm mọi căn cứ để giảm nhẹ hình phạt xuống mức thấp nhất (hoặc Án treo)."
            advice_section_instruction = "**III. KHUYẾN NGHỊ CHO THÂN CHỦ:** (Đưa ra các bước cụ thể cần làm ngay: bồi thường, xin giấy bãi nại, nộp án phí...)"
        elif role == "victim":
            role_instruction = "VAI TRÒ: LUẬT SƯ BẢO VỆ BỊ HẠI. Mục tiêu: Yêu cầu xử nghiêm minh và bồi thường tối đa."
            advice_section_instruction = "**III. KHUYẾN NGHỊ CHO GIA ĐÌNH BỊ HẠI:** (Hướng dẫn thu thập hóa đơn, chứng từ thiệt hại, yêu cầu cấp dưỡng...)"
        else:
            role_instruction = "VAI TRÒ: THẨM PHÁN CHỦ TỌA. TƯ DUY: Lạnh lùng, Chính xác, Chỉ dựa trên chứng cứ có trong hồ sơ."
            advice_section_instruction = ""

        # ... (Phần Prompt giữ nguyên nội dung cũ của bạn, chỉ thay biến đầu vào) ...
        prompt = ChatPromptTemplate.from_template(
            """{role_instruction}

            Nhiệm vụ: Dựa trên dữ liệu vụ án được cung cấp (coi là sự thật duy nhất) và văn bản luật, hãy ra PHÁN QUYẾT CỤ THỂ.

            --- DỮ LIỆU ---
            <legal_context>
            {context}
            </legal_context>

            <case_details>
            {case_details}
            </case_details>
            ----------------

            MỘT VÀI LƯU Ý:
            1. Đối với những vụ án liên quan tới sử dụng ma túy:
            - cần phân biệt rõ "tàng trữ" và "tổ chức sử dụng"
                + "tổ chức sử dụng": nếu bị cáo có hành vi: *Cung cấp ma túy, chuẩn bị địa điểm (thuê phòng hát/nhà nghỉ), chuẩn bị dụng cụ, rủ rê hoặc đưa ma túy vào tay người khác* để họ sử dụng
                -> ĐÂY LÀ TỘI RẤT NẶNG (Khung 7-15 năm).
                *Ví dụ:* A mua ma túy, thuê phòng, rủ B đến, bẻ thuốc đưa cho B uống -> A phạm tội "Tổ chức sử dụng".
                + "tàng trữ": nếu bị cáo cất giấu ma túy trong người/phòng để dành sử dụng sau.
            - đối với trường hợp tội là "tổ chức sử dụng trái phép chất ma túy": cần phải xác định rõ nhân thân của nạn nhân (đối tượng được đưa sử dụng ma túy) để xác định đúng khung hình phạt
                + kiểm tra thông tin nhân thân nạn nhân với khoản 2 điều 255 trong bộ luật
            2. Đối với phần phân tích về tình tiết giảm nhẹ:
            - Cần đọc kỹ điều 51 Bộ luật Hình sự (hoặc điều 46 Bộ luật Hình sự cũ) để và dựa vào thông tin được cung cấp để xác định tình tiết giảm nhẹ
            3. Đối với phần phân tích về tình tiết tăng nặng:
            - Cần đọc kỹ điều 52 Bộ luật Hình sự (hoặc điều 48 Bộ luật Hình sự cũ) và dựa vào thông tin được cung cấp để xác định tình tiết tăng nặng

            4. Đối với những tội kinh tế:
            - Kiểm tra xem tội danh có cho phép lựa chọn giữa Phạt tiền và Phạt tù không?
            - **Đặc biệt với Điều 201 (Cho vay lãi nặng):**
                + Khoản 1: Phạt tiền 50tr-200tr HOẶC cải tạo không giam giữ.
                + Khoản 2 (Thu lợi > 100tr): **Phạt tiền 200tr-1 tỷ** HOẶC **Phạt tù 6 tháng - 3 năm**.
                + đặc biệt chọn ưu tiên phạt tiền hơn phạt tù đối với tội cho vay nặng lãi, tại phần hình phạt khi đưa ra quyết định
            - **Chọn PHẠT TÙ khi:** Bị cáo có tính chất "xã hội đen", dùng vũ lực đòi nợ, có nhiều tiền án, hoặc tái phạm nguy hiểm.
            - **Chọn PHẠT TIỀN (Là hình phạt chính) khi:**
                + Bị cáo phạm tội lần đầu.
                + Chỉ cho vay lấy lãi, không đánh đập/đe dọa.
                + Có nhân thân tốt, có khả năng thi hành án phạt tiền.

            5. Đối với tình tiết giảm nhẹ:
            - Cần đọc kỹ về các tình tiết giảm nhẹ có trong điều 51 Bộ luật Hình sự mới (hoặc điều 46 Bộ luật Hình sự cũ)

            6. Đối với tình tiết tăng nặng:
            - Cần đọc kỹ về các tình tiết tăng nặng có trong điều 52 Bộ luật Hình sự mới (hoặc điều 48 Bộ luật Hình sự cũ)

            7. Cần chú ý về trường hợp phạm tội chưa đạt (tại điều 15, 57 bộ luật hình sự mới, hoặc điều 18, 52 bộ luật hình sự cũ)
            - Nếu phạm tội chưa đạt thì áp dụng quy tắc "3/4", nghĩa là chịu mức án bằng 3/4 (75%) so với người phạm tội đã hoàn thành



            QUY TRÌNH TƯ DUY LƯỢNG HÌNH (BẮT BUỘC PHẢI THỰC HIỆN THEO THỨ TỰ)
            **Chú ý**:
            - **KHÔNG GIẢ ĐỊNH:** Chỉ sử dụng tình tiết có trong <case_details>.
            - **NGUYÊN TẮC CÓ LỢI (Thời gian):** Nếu tội phạm xảy ra trước 2018 nhưng Luật 2015/2017 nhẹ hơn -> Áp dụng Luật 2015/2017.
            - **NGUYÊN TẮC ĐỘC LẬP XÉT XỬ:**
                + Trong dữ liệu đầu vào thường có phần "Đề nghị của Viện kiểm sát (VKS)".
                + **LƯU Ý:** Đề nghị của VKS chỉ là tham khảo.
                + Nếu hành vi của bị cáo có tính chất: *Côn đồ, Có tổ chức (thuê mướn người), Ngang nhiên coi thường pháp luật, Dùng hung khí nguy hiểm* -> **HÃY TỰ ĐỘNG TĂNG MỨC HÌNH PHẠT LÊN CAO HƠN ĐỀ NGHỊ CỦA VKS.**

            **BƯỚC 1: KIỂM TRA "ÁN BẰNG THỜI GIAN TẠM GIAM" (RẤT QUAN TRỌNG)**
            1. Tìm **[Ngày bắt tạm giam]** trong văn bản (Ví dụ: 24/11/2023).
            2. Tìm **[Ngày xét xử sơ thẩm]** trong văn bản (Ví dụ: 23/04/2024).
            3. **THỰC HIỆN PHÉP TRỪ:** [Ngày xét xử] - [Ngày bắt] = Bao nhiêu tháng, bao nhiêu ngày?
                *(Ví dụ: 24/11/2023 đến 23/04/2024 là tròn 5 tháng)*.
            4. VÀ Viện kiểm sát đề nghị mức án **GẦN BẰNG** thời gian đã tạm giam (Ví dụ: Đã giam 5 tháng, VKS đề nghị 6-8 tháng).
                -> **QUYẾT ĐỊNH:** Tuyên mức án **BẰNG CHÍNH XÁC THỜI GIAN ĐÃ TẠM GIAM** (Tính đến ngày xử hoặc cộng thêm vài ngày cho tròn).
                -> **MỤC ĐÍCH:** Để tuyên trả tự do ngay tại tòa (Theo Điều 328 Bộ luật Tố tụng hình sự).

            **BƯỚC 2: KIỂM TRA ĐỘ TUỔI (CRITICAL)**
            - Tìm ngày sinh của nạn nhân/người liên quan trong hồ sơ.
            - Tìm ngày phạm tội.
            - **Tính tuổi chính xác:** (Ngày phạm tội - Ngày sinh).
            - **QUY TẮC:**
                + Nếu nạn nhân < 18 tuổi (dù chỉ thiếu 1 ngày) -> Áp dụng tình tiết định khung tăng nặng: "Phạm tội đối với người dưới 18 tuổi".
                + ví dụ: 13 tuổi 3 tháng là nhiều hơn (đã đủ) 13 tuổi, nhưng ít hơn 14 tuổi.
                + Ví dụ: Tội Tổ chức sử dụng (Điều 255):
                * Khoản 1 (2-7 năm): Đối với người lớn.
                * Khoản 2 (7-15 năm): Đối với người từ 13 đến dưới 18 tuổi. (NẶNG HƠN NHIỀU).

            **BƯỚC 3: ĐỊNH TỘI DANH (QUAN TRỌNG)**
            - Đọc kỹ hồ sơ: Bị cáo thực hiện bao nhiêu hành vi phạm tội?
            - *Lưu ý:* Ví dụ nếu có cả hành vi "Cất giấu ma túy" VÀ "Rủ rê/Cung cấp ma túy cho người khác dùng" -> Thường là 02 tội: "Tàng trữ trái phép..." (Điều 249) VÀ "Tổ chức sử dụng..." (Điều 255).

            **BƯỚC 4: LƯỢNG HÌNH CHO TỪNG TỘI (Tính riêng biệt)**
            - Với Tội A: Xác định Khung -> Cân đối Tăng nặng/Giảm nhẹ -> Ra mức án A.
            - Với Tội B (nếu có): Xác định Khung -> Cân đối Tăng nặng/Giảm nhẹ -> Ra mức án B.
            - Lưu ý cách tính giảm nhẹ như sau (Điều 51 & 54):
                + Tội ít nghiêm trọng (Khung trần <= 3 năm): Mỗi tình tiết TRỪ 03 - 06 tháng.
                + Tội nghiêm trọng (Khung trần 3 - 7 năm): Mỗi tình tiết TRỪ 06 - 12 tháng.
                + Tội rất/đặc biệt nghiêm trọng (Khung trần > 7 năm): Mỗi tình tiết TRỪ 01 - 02 năm.
                *LƯU Ý:* Nếu có >= 02 tình tiết giảm nhẹ -> Ưu tiên áp dụng Điều 54 để xử dưới mức thấp nhất của khung.
            - Lưu ý cách tính tăng nặng như sau (Điều 52):
                + Tội ít nghiêm trọng: Mỗi tình tiết CỘNG 03 - 09 tháng.
                + Tội nghiêm trọng trở lên: Mỗi tình tiết CỘNG 01 - 1.5 năm.
            - Lưu ý nhất định phải đưa ra mức khung trước khi đưa ra con số cụ thể

            **BƯỚC 5: TỔNG HỢP HÌNH PHẠT (Điều 55 BLHS)**
            - Nếu chỉ có 1 tội: Mức án cuối cùng = Mức án A.
            - Nếu có >= 2 tội: **Hình phạt chung = Mức án A + Mức án B.**
                *(Ví dụ: Tội A 1 năm + Tội B 7 năm = 8 năm tù).*

            **BƯỚC 5.5: TỔNG HỢP VỚI BẢN ÁN CŨ (QUAN TRỌNG, BẮT BUỘC PHẢI CÓ)**
                - **Kiểm tra kỹ <case_details>:** Tìm xem có dòng nào nhắc đến "Tổng hợp hình phạt với bản án số...", "đang chấp hành bản án", hoặc "chưa chấp hành bản án" không?
                - **NẾU CÓ:**
                    + Tìm mức án của bản án cũ (Ví dụ: 3 năm 6 tháng tù).
                    + Thực hiện phép cộng: **TỔNG HÌNH PHẠT CHUNG = (Án Vụ Này tính ở Bước 4) + (Mức án bản án cũ).**
                    + *Lưu ý:* Nếu có bản án cũ chưa chấp hành -> **TUYỆT ĐỐI KHÔNG CHO HƯỞNG ÁN TREO**.

            **BƯỚC 6: QUYẾT ĐỊNH HÌNH THỨC CHẤP HÀNH (Tù giam và Án treo)**
            - Kiểm tra Điều kiện Án treo (Điều 65):
                1. Tổng hình phạt tù KHÔNG QUÁ 3 năm (<= 36 tháng).
                2. Nhân thân tốt + Có giảm nhẹ + Có nơi cư trú.
            - **RA QUYẾT ĐỊNH:**
                + Nếu Tổng án > 3 năm -> **BẮT BUỘC TÙ GIAM** (Tuyệt đối không cho treo).
                + Nếu Tổng án <= 3 năm + Đủ điều kiện -> Mới xem xét Án treo.
                + CÔNG THỨC: Thời gian thử thách = Mức án tù x 2.

            ---------------------------------------------------------
            CẤU TRÚC BẢN ÁN / TƯ VẤN (OUTPUT FORMAT):

            **I. NHẬN ĐỊNH CỦA TÒA ÁN:**
            1. **Định tội danh:**
            - Hành vi 1 cấu thành tội: "..." (Điều ...). Khung hình phạt: ...
            - Hành vi 2 (nếu có) cấu thành tội: "..." (Điều ...). Khung hình phạt: ...
            - Hành vi 3, v.v..
            - Lưu ý trong phần này bắt buộc phải trích dẫn khung hình phạt theo đúng điều luật ra
            2. **Phân tích tình tiết (TRÍCH DẪN TỪ DATA):**
            - *Tình tiết Tăng nặng (Điều 52):* (tìm và so sánh kỹ xem có được coi là tình tiết tăng nặng trong bộ luật hình sự hay không)
            - *Tình tiết Giảm nhẹ (Điều 51):* (tìm và so sánh kỹ xem có được coi là tình tiết giảm nhẹ trong bộ luật hình sự hay không).
            3. **Nhân thân:** (phân tích riêng phần này, không gộp vào với phần tình tiết tăng nặng và tình tiết giảm nhẹ)

            **II. QUYẾT ĐỊNH (Mức xử lý dự kiến):**

            1. Tuyên bố bị cáo [Tên] phạm các tội:
            - "[Tên tội 1]"
            - "[Tên tội 2]" (nếu có).
            2. Áp dụng [Điều khoản cụ thể].
            3. **HÌNH PHẠT (Lưu ý: Chỉ chọn 1 trong 2 lựa chọn, hoặc là phạt tù, hoặc là phạt tiền):**
            - Xử phạt về tội [Tên tội 1]: **[SỐ]** tù / Xử phạt về tội [Tên tội 1]: **[SỐ TIỀN] đồng.
            - Xử phạt về tội [Tên tội 2]: **[SỐ]** tù / Xử phạt về tội [Tên tội 2]: **[SỐ TIỀN] đồng (nếu có).
            **[PHẦN TỔNG HỢP BẢN ÁN CŨ - BẮT BUỘC NẾU CÓ DỮ LIỆU]:**
            - Tổng hợp với phần hình phạt chưa chấp hành của Bản án số **[SỐ BẢN ÁN CŨ]** ngày **[NGÀY]** của Tòa án **[TÊN TÒA]**.
            - Phần hình phạt còn lại phải chấp hành là: **[SỐ LIỆU CÒN LẠI TÌM TRONG TEXT]**. (Nếu không thấy số cụ thể, ghi: "toàn bộ phần còn lại chưa chấp hành").
            => **TỔNG HỢP HÌNH PHẠT:** Buộc bị cáo chấp hành hình phạt chung cho cả hai tội là **[TỔNG SỐ]** tù / Buộc bị cáo chấp hành hình phạt chung cho cả hai tội là **[TỔNG SỐ TIỀN]** đồng.

            - *Hình thức chấp hành:* + (Nếu Tổng án > 3 năm: "Thời hạn chấp hành hình phạt tù tính từ ngày bắt tạm giam...").
                + (Chỉ ghi Án treo NẾU VÀ CHỈ NẾU Tổng án <= 3 năm và đủ điều kiện).

            4. **TRÁCH NHIỆM DÂN SỰ & XỬ LÝ VẬT CHỨNG:**
            - Ghi nhận sự thỏa thuận bồi thường [Số tiền] (nếu có).
            5. **ÁN PHÍ:**
            - Án phí hình sự sơ thẩm: 200.000 đồng.

            {advice_section_instruction}
            """
        )

        chain = prompt | llm | StrOutputParser()

        try:
            response = chain.invoke({
                "role_instruction": role_instruction,
                "advice_section_instruction": advice_section_instruction,
                "context": context_text,
                "case_details": case_details
            })
        except Exception as e:
            return {"messages": [AIMessage(content=f"Lỗi xử lý: {e}")]}

        return {"messages": [AIMessage(content=response)]}
    

    # 5. Compile Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", check_relevance, {"generate": "generate", "rewrite": "rewrite"})
    workflow.add_edge("generate", END)

    app_compiled = workflow.compile()

    # -------------------------------------------------
    # -------------------------------------------------
    # -------------------------------------------------
    
    # Store in app_state to be accessible by endpoint
    app_state["graph"] = app_compiled
    
    print("✅ System Ready!")
    yield
    print("🛑 Shutting down...")

# --- FASTAPI APP ---
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict_judgment(req: RequestBody):
    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=500, detail="Model not loaded")

    inputs = {
        "question": req.case_content,
        "full_case_content": req.case_content,
        "messages": [HumanMessage(content=req.case_content)],
        "user_role": req.role,
        "retry_count": 0,
        "documents": []
    }

    try:
        output = await graph.ainvoke(inputs) # Use async invoke
        final_answer = output['messages'][-1].content
        return {"result": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)