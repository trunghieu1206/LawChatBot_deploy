import streamlit as st
import requests
import json

# --- CONFIGURATION ---
SERVER_IP = "209.121.195.118"
PORT = "13014"  # The mapped public port
API_URL = f"http://{SERVER_IP}:{PORT}/predict"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Vietnam Legal AI Assistant",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #FF3333;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Trợ Lý Pháp Lý AI")
st.markdown("Hệ thống tư vấn pháp luật.")

# --- SIDEBAR (Settings) ---
with st.sidebar:
    st.header("Cấu hình")
    role_choice = st.radio(
        "Chọn Vai Trò Tư Vấn:",
        ("Trung lập (Thẩm phán)", "Luật sư Bào chữa", "Luật sư Bảo vệ Bị hại"),
        index=0
    )
    
    # Map friendly names to API values
    role_map = {
        "Trung lập (Thẩm phán)": "neutral",
        "Luật sư Bào chữa": "defense",
        "Luật sư Bảo vệ Bị hại": "victim"
    }
    selected_role = role_map[role_choice]
    
    st.info("""
    **Hướng dẫn:**
    1. Nhập nội dung vụ án vào ô bên phải.
    2. Chọn vai trò bạn muốn AI đóng.
    3. Nhấn 'Phân Tích'.
    """)

# --- MAIN INPUT AREA ---
default_text = """Ngày 30 tháng 9 năm 2025, tại Thành phố Hồ Chí Minh.
Bị cáo: Đồng Quang H, sinh năm 1999.
Nội dung: Bị cáo lấy trộm 01 điện thoại iPhone 15 Pro Max và 01 iPhone 14 Pro Max.
Tổng trị giá tài sản là 35.900.000 đồng.
Bị cáo đã khai nhận toàn bộ hành vi."""

case_input = st.text_area(
    "Nội dung vụ việc / Tình huống:",
    value=default_text,
    height=250,
    placeholder="Nhập chi tiết vụ án tại đây..."
)

# --- ACTION BUTTON ---
if st.button("Phân Tích", use_container_width=True):
    if not case_input.strip():
        st.warning("Vui lòng nhập nội dung vụ việc trước khi phân tích!")
    else:
        # Prepare the payload
        payload = {
            "case_content": case_input,
            "role": selected_role
        }

        # Show a spinner while waiting
        with st.spinner("🤖 AI đang tra cứu luật và phân tích hồ sơ... (Mất khoảng 5-20 giây)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    result_text = data.get("result", "Không có dữ liệu trả về.")
                    
                    st.success("✅ Phân tích hoàn tất!")
                    st.divider()
                    st.markdown("### 📄 KẾT QUẢ TƯ VẤN:")
                    st.markdown(result_text) # Markdown renders nicely
                else:
                    st.error(f"❌ Lỗi Server ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Không thể kết nối đến Server ({API_URL}).")
                st.info("💡 Gợi ý: Kiểm tra xem Server GPU có đang chạy không hoặc Port có đúng không.")
            except requests.exceptions.Timeout:
                st.error("⏰ Hết thời gian chờ (Timeout). Server đang xử lý quá lâu.")
            except Exception as e:
                st.error(f"❌ Lỗi không xác định: {e}")

# --- FOOTER ---
st.divider()
st.caption("Base embedding model: BGE-M3, fine-tuned on Vietnamese legal case data.")