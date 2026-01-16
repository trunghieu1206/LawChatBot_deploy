# Description
Vietnam Penal Law AI Assistant

**A Specialized RAG System for Vietnamese Criminal Jurisprudence**

This project is a high-precision AI chatbot built exclusively for **Vietnamese Penal Law**. Unlike general legal assistants, this system is engineered to navigate the complex logic of criminal proceedings.

By combining Retrieval-Augmented Generation (RAG) with a fine-tuned LoRA model, the AI acts as a specialized criminal law consultant. It goes beyond simple text retrieval to actually **compute sentencing ranges**, analyze criminal liability, and simulate the judicial decision-making process for specific criminal offenses.

## Key Features

* **Deep Penal Code Specialization:** Specifically fine-tuned to understand criminal terminology and logic, such as distinguishing between *Theft*, *Robbery*, and *Embezzlement*.
* **Intelligent Sentencing Logic:**
    * **Age-Based Liability:** Automatically detects if the offender or victim is under 18 to apply specific leniency or aggravating circumstances (e.g., distinct sentencing frameworks for minors vs. adults).
    * **Penalty Calculation:** Weighs aggravating factors against mitigating factors to recommend precise jail terms or monetary fines based on the net liability.


* **Drug & Economic Crime Modules:**
    * Contains specific logic for complex crimes like **Drug Trafficking** (distinguishing "possession" vs. "organizing use").
    * **Usury** (Cho vay lãi nặng), prioritizing financial penalties where legally appropriate.


* **Multi-Perspective Simulation:**
    * **The Judge:** Delivers a cold, fact-based verdict focusing on the exact article and clause.
    * **Defense Attorney:** Scans the database for every possible mitigating factor to argue for a suspended sentence (Án treo).
    * **Victim's Lawyer:** Focuses on maximum penalty frameworks and civil compensation claims.


* **Hallucination Control:** Uses a "Time-Aware" rewriting engine to ensure the advice is based on the law *effective at the time of the crime*, preventing errors when laws overlap between the 1999 and 2025 codes.

## Technology Stack

* **Core Logic:** `LangChain` & `LangGraph` for decision trees (Retrieval -> Grading -> Sentencing).
* **Vector Database:** `Milvus` storing thousands of Vietnamese criminal judgments and precedents.
* **Model:** `Sentence-Transformers` with a **Custom LoRA Adapter** trained on penal case files.
* **Backend:** `FastAPI` (Python).
* **Frontend:** `Streamlit`.
* **Infrastructure:** GPU-accelerated environment (NVIDIA CUDA).

# Setup
## Server-hosting
Cd into the folder containing the code.

Update pip first to avoid resolver issues:
```
pip install --upgrade pip
```

Install project requirements:
```
pip install -r requirements.txt
```

Run the server:
```
python3 server.py
```

Set up Open Router API key Hugging Face key: In terminal run these commands:
```
export OPENROUTER_API_KEY=[your_key_here]

export HF_TOKEN=[your_key_here]
```

## Client-side
In terminal run this command:
```
streamlit run frontend.py
```