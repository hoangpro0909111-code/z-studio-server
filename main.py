import os
import json
import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Z Studio AI Assistant Server", version="2.1.0")

# 1. Cấu hình Gemini API (Lấy từ biến môi trường của Render)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

# 2. Tự động nạp toàn bộ các file JSON dữ liệu trên server
DATA_DIR = "."
def load_json_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Lỗi đọc file {filename}: {e}")
    return {}

# Load các module dữ liệu rời rạc
chat_data = load_json_file("chat_contexts.json")
mecha_data = load_json_file("knowledge_mecha.json")
iot_data = load_json_file("knowledge_iot.json")
custom_data = load_json_file("knowledge_custom.json")
vehicles_pc_data = load_json_file("knowledge_vehicles_pc.json")
misc_data = load_json_file("knowledge_misc.json")

# 3. Thuật toán tra cứu nội bộ (Rule-based keyword matching)
def search_local_knowledge(query: str) -> str:
    query_lower = query.lower()
    
    # Kiểm tra kho ngữ cảnh giao tiếp (trả về ngẫu nhiên câu phản hồi cho sinh động)
    if "intents" in chat_data:
        for intent in chat_data["intents"]:
            for kw in intent.get("keywords", []):
                if kw in query_lower:
                    responses = intent.get("responses", ["Hả? Nói lại nghe xem nào."])
                    return random.choice(responses)

    # Gom toàn bộ kho kiến thức chuyên sâu để quét
    all_knowledge_lists = [
        mecha_data.get("mecha_knowledge", []),
        iot_data.get("iot_knowledge", []),
        custom_data.get("custom_knowledge", []),
        vehicles_pc_data.get("vehicles_pc_knowledge", []),
        misc_data.get("misc_knowledge", [])
    ]

    for k_list in all_knowledge_lists:
        for item in k_list:
            for kw in item.get("keywords", []):
                if kw in query_lower:
                    return item.get("content")

    return None

# 4. Định nghĩa cấu trúc request từ ESP32-S3
class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_z(req: QueryRequest):
    user_query = req.prompt.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Bước 1: Quét kho tĩnh trên server (0 đồng, tốc độ tính bằng mili-giây)
    local_answer = search_local_knowledge(user_query)
    if local_answer:
        return {
            "source": "server_local",
            "response": local_answer
        }

    # Bước 2: Fallback sang Gemini nếu câu hỏi lạ hoặc nằm ngoài kho dữ liệu
    if gemini_model:
        try:
            system_instruction = (
                "Bạn là 'Z' - một trợ lý AI cá nhân cộc lốc, hay cà khịa chủ nhân "
                "nhưng cực kỳ trung thành và am hiểu sâu sắc về Gundam, Metal Kit, IoT, xe cộ và kỹ thuật. "
                "Hãy trả lời ngắn gọn, sắc sảo."
            )
            full_prompt = f"{system_instruction}\n\nChủ nhân hỏi: {user_query}"
            resp = gemini_model.generate_content(full_prompt)
            return {
                "source": "gemini_fallback",
                "response": resp.text.strip()
            }
        except Exception as e:
            return {
                "source": "server_error",
                "response": f"Hệ thống mây đang nghẽn mạng rồi: {str(e)}"
            }

    return {
        "source": "server_local",
        "response": "Không tìm thấy dữ liệu trên server và chưa cấu hình Gemini API Key!"
    }

@app.get("/")
async def root():
    return {"status": "Z Studio Server is online and ready!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
