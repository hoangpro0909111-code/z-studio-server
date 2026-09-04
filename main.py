import os
import json
import random

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai


# =========================================================
# Z STUDIO AI SERVER
# =========================================================

app = FastAPI(
    title="Z Studio AI Assistant Server",
    version="2.2.0"
)


# =========================================================
# GEMINI
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None


# =========================================================
# LOAD JSON
# =========================================================

DATA_DIR = "."


def load_json_file(filename):
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"[OK] Loaded: {filename}")
            return data

        except Exception as e:
            print(f"[ERROR] Cannot read {filename}: {e}")
            return {}

    print(f"[WARN] File not found: {filename}")
    return {}


# =========================================================
# LOAD ALL KNOWLEDGE
# =========================================================

chat_data = load_json_file("chat_contexts.json")

mecha_data = load_json_file("knowledge_mecha.json")
iot_data = load_json_file("knowledge_iot.json")
crochet_data = load_json_file("knowledge_crochet.json")
gemstones_data = load_json_file("knowledge_gemstones.json")
pets_data = load_json_file("knowledge_pets.json")
plants_data = load_json_file("knowledge_plants.json")
vehicles_data = load_json_file("knowledge_vehicles.json")


# =========================================================
# KNOWLEDGE BASE
# =========================================================

KNOWLEDGE_BASES = {
    "mecha": mecha_data,
    "iot": iot_data,
    "crochet": crochet_data,
    "gemstones": gemstones_data,
    "pets": pets_data,
    "plants": plants_data,
    "vehicles": vehicles_data,
}


# =========================================================
# LOCAL SEARCH
# =========================================================

def search_local_knowledge(query: str):
    query_lower = query.lower()

    # -----------------------------------------------------
    # 1. CHAT / CONVERSATION CONTEXT
    # -----------------------------------------------------

    if isinstance(chat_data, dict):

        intents = chat_data.get("intents", [])

        if isinstance(intents, list):

            for intent in intents:

                keywords = intent.get("keywords", [])

                for keyword in keywords:

                    if keyword.lower() in query_lower:

                        responses = intent.get(
                            "responses",
                            ["Hả? Nói lại nghe xem nào."]
                        )

                        if responses:
                            return random.choice(responses)


    # -----------------------------------------------------
    # 2. KNOWLEDGE DATABASE
    # -----------------------------------------------------

    best_match = None
    best_score = 0

    for category_name, data in KNOWLEDGE_BASES.items():

        if not isinstance(data, dict):
            continue

        # -------------------------------------------------
        # Tìm list dữ liệu trong JSON
        # -------------------------------------------------

        knowledge_list = []

        for key, value in data.items():

            if isinstance(value, list):
                knowledge_list.extend(value)

        # -------------------------------------------------
        # Search từng item
        # -------------------------------------------------

        for item in knowledge_list:

            if not isinstance(item, dict):
                continue

            keywords = item.get("keywords", [])

            if not isinstance(keywords, list):
                continue

            score = 0

            for keyword in keywords:

                if not isinstance(keyword, str):
                    continue

                if keyword.lower() in query_lower:
                    score += 1

            if score > best_score:

                content = item.get("content")

                if content:
                    best_score = score
                    best_match = content

    return best_match


# =========================================================
# REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):
    prompt: str


# =========================================================
# ASK Z
# =========================================================

@app.post("/ask")
async def ask_z(req: QueryRequest):

    user_query = req.prompt.strip()

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )


    # =====================================================
    # LOCAL KNOWLEDGE FIRST
    # =====================================================

    local_answer = search_local_knowledge(user_query)

    if local_answer:

        print("[LOCAL] " + user_query)

        return {
            "source": "server_local",
            "response": local_answer
        }


    # =====================================================
    # GEMINI FALLBACK
    # =====================================================

    if gemini_model:

        try:

            system_instruction = """
Bạn là Z - trợ lý AI cá nhân của chủ nhân.

Phong cách:
- Nói chuyện tự nhiên bằng tiếng Việt.
- Có thể hơi cộc, cà khịa nhẹ và thân mật.
- Trung thành với chủ nhân.
- Trả lời ngắn gọn nhưng hữu ích.
- Không nói dài dòng nếu không cần thiết.
- Nếu câu hỏi cần hướng dẫn kỹ thuật thì giải thích rõ từng bước.
- Am hiểu Gundam, Metal Kit, mô hình, 3D printing,
  airbrush, IoT, ESP32, điện tử, xe cộ và kỹ thuật.
"""

            full_prompt = (
                system_instruction
                + "\n\n"
                + "Chủ nhân hỏi:\n"
                + user_query
            )

            response = gemini_model.generate_content(full_prompt)

            if response and response.text:

                print("[GEMINI] " + user_query)

                return {
                    "source": "gemini_fallback",
                    "response": response.text.strip()
                }

            return {
                "source": "server_error",
                "response": "Gemini không trả về dữ liệu."
            }

        except Exception as e:

            print(f"[GEMINI ERROR] {e}")

            return {
                "source": "server_error",
                "response": f"Hệ thống mây đang nghẽn rồi: {str(e)}"
            }


    # =====================================================
    # NO GEMINI
    # =====================================================

    return {
        "source": "server_local",
        "response": (
            "Tao không tìm thấy dữ liệu trong bộ nhớ local "
            "và Gemini API Key cũng chưa được cấu hình."
        )
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "Z Studio Server is online and ready!",
        "version": "2.2.0"
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
