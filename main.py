import os
import json
import random

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq


# =========================================================
# Z STUDIO AI SERVER
# =========================================================

app = FastAPI(
    title="Z Studio AI Assistant Server",
    version="2.3.0"
)


# =========================================================
# GROQ
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Có thể đổi model trực tiếp trên Render bằng Environment Variable
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("[OK] Groq API configured")
else:
    groq_client = None
    print("[WARN] GROQ_API_KEY not configured")


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


    # =====================================================
    # 1. CHAT / CONVERSATION CONTEXT
    # =====================================================

    if isinstance(chat_data, dict):

        intents = chat_data.get("intents", [])

        if isinstance(intents, list):

            for intent in intents:

                if not isinstance(intent, dict):
                    continue

                keywords = intent.get("keywords", [])

                if not isinstance(keywords, list):
                    continue

                for keyword in keywords:

                    if not isinstance(keyword, str):
                        continue

                    if keyword.lower() in query_lower:

                        responses = intent.get(
                            "responses",
                            ["Hả? Nói lại nghe xem nào."]
                        )

                        if responses:

                            return random.choice(responses)


    # =====================================================
    # 2. KNOWLEDGE DATABASE
    # =====================================================

    best_match = None
    best_score = 0


    for category_name, data in KNOWLEDGE_BASES.items():

        if not isinstance(data, dict):
            continue


        # -------------------------------------------------
        # Tìm tất cả list trong JSON
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
    # GROQ FALLBACK
    # =====================================================

    if groq_client:

        try:

            system_instruction = """
Bạn là Z - trợ lý AI cá nhân của chủ nhân.

Phong cách:
- Nói chuyện tự nhiên bằng tiếng Việt.
- Xưng hô thân mật kiểu tao/mày khi phù hợp.
- Có thể hơi cộc và cà khịa nhẹ.
- Trung thành với chủ nhân.
- Trả lời ngắn gọn nhưng hữu ích.
- Không nói dài dòng nếu không cần thiết.
- Nếu câu hỏi cần hướng dẫn kỹ thuật thì giải thích rõ từng bước.
- Am hiểu Gundam, Metal Kit, mô hình, 3D printing,
  airbrush, IoT, ESP32, điện tử, xe cộ và kỹ thuật.

Quan trọng:
- Không bịa thông tin nếu không chắc chắn.
- Nếu không biết, nói rõ là không biết.
- Bộ nhớ local của server được ưu tiên trước AI.
"""


            response = groq_client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[

                    {
                        "role": "system",
                        "content": system_instruction
                    },

                    {
                        "role": "user",
                        "content": user_query
                    }

                ],

                temperature=0.7,

                max_tokens=500

            )


            ai_text = response.choices[0].message.content


            if ai_text:

                print(
                    f"[GROQ:{GROQ_MODEL}] "
                    + user_query
                )

                return {

                    "source": "groq_fallback",

                    "response": ai_text.strip()

                }


            return {

                "source": "server_error",

                "response": "Groq không trả về dữ liệu."

            }


        except Exception as e:

            print(f"[GROQ ERROR] {e}")

            return {

                "source": "server_error",

                "response": (
                    "Hệ thống AI đang nghẽn rồi: "
                    + str(e)
                )

            }


    # =====================================================
    # NO GROQ
    # =====================================================

    return {

        "source": "server_local",

        "response": (
            "Tao không tìm thấy dữ liệu trong bộ nhớ local "
            "và Groq API Key cũng chưa được cấu hình."
        )

    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {

        "status": "Z Studio Server is online and ready!",

        "version": "2.3.0",

        "ai": "groq" if groq_client else "none"

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
