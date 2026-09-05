import os
import json
import random
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq


# =========================================================
# Z STUDIO AI SERVER
# VERSION 2.5.0
# /ask  -> Z AI
# /stt  -> Groq Whisper
# =========================================================

app = FastAPI(
    title="Z Studio AI Assistant Server",
    version="2.5.0"
)


# =========================================================
# GROQ
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_REQUESTED = os.getenv("GROQ_MODEL", "").strip()

groq_client = None
GROQ_MODEL = None
STT_MODEL = None


# =========================================================
# FIND AVAILABLE GROQ TEXT MODEL
# =========================================================

def find_groq_model():
    global groq_client
    global GROQ_MODEL

    if not GROQ_API_KEY:
        print("[WARN] GROQ_API_KEY not configured")
        return

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("[OK] Groq API configured")

        models_response = groq_client.models.list()

        available_models = []

        for model in models_response.data:
            model_id = getattr(model, "id", "")
            if model_id:
                available_models.append(model_id)

        print(
            f"[GROQ] Available models: "
            f"{len(available_models)}"
        )

        for model_id in available_models:
            print(f"[GROQ MODEL] {model_id}")

        # -------------------------------------------------
        # 1. Configured model
        # -------------------------------------------------

        if GROQ_MODEL_REQUESTED:
            if GROQ_MODEL_REQUESTED in available_models:
                GROQ_MODEL = GROQ_MODEL_REQUESTED
                print(
                    f"[GROQ] Using configured model: "
                    f"{GROQ_MODEL}"
                )
                return
            else:
                print(
                    f"[GROQ] Configured model unavailable: "
                    f"{GROQ_MODEL_REQUESTED}"
                )
                print(
                    "[GROQ] Automatically selecting "
                    "an available model..."
                )

        # -------------------------------------------------
        # 2. Preferred models
        # -------------------------------------------------

        preferred_models = [
            "llama-4-maverick-17b-128e-instruct",
            "llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            "qwen3-32b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

        for preferred in preferred_models:
            if preferred in available_models:
                GROQ_MODEL = preferred
                print(
                    f"[GROQ] Auto-selected model: "
                    f"{GROQ_MODEL}"
                )
                return

        # -------------------------------------------------
        # 3. Generic text/instruct fallback
        # -------------------------------------------------

        excluded_words = [
            "whisper",
            "guard",
            "safety",
            "tts",
            "speech",
            "audio",
        ]

        for model_id in available_models:
            model_lower = model_id.lower()

            if any(
                word in model_lower
                for word in excluded_words
            ):
                continue

            if (
                "instruct" in model_lower
                or "llama" in model_lower
                or "qwen" in model_lower
                or "gpt" in model_lower
            ):
                GROQ_MODEL = model_id
                print(
                    f"[GROQ] Fallback selected model: "
                    f"{GROQ_MODEL}"
                )
                return

        print(
            "[GROQ ERROR] "
            "No suitable text model found."
        )
        GROQ_MODEL = None

    except Exception as e:
        print(f"[GROQ INIT ERROR] {e}")
        groq_client = None
        GROQ_MODEL = None


# =========================================================
# FIND GROQ STT MODEL
# =========================================================

def find_groq_stt_model():
    global groq_client
    global STT_MODEL

    if not groq_client:
        print("[STT] Groq client chưa sẵn sàng")
        return

    try:
        models_response = groq_client.models.list()

        available_models = []

        for model in models_response.data:
            model_id = getattr(model, "id", "")
            if model_id:
                available_models.append(model_id)

        preferred_stt_models = [
            "whisper-large-v3-turbo",
            "whisper-large-v3",
        ]

        for preferred in preferred_stt_models:
            if preferred in available_models:
                STT_MODEL = preferred
                print(
                    f"[STT] Using model: "
                    f"{STT_MODEL}"
                )
                return

        for model_id in available_models:
            if "whisper" in model_id.lower():
                STT_MODEL = model_id
                print(
                    f"[STT] Auto-selected model: "
                    f"{STT_MODEL}"
                )
                return

        print(
            "[STT ERROR] "
            "Không tìm thấy Groq Whisper model."
        )
        STT_MODEL = None

    except Exception as e:
        print(f"[STT INIT ERROR] {e}")
        STT_MODEL = None


# =========================================================
# INITIALIZE GROQ
# =========================================================

find_groq_model()
find_groq_stt_model()


# =========================================================
# LOAD JSON
# =========================================================

DATA_DIR = "."


def load_json_file(filename):
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            print(f"[OK] Loaded: {filename}")
            return data

        except Exception as e:
            print(
                f"[ERROR] Cannot read "
                f"{filename}: {e}"
            )
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
                            [
                                "Hả? Nói lại nghe xem nào."
                            ]
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

        knowledge_list = []

        for key, value in data.items():
            if isinstance(value, list):
                knowledge_list.extend(value)

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

    if groq_client and GROQ_MODEL:
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
        "source": "server_error",
        "response": "Không có Groq model khả dụng."
    }


# =========================================================
# SPEECH TO TEXT
# ESP32 gửi trực tiếp WAV binary
# =========================================================

@app.post("/stt")
async def speech_to_text(request: Request):

    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="Groq client chưa sẵn sàng"
        )

    if not STT_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Không tìm thấy Groq Whisper model"
        )

    try:
        audio_bytes = await request.body()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio body rỗng"
            )

        if len(audio_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio quá lớn"
            )

        print(
            f"[STT] Received WAV: "
            f"{len(audio_bytes)} bytes"
        )

        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=STT_MODEL,
            response_format="json",
            language="vi"
        )

        text = (
            getattr(transcription, "text", "") or ""
        ).strip()

        print(f"[STT:{STT_MODEL}] {text}")

        return {
            "source": "groq_stt",
            "model": STT_MODEL,
            "text": text
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"[STT ERROR] {e}")

        raise HTTPException(
            status_code=500,
            detail=f"STT error: {e}"
        )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "Z Studio Server is online and ready!",
        "version": "2.5.0",
        "ai": (
            "groq"
            if groq_client and GROQ_MODEL
            else "none"
        ),
        "model": GROQ_MODEL or "none",
        "stt": STT_MODEL or "none"
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
