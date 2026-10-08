import os
import re
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Any]] = []


@app.get("/")
def read_root():
    return {"status": "ok", "source": "typhoon_api"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วย มีอะไรให้ช่วยไหมครับ",
        }

    history_messages = []
    if req.history:
        for item in req.history[-6:]:
            if isinstance(item, dict):
                role = "assistant" if item.get("role") in ["bot", "model", "assistant"] else "user"
                content = item.get("text", "")
                if content:
                    history_messages.append({"role": role, "content": content})

    messages_payload = [
        {"role": "system", "content": "คุณคือ AI ผู้ช่วย ตอบคำถามอย่างถูกต้อง ชัดเจน และตรงไปตรงมา หากไม่ทราบให้บอกตามตรง"},
        *history_messages,
        {"role": "user", "content": user_msg},
    ]

    api_key = os.getenv("TYPHOON_API_KEY", "").strip()
    if not api_key:
        return {
            "source": "configuration_error",
            "reply": "ระบบยังไม่ได้ตั้งค่า API Key กรุณาตั้งค่า TYPHOON_API_KEY ก่อนใช้งาน",
        }

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
        response = client.chat.completions.create(
            model="typhoon-v2.5-30b-a3b-instruct",
            messages=messages_payload,
            temperature=0.4,
            max_tokens=1000,
        )
        return {"source": "typhoon", "reply": response.choices[0].message.content}
    except Exception as exc:
        print(f"Typhoon API error: {exc}")
        return {
            "source": "fallback",
            "reply": "ขออภัย เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI",
        }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
