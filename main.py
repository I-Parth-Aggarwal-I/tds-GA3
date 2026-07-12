import base64
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_base64: str
    question: str


SYSTEM_PROMPT = """
Answer the user's question using only the image.

Return ONLY JSON:

{
  "answer":"..."
}

Rules:
- answer must always be a string.
- If the answer is numeric, return only the number.
- Do not include currency symbols.
- Do not include units.
- Do not include markdown.
"""


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/answer-image")
def answer_image(req: ImageRequest):

    image_url = f"data:image/png;base64,{req.image_base64}"

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": req.question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)

    return {
        "answer": str(result.get("answer", ""))
    }
