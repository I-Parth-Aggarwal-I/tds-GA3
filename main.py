import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests

load_dotenv()

TOKEN = os.getenv("AIPIPE_TOKEN")
print("TOKEN loaded:", TOKEN is not None)
print("Length:", len(TOKEN) if TOKEN else 0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestModel(BaseModel):
    image_base64: str
    question: str

class ResponseModel(BaseModel):
    answer: str


@app.get("/")
def home():
    return {"status":"running"}


@app.post("/answer-image", response_model=ResponseModel)
def answer(req: RequestModel):

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type":"application/json"
    }

    payload = {
        "model":"openai/gpt-4.1-mini",
        "messages":[
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":f"""
Answer the question from the image.

Question:
{req.question}

Rules:

Return ONLY the answer.

If numeric:
return only the number.

No explanation.
No currency.
No units.
"""
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:image/png;base64,{req.image_base64}"
                        }
                    }
                ]
            }
        ]
    }

    r = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )

    r.raise_for_status()

    answer = r.json()["choices"][0]["message"]["content"].strip()

    return {
        "answer":answer
    }