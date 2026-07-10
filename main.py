import os
import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

MODEL = "gpt-4.1-mini"


class InvoiceRequest(BaseModel):
    invoice_text: str


SYSTEM_PROMPT = """
You are an invoice extraction engine.

Extract the following fields.

Return ONLY valid JSON.

{
  "invoice_no": string|null,
  "date": string|null,
  "vendor": string|null,
  "amount": number|null,
  "tax": number|null,
  "currency": string|null
}

Rules:
- date must be YYYY-MM-DD
- amount is subtotal before tax
- tax is only the tax amount
- Return null if a field is missing
- No markdown
"""


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/extract")
def extract(req: InvoiceRequest):

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": req.invoice_text,
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)

    return {
        "invoice_no": result.get("invoice_no"),
        "date": result.get("date"),
        "vendor": result.get("vendor"),
        "amount": result.get("amount"),
        "tax": result.get("tax"),
        "currency": result.get("currency"),
    }