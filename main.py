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
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    invoice_text: str


SYSTEM_PROMPT = """
Extract invoice information.

Return ONLY valid JSON with EXACTLY these six keys:

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
- amount = subtotal BEFORE tax
- tax = tax amount only
- currency should be ISO code such as INR, USD, EUR
- Missing fields must be null.
- No markdown.
"""


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: InvoiceRequest):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.invoice_text},
        ],
    )

    result = json.loads(response.choices[0].message.content)

    # Ensure all required keys exist
    output = {
        "invoice_no": result.get("invoice_no"),
        "date": result.get("date"),
        "vendor": result.get("vendor"),
        "amount": result.get("amount"),
        "tax": result.get("tax"),
        "currency": result.get("currency"),
    }

    return output
