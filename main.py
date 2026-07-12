import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()


class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice",
                "schema": req.schema,
            },
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract invoice information from the document. "
                    "Return ONLY JSON matching the provided schema exactly. "
                    "Normalize dates, currencies, numbers and booleans as requested."
                ),
            },
            {
                "role": "user",
                "content": req.text,
            },
        ],
    )

    return json.loads(response.choices[0].message.content)