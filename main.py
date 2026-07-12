from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()


class ExtractRequest(BaseModel):
    text: str
    schema: dict


SYSTEM_PROMPT = """
You are an information extraction engine.

You will receive:

1. Text
2. A schema describing required fields and types.

Rules:

- Return EXACTLY the keys in schema.
- Never add extra keys.
- Missing values become null.
- integer -> JSON integer
- float -> JSON number
- boolean -> true/false
- date -> YYYY-MM-DD
- array[string] -> JSON array
- array[integer] -> JSON array of integers

Return ONLY JSON.
"""


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/dynamic-extract")
def extract(req: ExtractRequest):

    schema_description = json.dumps(req.schema, indent=2)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
f"""TEXT

{req.text}

SCHEMA

{schema_description}

Return JSON only."""
            }
        ]
    )

    return json.loads(response.choices[0].message.content)