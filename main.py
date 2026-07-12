import base64
import json
import os
import tempfile

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


SYSTEM_PROMPT = """
You will receive a transcript of a spoken dataset.

Infer the complete dataset.

Return ONLY JSON having exactly these keys:

rows
columns
mean
std
variance
min
max
median
mode
range
allowed_values
value_range
correlation

Do NOT return markdown.
Compute every statistic exactly.
"""

@app.post("/analyse")
@app.post("/analyze")
async def analyze(req: AudioRequest):

    audio = base64.b64decode(req.audio_base64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
        path = f.name

    try:
        with open(path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            ).text
    except Exception as e:
        return {"error": str(e)}

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ]
    )

    os.remove(path)

    return json.loads(response.choices[0].message.content)