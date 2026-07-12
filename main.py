import base64
import binascii
import json
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
You are given the transcript of a spoken dataset.

Infer the dataset completely.

Return ONLY valid JSON having EXACTLY these keys:

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

No markdown.
No explanations.
"""


def decode_audio(audio_base64: str) -> bytes:
    """
    Robust base64 decoder.
    Handles:
    - data:audio/...;base64,...
    - missing padding
    """

    if "," in audio_base64:
        audio_base64 = audio_base64.split(",", 1)[1]

    audio_base64 = audio_base64.strip()

    missing = len(audio_base64) % 4
    if missing:
        audio_base64 += "=" * (4 - missing)

    return base64.b64decode(audio_base64)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/analyse")
@app.post("/analyze")
async def analyze(req: AudioRequest):

    try:
        audio_bytes = decode_audio(req.audio_base64)
    except binascii.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 audio: {e}"
        )

    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )

        transcript = transcription.text

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ]
        )

        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
