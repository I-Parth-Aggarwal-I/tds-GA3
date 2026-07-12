import base64
import io
import os
import re
from typing import Any

import pandas as pd
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


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def decode_audio_base64(audio_base64: str) -> bytes:
    """
    Defensively decode base64 audio:
    - strips data URI prefix if present (e.g. "data:audio/wav;base64,....")
    - strips whitespace/newlines
    - fixes missing padding
    """
    audio_base64 = audio_base64.strip()

    if audio_base64.lower().startswith("data:") and "," in audio_base64:
        audio_base64 = audio_base64.split(",", 1)[1]

    audio_base64 = re.sub(r"\s+", "", audio_base64)

    missing_padding = len(audio_base64) % 4
    if missing_padding:
        audio_base64 += "=" * (4 - missing_padding)

    return base64.b64decode(audio_base64)


def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"  # whisper API needs a filename hint
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        # language="ko",  # uncomment if audio is always Korean
    )
    return resp.text


def parse_transcript_to_dataframe(transcript: str) -> pd.DataFrame:
    """
    PLACEHOLDER — replace once we know the real transcript pattern.
    Currently: extracts every number mentioned in the transcript into
    a single-column dataframe called "value".
    """
    numbers = re.findall(r"-?\d+\.?\d*", transcript)
    values = [float(n) for n in numbers]
    df = pd.DataFrame({"value": values})
    return df


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    def col_dict(func):
        return {c: func(df[c]) for c in numeric_cols}

    def safe_mode(series: pd.Series):
        m = series.mode()
        return float(m.iloc[0]) if not m.empty else None

    result: dict[str, Any] = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "mean": col_dict(lambda s: float(s.mean()) if len(s) else 0.0),
        "std": col_dict(lambda s: float(s.std()) if len(s) > 1 else 0.0),
        "variance": col_dict(lambda s: float(s.var()) if len(s) > 1 else 0.0),
        "min": col_dict(lambda s: float(s.min()) if len(s) else 0.0),
        "max": col_dict(lambda s: float(s.max()) if len(s) else 0.0),
        "median": col_dict(lambda s: float(s.median()) if len(s) else 0.0),
        "mode": {c: safe_mode(df[c]) for c in numeric_cols},
        "range": col_dict(lambda s: float(s.max() - s.min()) if len(s) else 0.0),
        "allowed_values": {
            c: sorted(df[c].dropna().unique().tolist()) for c in numeric_cols
        },
        "value_range": col_dict(
            lambda s: [float(s.min()), float(s.max())] if len(s) else [0.0, 0.0]
        ),
        "correlation": (
            df[numeric_cols].corr().round(4).to_dict()
            if len(numeric_cols) > 1
            else []
        ),
    }
    return result


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/answer-audio")
def answer_audio(req: AudioRequest):
    audio_bytes = decode_audio_base64(req.audio_base64)
    transcript = transcribe_audio(audio_bytes)
    df = parse_transcript_to_dataframe(transcript)
    return compute_stats(df)
