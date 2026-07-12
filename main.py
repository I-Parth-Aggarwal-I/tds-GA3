import base64
import io
import json
import os
import re
import statistics
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


def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"  # whisper API needs a filename hint
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        # language="ko",  # uncomment if you know it's always Korean
    )
    return resp.text


def parse_transcript_to_dataframe(transcript: str) -> pd.DataFrame:
    """
    TODO: This is the part that depends on what the audio actually says.
    Placeholder: extracts every number mentioned in the transcript into
    a single-column dataframe. Replace once you know the real pattern
    (e.g. multiple named columns, row-by-row dictation, etc.)
    """
    numbers = re.findall(r"-?\d+\.?\d*", transcript)
    values = [float(n) for n in numbers]
    df = pd.DataFrame({"value": values})
    return df


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    def col_dict(func):
        return {c: func(df[c]) for c in numeric_cols}

    def safe_mode(series):
        m = series.mode()
        return float(m.iloc[0]) if not m.empty else None

    result = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "mean": col_dict(lambda s: float(s.mean())),
        "std": col_dict(lambda s: float(s.std()) if len(s) > 1 else 0.0),
        "variance": col_dict(lambda s: float(s.var()) if len(s) > 1 else 0.0),
        "min": col_dict(lambda s: float(s.min())),
        "max": col_dict(lambda s: float(s.max())),
        "median": col_dict(lambda s: float(s.median())),
        "mode": {c: safe_mode(df[c]) for c in numeric_cols},
        "range": col_dict(lambda s: float(s.max() - s.min())),
        "allowed_values": {c: sorted(df[c].unique().tolist()) for c in numeric_cols},
        "value_range": col_dict(lambda s: [float(s.min()), float(s.max())]),
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
    audio_bytes = base64.b64decode(req.audio_base64)
    transcript = transcribe_audio(audio_bytes)
    df = parse_transcript_to_dataframe(transcript)
    return compute_stats(df)
