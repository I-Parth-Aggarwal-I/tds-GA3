import base64
import io
import re
import tempfile
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup, not per-request.
# "base" is a good speed/accuracy tradeoff for CPU deployment (e.g. Render free tier).
# compute_type="int8" keeps memory/CPU usage low.
# Bump to "small" or "medium" if accuracy is poor and your instance can handle it.
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


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
    """
    Transcribes audio using a local faster-whisper model.
    Writes to a temp file since faster-whisper expects a file path (or
    file-like object) it can seek on.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, info = whisper_model.transcribe(
            tmp.name,
            language="ko",  # force Korean; remove/auto-detect if clips aren't all Korean
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def parse_transcript_to_dataframe(transcript: str) -> pd.DataFrame:
    """
    PLACEHOLDER / BEST-EFFORT PARSER.

    We don't yet know the exact dictation pattern used in the audio clips.
    This function tries a few common patterns in order, and falls back to
    "every number becomes one row of a single 'value' column" if nothing
    else matches. Once you have a real transcript, replace this with exact
    parsing logic matching the real pattern.

    Pattern 1 tried: "<column>: <number>, <column>: <number>, ..." repeated
    per row, with rows separated by some delimiter (e.g. "행 1", "row 1",
    newlines, or numbered markers). This handles multi-column dictation.

    Pattern 2 fallback: just extract every number found, single column.
    """
    # Try to find "label: number" pairs anywhere in the text
    pairs = re.findall(r"([\w\uAC00-\uD7A3]+)\s*[:：]\s*(-?\d+\.?\d*)", transcript)

    if pairs:
        # Group into rows: assume the pattern repeats per row, so if we see
        # the same label twice, that's the start of a new row.
        rows: list[dict[str, float]] = []
        current_row: dict[str, float] = {}
        seen_labels_in_row: set[str] = set()

        for label, value in pairs:
            if label in seen_labels_in_row:
                rows.append(current_row)
                current_row = {}
                seen_labels_in_row = set()
            current_row[label] = float(value)
            seen_labels_in_row.add(label)

        if current_row:
            rows.append(current_row)

        if rows:
            return pd.DataFrame(rows)

    # Fallback: flat list of numbers
    numbers = re.findall(r"-?\d+\.?\d*", transcript)
    values = [float(n) for n in numbers]
    return pd.DataFrame({"value": values})


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
    print(f"TRANSCRIPT for {req.audio_id}: {transcript}")  # visible in server logs
    df = parse_transcript_to_dataframe(transcript)
    return compute_stats(df)