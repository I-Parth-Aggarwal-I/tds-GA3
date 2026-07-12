from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()


class RankRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/rank")
def rank(req: RankRequest):
    texts = [req.query] + req.candidates

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [x.embedding for x in response.data]

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    scores = [
        cosine(query_embedding, emb)
        for emb in candidate_embeddings
    ]

    top3 = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    return {
        "ranking": top3
    }