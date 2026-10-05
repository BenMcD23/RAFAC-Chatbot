from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import rag

rag.get_model()  # load at startup so the device is logged and the first question isn't slow
app = FastAPI()


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def index():
    return FileResponse(rag.HERE / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": rag.get_collection().count()}


@app.post("/ask")
def ask(body: Ask):
    return rag.answer(body.question.strip())
