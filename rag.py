import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
import torch  # noqa: E402
from openai import OpenAI, OpenAIError, RateLimitError  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag")
logging.getLogger("httpx").setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / os.path.expanduser(os.getenv("DOCS_DIR", "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or HERE / "data" / "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SHAREPOINT_BASE = "https://rafac.sharepoint.com/sites/interim/QM/Controlled%20Documents/"
NOT_FOUND = "I couldn't find that in the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources provided.
Rules:
- Answer ONLY from the provided sources. Do not use outside knowledge.
- Cite sources inline as [n] after each claim.
- If the sources don't contain the answer, reply exactly: "{NOT_FOUND}" Do not guess.
- If sources conflict or look like different versions of the same document, say so.
- Be concise.
- Write plain text, not Markdown: no **bold**, headings or tables. Use "- " for lists."""


@lru_cache
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    log.info("Embedding model %s on device: %s%s", EMBED_MODEL, device, name)
    try:
        return SentenceTransformer(EMBED_MODEL, device=device, local_files_only=True)
    except Exception:
        log.info("Model not cached, downloading from Hugging Face (one-time)")
        return SentenceTransformer(EMBED_MODEL, device=device)


def embed(texts, query=False):
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    return get_model().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"}, embedding_function=None)


def sharepoint_url(rel_path):
    return SHAREPOINT_BASE + quote(rel_path)


def location(meta):
    return f"page {meta['page']}" if "page" in meta else meta.get("section", "")


def retrieve(question, k=8):
    col = get_collection()
    if col.count() == 0:
        return []
    r = col.query(query_embeddings=embed([question], query=True), n_results=k)
    return [
        {"text": doc, "meta": meta, "score": round(1 - dist, 4)}
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


def _chat(messages, retries=5):
    api_key = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise OpenAIError("no API key set; put LLM_API_KEY or NVIDIA_API_KEY in .env")
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
        max_retries=0,
        timeout=120,
    )
    model = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.1,
                                               extra_body=json.loads(os.getenv("LLM_EXTRA_BODY") or "{}"))
            return (r.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            log.warning("Rate limited (429), retrying in %ss", 2**attempt)
            time.sleep(2**attempt)


def answer(question):
    chunks = retrieve(question)
    if not chunks:
        return {"answer": NOT_FOUND, "sources": []}

    sources, context = [], []
    for n, c in enumerate(chunks, 1):
        m = c["meta"]
        loc = location(m)
        sources.append({
            "n": n,
            "filename": m["filename"],
            "location": loc,
            "url": sharepoint_url(m["path"]),
            "snippet": c["text"][:240].replace("\n", " ") + ("…" if len(c["text"]) > 240 else ""),
            "score": c["score"],
        })
        context.append(f"[{n}] {m['filename']}{' (' + loc + ')' if loc else ''}\n{c['text']}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Sources:\n\n" + "\n\n---\n\n".join(context) + f"\n\nQuestion: {question}"},
    ]
    try:
        text = _chat(messages)
    except RateLimitError:
        text = "The LLM API is rate limiting requests. Try again in a minute."
    except OpenAIError as e:  # also raised when no API key is configured
        log.error("LLM call failed: %s", e)
        text = f"LLM request failed: {e}"
    return {"answer": text, "sources": sources}
