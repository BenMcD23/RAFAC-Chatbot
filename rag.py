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
from openai import BadRequestError, OpenAI, OpenAIError, RateLimitError  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

import cite  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag")
logging.getLogger("httpx").setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / os.path.expanduser(os.getenv("DOCS_DIR", "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or HERE / "data" / "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# Empty disables reranking (results then stay in embedding order).
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SHAREPOINT_BASE = "https://rafac.sharepoint.com/sites/interim/QM/Controlled%20Documents/"
NOT_FOUND = "I couldn't find that in the documents."
# Vector search casts a wide net, the reranker picks the best few: fewer, better chunks in the
# prompt means better answers and fewer tokens against the provider's daily limit.
CANDIDATES, CONTEXT_CHUNKS = 25, 6
location = cite.location

SYSTEM_PROMPT = """You answer questions about RAFAC controlled documents using ONLY the numbered sources provided.

Reply with a JSON object and nothing else:
{"answer": "...", "citations": [{"source": 1, "quote": "..."}]}

answer:
- Use only the sources. Never use outside knowledge and never guess.
- Put [n] straight after each claim it supports, e.g. "Cadets must be at least 12 [2]."
- Open with the direct answer in one sentence, then the conditions, exceptions and limits the sources give.
- Plain text. For several items use lines starting "- ". No headings, tables or bold.
- If sources conflict or look like different versions of one document, say so and cite each.
- If the sources don't answer the question, answer is exactly "NOT_FOUND" and citations is [].

citations: one entry for every [n] used in the answer. quote is the shortest passage (a sentence or
two) from source n that supports the claim, copied character for character. Don't paraphrase,
shorten with "...", or correct it."""


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_model():
    device = _device()
    name = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    log.info("Embedding model %s on device: %s%s", EMBED_MODEL, device, name)
    try:
        return SentenceTransformer(EMBED_MODEL, device=device, local_files_only=True)
    except Exception:
        log.info("Model not cached, downloading from Hugging Face (one-time)")
        return SentenceTransformer(EMBED_MODEL, device=device)


@lru_cache
def get_reranker():
    """The cross-encoder, or None: a missing reranker costs answer quality, never the answer."""
    if not RERANK_MODEL:
        return None
    try:
        return CrossEncoder(RERANK_MODEL, device=_device(), max_length=512, local_files_only=True)
    except Exception:
        try:
            log.info("Reranker not cached, downloading from Hugging Face (one-time)")
            return CrossEncoder(RERANK_MODEL, device=_device(), max_length=512)
        except Exception as e:
            log.warning("Reranker %s unavailable, using embedding order: %s", RERANK_MODEL, e)
            return None


def embed(texts, query=False):
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    return get_model().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()


def passage(meta, text):
    """Chunk text with its document title and section on top, for embedding and reranking.

    A chunk on its own often never names its document ("Cadets must not..."), so a question
    about "the ACP 20 rules" would otherwise not match it.
    """
    title = Path(meta["filename"]).stem.replace("_", " ")
    return "\n".join(x for x in (title, meta.get("section") or "") if x) + "\n\n" + text


@lru_cache
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"}, embedding_function=None)


def sharepoint_url(rel_path, page=None):
    # Browser PDF viewers honour #page; anything that doesn't simply opens the document at the top.
    return SHAREPOINT_BASE + quote(rel_path) + (f"#page={page}" if page and rel_path.lower().endswith(".pdf") else "")


def retrieve(question, k=CONTEXT_CHUNKS, candidates=CANDIDATES):
    col = get_collection()
    count = col.count()
    if count == 0:
        return []
    r = col.query(query_embeddings=embed([question], query=True), n_results=min(max(k, candidates), count))
    chunks = [
        {"text": doc, "meta": meta, "score": round(1 - dist, 4)}
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
    reranker = get_reranker()
    if reranker and len(chunks) > 1:
        scores = reranker.predict([(question, passage(c["meta"], c["text"])) for c in chunks], show_progress_bar=False)
        for c, s in zip(chunks, scores):
            c["score"] = round(float(s), 4)
        chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:k]


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
    extra = json.loads(os.getenv("LLM_EXTRA_BODY") or "{}")
    json_mode = True
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.1, extra_body=extra,
                **({"response_format": {"type": "json_object"}} if json_mode else {}))
            return (r.choices[0].message.content or "").strip()
        except BadRequestError as e:
            # Not every provider/model does JSON mode, and Groq rejects a reply that isn't valid
            # JSON. The prompt still asks for JSON and parse_reply copes with plain text.
            if not json_mode:
                raise
            log.warning("JSON mode rejected, retrying without it: %s", e)
            json_mode = False
        except RateLimitError:
            if attempt == retries - 1:
                raise
            log.warning("Rate limited (429), retrying in %ss", 2**attempt)
            time.sleep(2**attempt)


def _related(chunks, limit=3):
    """The best-matching distinct documents, offered when there's no answer to cite."""
    seen, out = set(), []
    for c in chunks:
        m = c["meta"]
        if m["path"] in seen:
            continue
        seen.add(m["path"])
        out.append({"filename": m["filename"], "location": location(m),
                    "url": sharepoint_url(m["path"], m.get("page"))})
        if len(out) == limit:
            break
    return out


def _source(n, chunk, quote_spans):
    m, text = chunk["meta"], chunk["text"]
    try:
        blocks = json.loads(m.get("spans") or "[]")
    except ValueError:
        blocks = []
    quotes = []
    for start, end in quote_spans:
        block = cite.span_at(blocks, start)
        quotes.append({"start": start, "end": end, "text": text[start:end],
                       "location": block[3] if block else location(m),
                       "page": (block[2] if block else m.get("page")) or None})
    first = quotes[0] if quotes else {"location": location(m), "page": m.get("page")}
    snippet = quotes[0]["text"] if quotes else text
    return {
        "n": n,
        "filename": m["filename"],
        "location": first["location"],
        "url": sharepoint_url(m["path"], first["page"]),
        "snippet": snippet[:240].replace("\n", " ") + ("…" if len(snippet) > 240 else ""),
        "score": chunk["score"],
        "text": text,
        "quotes": quotes,
    }


def answer(question):
    """{answer, found, sources, related}.

    answer cites [n]; sources are only the ones it cites, renumbered in order of use, each
    with the chunk text and the verified quotes in it (character offsets plus page/para).
    related lists the nearest documents when there's no answer.
    """
    chunks = retrieve(question)
    if not chunks:
        return {"answer": NOT_FOUND, "found": False, "sources": [], "related": []}

    context = []
    for n, c in enumerate(chunks, 1):
        loc = location(c["meta"])
        context.append(f"[{n}] {c['meta']['filename']}{' (' + loc + ')' if loc else ''}\n{c['text']}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Sources:\n\n" + "\n\n---\n\n".join(context) + f"\n\nQuestion: {question}"},
    ]
    try:
        reply = _chat(messages)
    except RateLimitError:
        return {"answer": "The AI service is busy right now. Try again in a minute.",
                "found": False, "sources": [], "related": _related(chunks)}
    except OpenAIError as e:  # also raised when no API key is configured
        log.error("LLM call failed: %s", e)
        return {"answer": "The AI service couldn't answer just now. Try again shortly.",
                "found": False, "sources": [], "related": _related(chunks)}

    text, citations = cite.parse_reply(reply)
    if not text or "NOT_FOUND" in text or text.replace("’", "'").startswith(NOT_FOUND[:20]):
        return {"answer": NOT_FOUND, "found": False, "sources": [], "related": _related(chunks)}

    valid = range(1, len(chunks) + 1)
    spans = {n: [] for n in valid}
    for n, q in citations:
        if n in spans:
            span = cite.find_quote(q, chunks[n - 1]["text"])
            if span and not any(s[0] < span[1] and span[0] < s[1] for s in spans[n]):
                spans[n].append(span)
            elif not span:
                log.info("Dropped unverifiable quote for [%s]: %.80s", n, q)
    # A source counts when the answer cites it; if the model left out the [n]s, fall back to its quotes.
    order = cite.cited_order(text, valid) or [n for n in valid if spans[n]]
    mapping = {old: new for new, old in enumerate(order, 1)}
    return {
        "answer": cite.renumber(text, mapping),
        "found": True,
        "sources": [_source(mapping[old], chunks[old - 1], sorted(spans[old])) for old in order],
        "related": [] if order else _related(chunks),
    }
