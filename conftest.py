"""Shared test setup: no torch, no Hugging Face, no LLM provider.

The image and the GPU PC have torch and the models; CI and a fresh checkout may not,
and tests must never download anything. So torch and sentence_transformers are faked
when missing, and every test gets a word-hashing embedder in place of bge (similar
wording -> similar vectors, which is all retrieval tests need), no reranker, and a
fresh Chroma index in a temp folder.
"""

import re
import sys
import types
import zlib

import numpy as np
import pytest

try:
    import sentence_transformers  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=None, CrossEncoder=None)

import rag  # noqa: E402

_WORD = re.compile(r"\S+")


class FakeTokenizer:
    """One token per whitespace-separated word, with the offsets a fast tokenizer returns."""

    def __call__(self, text, **_):
        return {"offset_mapping": [m.span() for m in _WORD.finditer(text)]}


class FakeModel:
    tokenizer = FakeTokenizer()

    def encode(self, texts, **_):
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower().replace(rag.QUERY_PREFIX.lower(), "")):
                out[i, zlib.crc32(w.encode()) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "get_model", lambda: FakeModel())
    monkeypatch.setattr(rag, "get_reranker", lambda: None)
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    rag.get_collection.cache_clear()
    yield
    rag.get_collection.cache_clear()
