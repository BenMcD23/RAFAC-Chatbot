"""Run: python test_chunk.py  (checks chunk sizes, overlap and oversized-paragraph splitting)"""
from transformers import AutoTokenizer

from ingest import CHUNK_TOKENS, chunk

tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", local_files_only=True)
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

paras = [(f"Paragraph {i} " + "word " * 60, {"page": i // 5 + 1}) for i in range(40)]
paras.append(("huge " * 1500, {"page": 99}))  # one paragraph far bigger than a chunk
chunks = chunk(paras, tok)

assert all(n(text) <= CHUNK_TOKENS for text, _ in chunks), "chunk exceeds model limit"
assert chunks[0][1] == {"page": 1} and chunks[-1][1] == {"page": 99}
first_paras = [p for p in chunks[0][0].split("\n\n")]
assert first_paras[-1] in chunks[1][0], "no overlap carried into next chunk"
assert sum(t.count("huge") for t, m in chunks if m["page"] == 99) >= 1500, "oversized paragraph lost text"
print(f"ok: {len(chunks)} chunks, max {max(n(t) for t, _ in chunks)} tokens")
