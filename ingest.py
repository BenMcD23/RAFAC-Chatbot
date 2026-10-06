import hashlib
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from fnmatch import fnmatch

import pymupdf
import yaml
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from tqdm import tqdm

import cite
import rag

# ponytail: 450 because bge-base truncates at 512 tokens and rag.passage() adds the title/section
# on top; raise if EMBED_MODEL has a longer context
CHUNK_TOKENS, OVERLAP_TOKENS = 450, 100
# Bump when what's stored per chunk changes, so the next ingest redoes every file rather than
# leaving unchanged files in the old shape. 2: block spans (page/para of every line), titled embeddings.
INDEX_VERSION = 2
SUPPORTED = {".pdf", ".docx"}
STATE_FILE = rag.HERE / "data" / "state.json"
log = logging.getLogger("ingest")


def pdf_blocks(path):
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("password-protected")
        number = None
        for page in doc:
            for b in page.get_text("blocks"):
                if b[6] != 0 or not b[4].strip():  # b[6] == 0 means a text block, not an image
                    continue
                text = b[4].strip()
                # A paragraph number set in the margin comes out as its own block; join it to its text.
                if cite.is_para_number(text):
                    if number:
                        yield number, {"page": page.number + 1, "para": cite.para_label(number)}
                    number = text
                    continue
                if number:
                    text, number = f"{number} {text}", None
                yield text, {"page": page.number + 1, "para": cite.para_label(text)}
        if number:
            yield number, {"page": doc.page_count, "para": cite.para_label(number)}


def is_heading(p):
    """Heading style, or (as most of these docs do) a short paragraph that is entirely bold."""
    style = p.style.name if p.style is not None else ""
    if style.startswith(("Heading", "Title")):
        return True
    runs = [r for r in p.runs if r.text.strip()]
    # ponytail: bold-line heuristic also catches TOC lines and bold one-liners; fine for a "section" hint
    return len(p.text.strip()) < 100 and bool(runs) and all(
        r.bold or (r.bold is None and p.style is not None and p.style.font.bold) for r in runs)


def cell_text(tc):
    return " ".join("".join(t.text or "" for t in p.iter(qn("w:t"))) for p in tc.iter(qn("w:p"))).strip()


def docx_blocks(path):
    section = ""
    for item in Document(path).iter_inner_content():
        if isinstance(item, Table):
            for row in item.rows:
                # Read the row's physical <w:tc> cells: row.cells crashes on malformed grids and repeats merged cells.
                cells = [t for t in (cell_text(tc) for tc in row._tr.tc_lst) if t]
                if cells:
                    yield " | ".join(cells), {"section": section}
            continue
        text = item.text.strip()
        if not text:
            continue
        if is_heading(item):
            section = text
        yield text, {"section": section, "para": cite.para_label(text)}


def chunk(blocks, tok):
    """Pack paragraph blocks into chunks of <= CHUNK_TOKENS, carrying ~OVERLAP_TOKENS of trailing blocks forward.

    Returns (text, meta, spans) per chunk. spans maps character ranges of text back to where
    they came from ([start, end, page or 0, "page 4, para 12"]), so a quote anywhere in the
    chunk can be pinned to its own page and paragraph, not just the chunk's first one.
    """
    pieces = []  # (text, meta, n_tokens)
    for text, meta in blocks:
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        if len(offs) <= CHUNK_TOKENS:
            pieces.append((text, meta, len(offs)))
            continue
        for i in range(0, len(offs), CHUNK_TOKENS - OVERLAP_TOKENS):  # oversized paragraph: token windows
            w = offs[i:i + CHUNK_TOKENS]
            pieces.append((text[w[0][0]:w[-1][1]], meta, len(w)))
            if i + CHUNK_TOKENS >= len(offs):
                break

    chunks, cur = [], []
    for p in pieces:
        if cur and sum(x[2] for x in cur) + p[2] > CHUNK_TOKENS:
            chunks.append(cur)
            keep, n = [], 0
            for x in reversed(cur):
                if n + x[2] > OVERLAP_TOKENS:
                    break
                keep.insert(0, x)
                n += x[2]
            cur = keep
            while cur and sum(x[2] for x in cur) + p[2] > CHUNK_TOKENS:
                cur.pop(0)
        cur.append(p)
    if cur:
        chunks.append(cur)
    return [("\n\n".join(x[0] for x in c), c[0][1], _spans(c)) for c in chunks]


def _spans(pieces):
    spans, pos = [], 0
    for text, meta, _ in pieces:
        loc = cite.location(meta)
        if spans and spans[-1][3] == loc and spans[-1][1] + 2 == pos:  # same paragraph, carry on
            spans[-1][1] = pos + len(text)
        else:
            spans.append([pos, pos + len(text), meta.get("page") or 0, loc])
        pos += len(text) + 2  # the "\n\n" between pieces
    return spans


def load_patterns():
    return (yaml.safe_load((rag.HERE / "config.yaml").read_text()) or {}).get("exclude_patterns") or []


def is_excluded(name, patterns):
    return any(fnmatch(name.lower(), pat.lower()) for pat in patterns)


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))


def main():
    t0 = time.monotonic()
    docs_dir = rag.DOCS_DIR
    if not docs_dir.is_dir():
        sys.exit(f"DOCS_DIR not found: {docs_dir}")
    patterns = load_patterns()
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    col = rag.get_collection()
    tok = rag.get_model().tokenizer

    wanted, unsupported, excluded = {}, Counter(), 0
    for p in sorted(docs_dir.rglob("*")):
        if not p.is_file() or p.name == "manifest.json":
            continue
        rel = p.relative_to(docs_dir).as_posix()
        if p.suffix.lower() not in SUPPORTED:
            unsupported[p.suffix.lower() or "(none)"] += 1
            log.info("Skipping unsupported file: %s", rel)
            continue
        if is_excluded(p.name, patterns):
            excluded += 1
            continue
        wanted[rel] = p

    # Files gone from disk (or now excluded) lose their chunks.
    removed = [rel for rel in state if rel not in wanted]
    for rel in removed:
        col.delete(where={"path": rel})
        del state[rel]
    save_state(state)

    indexed = unchanged = 0
    failed = []
    for rel, p in tqdm(wanted.items(), desc="Indexing", unit="file"):
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if state.get(rel, {}).get("hash") == h and state[rel].get("v") == INDEX_VERSION:
                unchanged += 1
                continue
            blocks = list(pdf_blocks(p) if p.suffix.lower() == ".pdf" else docx_blocks(p))
            chunks = chunk(blocks, tok)
            col.delete(where={"path": rel})
            if chunks:
                mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
                # Chroma metadata can't hold None or lists: keep where the chunk starts, spans as JSON.
                metas = [{"path": rel, "filename": p.name, "chunk": i, "mtime": mtime, "spans": json.dumps(spans),
                          **{k: v for k, v in meta.items() if k in ("page", "section") and v}}
                         for i, (_, meta, spans) in enumerate(chunks)]
                col.add(
                    ids=[f"{rel}::{i}" for i in range(len(chunks))],
                    documents=[text for text, _, _ in chunks],
                    embeddings=rag.embed([rag.passage(m, text) for m, (text, _, _) in zip(metas, chunks)]),
                    metadatas=metas,
                )
            state[rel] = {"hash": h, "chunks": len(chunks), "v": INDEX_VERSION}
            save_state(state)
            indexed += 1
        except Exception as e:  # one bad file must never abort the run
            failed.append((rel, f"{type(e).__name__}: {e}"))
            tqdm.write(f"FAILED {rel}: {type(e).__name__}: {e}")

    no_text = sorted(rel for rel, s in state.items() if s["chunks"] == 0)
    print("\n==== Ingest summary ====")
    print(f"Indexed (new/changed): {indexed}")
    print(f"Unchanged:             {unchanged}")
    print(f"Removed from index:    {len(removed)}")
    print(f"Excluded (config.yaml): {excluded}")
    print(f"Skipped (unsupported): {sum(unsupported.values())}  {dict(unsupported.most_common())}")
    print(f"Failed:                {len(failed)}")
    for rel, err in failed:
        print(f"  - {rel}: {err}")
    print(f"No text extracted (may need OCR): {len(no_text)}")
    for rel in no_text:
        print(f"  - {rel}")
    print(f"Chunks in index:       {col.count()}")
    print(f"Time taken:            {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
