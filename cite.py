"""Turning the LLM's reply into checked citations.

The model is asked to back each [n] with a passage copied from source n. Models
misquote (fixing typos, dropping words, joining sentences), so a quote is only
shown if it can be found in the chunk: exactly once whitespace, case and smart
punctuation are ignored, or failing that as the source sentence(s) sharing most
of its words. Anything that can't be found is dropped, so every highlighted
line the UI shows really is in the document.

Kept free of torch/Chroma imports so it's cheap to test.
"""

import json
import re

# Numbered paragraphs ("12.", "3.4", "12)") are how RAFAC documents are referenced,
# so they're worth surfacing. A bare "12 May" isn't one: a single number needs a "." or ")".
_PARA = re.compile(r"^\(?(\d{1,3}(?:\.\d{1,3}){1,3}|\d{1,3}[.)])(?=\s|$)")
_REF = re.compile(r"\[(\d+(?:\s*(?:,|-|–)\s*\d+)*)\]")
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")
_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def para_label(text):
    """'12' for a block starting "12. Cadets must...", '3.4' for "3.4 The OC...", else None."""
    m = _PARA.match(text.strip())
    return m.group(1).rstrip(".)") if m else None


def is_para_number(text):
    """A PDF block that is only a paragraph number: these sit in the margin as their own block."""
    t = text.strip()
    return bool(t) and para_label(t) is not None and len(t) <= 10


def _normalize(s):
    """Lowercased, smart punctuation flattened, whitespace collapsed; plus each char's index in s."""
    out, idx = [], []
    prev_space = True
    for i, ch in enumerate(s.translate(_TRANSLATE).lower()):
        if ch.isspace():
            if prev_space:
                continue
            ch = " "
        prev_space = ch == " "
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def _clean_quote(q):
    q = q.strip().strip('"\'“”‘’').strip()
    # A quote shortened with an ellipsis can't match as a whole; the longest piece still points at the line.
    parts = [p.strip() for p in re.split(r"\.\.\.|…", q) if p.strip()]
    return max(parts, key=len) if parts else ""


def _sentences(text):
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        if text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


def find_quote(quote, text):
    """(start, end) of quote in text, or None if the source doesn't say it."""
    q = _clean_quote(quote)
    if len(q) < 8:
        return None
    nq, _ = _normalize(q)
    nt, idx = _normalize(text)
    pos = nt.find(nq)
    if pos >= 0:
        return idx[pos], idx[pos + len(nq) - 1] + 1

    # Near miss: the run of up to three sentences whose words best match the quote's.
    qwords = _WORD.findall(nq)
    if len(qwords) < 4:
        return None
    sents = _sentences(text)
    best, best_f1 = None, 0.0
    for i in range(len(sents)):
        for j in range(i, min(i + 3, len(sents))):
            start, end = sents[i][0], sents[j][1]
            swords = _WORD.findall(_normalize(text[start:end])[0])
            common = sum(min(qwords.count(w), swords.count(w)) for w in set(qwords))
            if not common:
                continue
            p, r = common / len(swords), common / len(qwords)
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best, best_f1 = (start, end), f1
    return best if best_f1 >= 0.75 else None


def parse_reply(content):
    """(answer, [(source_n, quote)]) from the model's JSON; the raw text and no quotes if it isn't JSON."""
    raw = (content or "").strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(body)
    except ValueError:
        m = re.search(r"\{.*\}", body, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return raw, []
    cites = []
    for c in data.get("citations") or []:
        if not isinstance(c, dict):
            continue
        try:
            n = int(c.get("source"))
        except (TypeError, ValueError):
            continue
        if isinstance(c.get("quote"), str):
            cites.append((n, c["quote"]))
    return data["answer"].strip(), cites


def _expand(group):
    """"1, 3-5" -> [1, 3, 4, 5]."""
    nums = []
    for part in re.split(r"\s*,\s*", group):
        bounds = [int(x) for x in re.split(r"\s*[-–]\s*", part)]
        if len(bounds) == 2 and 0 < bounds[1] - bounds[0] < 10:
            nums.extend(range(bounds[0], bounds[1] + 1))
        else:
            nums.extend(bounds)
    return nums


def cited_order(answer, valid):
    """Source numbers referenced in the answer, in order of first use, ignoring ones not in valid."""
    order = []
    for m in _REF.finditer(answer):
        for n in _expand(m.group(1)):
            if n in valid and n not in order:
                order.append(n)
    return order


def renumber(answer, mapping):
    """Rewrite [n] refs through mapping (old -> new) as [1][2]; refs to unknown sources are dropped."""
    def sub(m):
        new = []
        for n in _expand(m.group(1)):
            if n in mapping and mapping[n] not in new:
                new.append(mapping[n])
        return "".join(f"[{n}]" for n in new)

    out = _REF.sub(sub, answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", out)).strip()


def location(meta):
    """Human-readable place in a document: "page 4, para 12" for PDFs, "Section name, para 3" for Word."""
    where = f"page {meta['page']}" if meta.get("page") else (meta.get("section") or "")[:80]
    para = f"para {meta['para']}" if meta.get("para") else ""
    return ", ".join(x for x in (where, para) if x)


def span_at(spans, offset):
    """The [start, end, page, location] block of a chunk that contains offset, or None."""
    for s in spans:
        if s[0] <= offset < s[1]:
            return s
    return None
