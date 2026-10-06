"""Chunk packing: sizes, overlap, oversized paragraphs, and the spans that pin each line to its page/para."""

import json

import pytest

from conftest import FakeTokenizer
from ingest import CHUNK_TOKENS, chunk, is_excluded, load_patterns

tok = FakeTokenizer()


def n(s):
    return len(tok(s)["offset_mapping"])


def make_chunks():
    paras = [(f"{i}. Paragraph {i} " + "word " * 60, {"page": i // 5 + 1, "para": str(i)}) for i in range(1, 41)]
    paras.append(("huge " * 1500, {"page": 99}))  # one paragraph far bigger than a chunk
    return chunk(paras, tok)


def test_no_chunk_exceeds_the_embedding_model_limit():
    assert all(n(text) <= CHUNK_TOKENS for text, _, _ in make_chunks())


def test_trailing_paragraphs_overlap_into_the_next_chunk():
    chunks = make_chunks()
    assert chunks[0][0].split("\n\n")[-1] in chunks[1][0]


def test_an_oversized_paragraph_is_split_without_losing_text():
    chunks = make_chunks()
    assert chunks[0][1]["page"] == 1 and chunks[-1][1]["page"] == 99
    assert sum(t.count("huge") for t, m, _ in chunks if m["page"] == 99) >= 1500


def test_spans_point_every_paragraph_at_its_own_page_and_para():
    for text, _, spans in make_chunks():
        for start, end, page, loc in spans:
            piece = text[start:end]
            if piece.startswith("huge"):
                assert (page, loc) == (99, "page 99")
                continue
            num = piece.split(".")[0]
            assert loc == f"page {int(num) // 5 + 1}, para {num}"
            assert page == int(num) // 5 + 1
        json.dumps(spans)  # stored in Chroma metadata as JSON


def test_consecutive_lines_of_one_paragraph_share_a_span():
    blocks = [("12. First line of the para", {"page": 3, "para": "12"}),
              ("second line, same para", {"page": 3, "para": "12"}),
              ("13. Next para", {"page": 3, "para": "13"})]
    [(text, _, spans)] = chunk(blocks, tok)
    assert [s[3] for s in spans] == ["page 3, para 12", "page 3, para 13"]
    assert text[spans[0][0]:spans[0][1]] == "12. First line of the para\n\nsecond line, same para"


@pytest.mark.parametrize("name,skip", [
    ("RAFAC Form 1771.pdf", True),
    ("Form_123.docx", True),
    ("Application_Form.docx", True),
    ("Forms - Annex A.pdf", True),
    ("Uniform Guide.pdf", False),  # "uniform " used to match "*form[ _.-]*"
    ("Information Pack.docx", False),
    ("ACP 20 Platform 2.pdf", False),
])
def test_config_excludes_forms_but_not_words_containing_form(name, skip):
    assert is_excluded(name, load_patterns()) is skip
