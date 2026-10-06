import pytest

import cite

SOURCE = (
    "12. Cadets may sleep overnight in Army Reserve Centres, provided that the OC has written\n"
    "approval from the Wing.\n\n13. Adult staff of both sexes must be present when mixed groups stay overnight."
)


@pytest.mark.parametrize("text,label", [
    ("12. Cadets must...", "12"),
    ("3.4 The OC is responsible", "3.4"),
    ("(2) Where a cadet", "2"),
    ("12) Staff", "12"),
    ("12 May 2023 update", None),  # a date, not a paragraph
    ("2024 review", None),
    ("Cadets must", None),
    ("1.5 metres from the edge", "1.5"),  # can't tell from a numbered para; accepted
])
def test_para_label(text, label):
    assert cite.para_label(text) == label


def test_a_lone_number_block_is_a_para_number_but_a_sentence_is_not():
    assert cite.is_para_number("12.")
    assert cite.is_para_number("3.4")
    assert not cite.is_para_number("12. Cadets must not")
    assert not cite.is_para_number("12")  # a page number in the footer


def test_an_exact_quote_is_found_across_a_line_break():
    q = "provided that the OC has written approval from the Wing"
    start, end = cite.find_quote(q, SOURCE)
    assert SOURCE[start:end] == "provided that the OC has written\napproval from the Wing"


def test_quote_matching_ignores_case_smart_quotes_and_surrounding_quote_marks():
    src = "The cadet’s parent must sign the form — in ink."
    start, end = cite.find_quote('"THE CADET\'S PARENT MUST SIGN THE FORM - in ink."', src)
    assert src[start:end] == src


def test_a_slightly_misquoted_sentence_resolves_to_the_real_sentence():
    q = "Adult staff of both sexes should be present when mixed groups stay overnight"
    start, end = cite.find_quote(q, SOURCE)
    assert SOURCE[start:end] == "Adult staff of both sexes must be present when mixed groups stay overnight."


def test_an_ellipsis_quote_matches_on_its_longest_piece():
    q = "Cadets may sleep overnight in Army Reserve Centres ... from the Wing"
    start, end = cite.find_quote(q, SOURCE)
    assert SOURCE[start:end] == "Cadets may sleep overnight in Army Reserve Centres"


@pytest.mark.parametrize("q", [
    "Cadets may never sleep at the centre under any circumstances whatsoever",
    "Wing",  # too short to pin to a line
    "",
])
def test_a_quote_the_source_does_not_contain_is_rejected(q):
    assert cite.find_quote(q, SOURCE) is None


def test_parse_reply_reads_json_and_skips_malformed_citations():
    reply = ('{"answer": "Yes [1].", "citations": [{"source": 1, "quote": "x y"}, {"source": "two", "quote": "z"},'
             ' "junk", {"source": 2}, {"source": "3", "quote": "w"}]}')
    assert cite.parse_reply(reply) == ("Yes [1].", [(1, "x y"), (3, "w")])


def test_parse_reply_accepts_a_fenced_or_wrapped_json_reply():
    assert cite.parse_reply('```json\n{"answer": "A", "citations": []}\n```') == ("A", [])
    assert cite.parse_reply('Here you go: {"answer": "B", "citations": null} hope that helps') == ("B", [])


@pytest.mark.parametrize("reply", ["Plain text answer [1].", '{"answer": 5}', "[1, 2]", "{broken", ""])
def test_parse_reply_falls_back_to_the_raw_text_when_it_is_not_the_expected_json(reply):
    assert cite.parse_reply(reply) == (reply, [])


def test_cited_order_follows_first_use_and_expands_lists_and_ranges():
    answer = "A [3]. B [1, 3]. C [4-6]. D [9]."
    assert cite.cited_order(answer, range(1, 7)) == [3, 1, 4, 5, 6]


def test_renumber_rewrites_refs_and_drops_unknown_ones():
    out = cite.renumber("A [3]. B [1, 3] and [9]. C [2]", {3: 1, 1: 2})
    assert out == "A [1]. B [2][1] and. C"


def test_location_formats_pdf_and_word_positions():
    assert cite.location({"page": 4, "para": "12"}) == "page 4, para 12"
    assert cite.location({"section": "Overnight stays", "para": "3.2"}) == "Overnight stays, para 3.2"
    assert cite.location({"section": ""}) == ""
    assert cite.location({"page": 7}) == "page 7"


def test_span_at_finds_the_block_holding_an_offset():
    spans = [[0, 10, 1, "page 1"], [12, 30, 2, "page 2, para 4"]]
    assert cite.span_at(spans, 15)[3] == "page 2, para 4"
    assert cite.span_at(spans, 11) is None
    assert cite.span_at([], 0) is None
