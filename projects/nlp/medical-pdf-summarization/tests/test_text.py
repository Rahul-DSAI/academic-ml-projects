from medical_summarizer.text import normalize_whitespace, split_sentences


def test_normalize_whitespace_preserves_words():
    assert normalize_whitespace("  medical \n  study\tresults ") == "medical study results"


def test_split_sentences_empty_text():
    assert split_sentences("   ") == []


def test_split_sentences_returns_multiple_sentences():
    sentences = split_sentences("The study enrolled 100 patients. Results were significant. More work is needed.")
    assert len(sentences) == 3
