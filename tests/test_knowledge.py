"""Tests for the knowledge base loader and search.

These tests build their OWN small fake documents in a temporary folder
(pytest's tmp_path fixture), so they never depend on the real company
files. If someone edits loan-product.md, these tests still mean the same thing.
"""

import pytest

from knowledge import load_chunks, search

SAMPLE_LOAN_DOC = """# Sample Loan

## Fees
The processing fee is 2% of the loan amount.

## Who Qualifies
Active members with regular income may qualify.
"""


def _make_knowledge_dir(tmp_path):
    """Create a temporary knowledge folder with one sample document."""
    (tmp_path / "loan.md").write_text(SAMPLE_LOAN_DOC, encoding="utf-8")
    return tmp_path


def test_document_is_split_into_sections(tmp_path):
    chunks = load_chunks(_make_knowledge_dir(tmp_path))
    headings = [chunk.heading for chunk in chunks]
    assert "Fees" in headings
    assert "Who Qualifies" in headings


def test_search_returns_the_most_relevant_section_first(tmp_path):
    chunks = load_chunks(_make_knowledge_dir(tmp_path))
    results = search(chunks, "what are the fees for the loan")
    assert results[0].heading == "Fees"


def test_search_returns_nothing_for_unrelated_question(tmp_path):
    chunks = load_chunks(_make_knowledge_dir(tmp_path))
    assert search(chunks, "weather in Paris") == []


def test_missing_knowledge_folder_fails_loudly(tmp_path):
    # A support agent with no knowledge must NOT start silently.
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "does-not-exist")