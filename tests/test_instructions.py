"""Tests for the system prompt builder."""

from agent import build_instructions,build_greeting



def test_company_name_is_filled_in():
    text = build_instructions("Test SACCO")
    assert "Test SACCO" in text


def test_no_unfilled_placeholders():
    # Catches exactly the bug we hit: a missing f-string prefix
    # leaves "{COMPANY_NAME}" in the text the agent speaks.
    text = build_instructions("Test SACCO")
    assert "{COMPANY_NAME}" not in text

def test_greeting_names_the_company():
    text = build_greeting("Test SACCO")
    assert "Test SACCO" in text

def test_greeting_says_it_is_automated():
    text = build_greeting("Test SACCO").lower()
    assert "automated" in text