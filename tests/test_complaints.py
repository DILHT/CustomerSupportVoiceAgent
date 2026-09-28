"""Tests for complaint validation, routing, reference numbers, storage, and limits."""

import re
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from complaints import (
    Complaint,
    ComplaintLimitReachedError,
    ComplaintStore,
    generate_reference,
    route_complaint,
)


def _valid(**overrides):
    """Build a valid complaint; tests override one field at a time."""
    data = {
        "member_name": "Test Member",
        "phone": "0888 123 456",
        "category": "charges",
        "description": "I was charged a fee twice on my last loan repayment.",
    }
    data.update(overrides)
    return Complaint.model_validate(data)


# --- Validation: bad data must never reach the database ---


def test_phone_is_normalised_to_one_format():
    assert _valid(phone="0888 123 456").phone == "+265888123456"
    assert _valid(phone="+265 999 123 456").phone == "+265999123456"


def test_invalid_phone_is_rejected():
    with pytest.raises(ValidationError):
        _valid(phone="12345")


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        _valid(category="car_loans")


def test_too_short_description_is_rejected():
    with pytest.raises(ValidationError):
        _valid(description="bad")


# --- Routing: a control, so it must be deterministic ---


def test_fraud_goes_to_risk_and_compliance_as_urgent():
    assert route_complaint("fraud_or_security") == ("Risk and Compliance", "urgent")


def test_ordinary_complaint_goes_to_customer_service():
    assert route_complaint("charges") == ("Customer Service", "standard")


# --- Reference numbers ---


def test_reference_has_expected_format():
    ref = generate_reference(datetime(2026, 9, 24, tzinfo=timezone.utc))
    assert re.fullmatch(r"CMP-20260924-[0-9A-F]{6}", ref)


def test_references_are_unique():
    now = datetime.now(timezone.utc)
    refs = {generate_reference(now) for _ in range(100)}
    assert len(refs) == 100


# --- Storage ---


def test_complaint_is_saved(tmp_path):
    store = ComplaintStore(tmp_path / "complaints.db")
    record = store.save(_valid())
    assert record.reference.startswith("CMP-")
    assert store.count() == 1


def test_sql_injection_is_stored_as_plain_text(tmp_path):
    # If queries were built by pasting strings together, this description
    # would delete the table. With parameterised queries it's just text.
    store = ComplaintStore(tmp_path / "complaints.db")
    store.save(_valid(description="Robert'); DROP TABLE complaints; -- test"))
    store.save(_valid())
    assert store.count() == 2


# --- Daily limit per phone number ---

# 08:00 UTC = 10:00 in Malawi, safely in the middle of the day
MORNING = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)


def test_three_complaints_in_one_day_are_allowed(tmp_path):
    store = ComplaintStore(tmp_path / "c.db")
    for _ in range(3):
        store.save(_valid(), now=MORNING)
    assert store.count() == 3


def test_fourth_complaint_same_day_is_rejected(tmp_path):
    store = ComplaintStore(tmp_path / "c.db")
    for _ in range(3):
        store.save(_valid(), now=MORNING)
    with pytest.raises(ComplaintLimitReachedError):
        store.save(_valid(), now=MORNING + timedelta(hours=2))


def test_fraud_report_is_never_blocked(tmp_path):
    store = ComplaintStore(tmp_path / "c.db")
    for _ in range(3):
        store.save(_valid(), now=MORNING)
    store.save(_valid(category="fraud_or_security"), now=MORNING)  # must NOT raise
    assert store.count() == 4


def test_limit_resets_the_next_day(tmp_path):
    store = ComplaintStore(tmp_path / "c.db")
    for _ in range(3):
        store.save(_valid(), now=MORNING)
    store.save(_valid(), now=MORNING + timedelta(days=1))  # must NOT raise
    assert store.count() == 4


def test_day_boundary_uses_malawi_time(tmp_path):
    # 21:30 UTC = 23:30 in Malawi on 27 Sept.
    # 22:30 UTC = 00:30 in Malawi on 28 Sept: a NEW day for the member.
    store = ComplaintStore(tmp_path / "c.db")
    late_evening = datetime(2026, 9, 27, 21, 30, tzinfo=timezone.utc)
    for _ in range(3):
        store.save(_valid(), now=late_evening)
    store.save(_valid(), now=late_evening + timedelta(hours=1))  # must NOT raise
    assert store.count() == 4


def test_other_phone_number_is_unaffected(tmp_path):
    # Arrange: a fresh database where one number has used up its limit
    store = ComplaintStore(tmp_path / "c.db")
    for _ in range(3):
        store.save(_valid(), now=MORNING)

    # Act: a DIFFERENT number complains on the same day
    store.save(_valid(phone="0999 123 456"), now=MORNING)

    # Assert: it was saved, so there are 4 complaints in total
    assert store.count() == 4
