"""Complaint intake: validate a complaint, route it, and store it.

The agent's tool only calls ComplaintStore.save(). Storage is SQLite for
now; in Phase 2 we swap in Postgres (Neon) behind the same method, and the
agent's tool doesn't change. Same "contract" idea as knowledge.search().
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# SECTION 1: Settings and constants
# =============================================================================

# The only categories allowed. Literal means "must be exactly one of these".
Category = Literal[
    "service",
    "staff_conduct",
    "transaction",
    "charges",
    "loan_application",
    "digital_channel",
    "fraud_or_security",
    "data_protection",
    "other",
]

# Per the complaints policy: these go to a control function, not
# ordinary customer service.
_ESCALATE_TO_RISK = {"fraud_or_security", "data_protection", "staff_conduct"}

# Malawi uses Central Africa Time: UTC+2 all year, no daylight saving,
# so a fixed offset is correct and needs no timezone database.
MALAWI_TZ = timezone(timedelta(hours=2), name="CAT")

# How many complaints one phone number may log per day.
DEFAULT_DAILY_LIMIT = 3

# Never block urgent reports: a member reporting stolen money or a data
# leak must always get through, even on their fourth call that day.
_EXEMPT_FROM_LIMIT = {"fraud_or_security", "data_protection"}


# =============================================================================
# SECTION 2: Errors
# =============================================================================


class ComplaintLimitReachedError(Exception):
    """Raised when a phone number has already logged its daily maximum."""


# =============================================================================
# SECTION 3: The complaint itself (validation)
# =============================================================================


class Complaint(BaseModel):
    """A complaint as collected from the member. Pydantic validates every field."""

    member_name: str = Field(min_length=2, max_length=100)
    phone: str
    category: Category
    description: str = Field(min_length=10, max_length=1000)

    @field_validator("member_name", "description", mode="before")
    @classmethod
    def _strip_whitespace(cls, value: object) -> object:
        # mode="before" runs BEFORE the length checks, so "   " can't sneak
        # past min_length as three characters.
        return value.strip() if isinstance(value, str) else value

    @field_validator("phone", mode="before")
    @classmethod
    def _normalise_phone(cls, value: object) -> str:
        """Accept '0888 123 456' or '+265 888 123 456'; store '+265888123456'."""
        if not isinstance(value, str):
            raise ValueError("phone must be text")
        digits = re.sub(r"\D", "", value)  # \D = any non-digit; remove them all
        if digits.startswith("265") and len(digits) == 12:
            return "+" + digits
        if digits.startswith("0") and len(digits) == 10:
            return "+265" + digits[1:]
        raise ValueError(
            "must be a Malawi number, for example 0888123456 or +265888123456"
        )


@dataclass(frozen=True)
class ComplaintRecord:
    """What the store returns after saving: the details the member needs."""

    reference: str
    category: str
    assigned_team: str
    priority: str
    created_at: str


# =============================================================================
# SECTION 4: Helper functions (pure: same input, same output, no database)
# =============================================================================


def route_complaint(category: str) -> tuple[str, str]:
    """Decide which team handles a complaint, and how urgently.

    Plain code, not the LLM: routing is a control, so it must behave
    the same way every time and be easy to audit.
    """
    if category in _ESCALATE_TO_RISK:
        return ("Risk and Compliance", "urgent")
    return ("Customer Service", "standard")


def generate_reference(now: datetime) -> str:
    """Create a reference like CMP-20260924-A3F9C1.

    secrets (not random) makes it unpredictable, so nobody can guess
    another member's reference number.
    """
    return f"CMP-{now:%Y%m%d}-{secrets.token_hex(3).upper()}"


def _db_time(moment: datetime) -> str:
    """The ONE format for times in the database: UTC with microseconds.

    Saving and counting must use the same format, because the count
    compares these values as text.
    """
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def start_of_malawi_day(moment: datetime) -> datetime:
    """Midnight in Malawi on the day that `moment` falls on."""
    local = moment.astimezone(MALAWI_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


# =============================================================================
# SECTION 5: Storage
# =============================================================================


class ComplaintStore:
    """Saves complaints to a SQLite database file."""

    def __init__(self, db_path: Path, daily_limit: int = DEFAULT_DAILY_LIMIT) -> None:
        self._db_path = Path(db_path)
        self._daily_limit = daily_limit
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                reference     TEXT PRIMARY KEY,
                created_at    TEXT NOT NULL,
                member_name   TEXT NOT NULL,
                phone         TEXT NOT NULL,
                category      TEXT NOT NULL,
                description   TEXT NOT NULL,
                assigned_team TEXT NOT NULL,
                priority      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'open'
            )
            """
        )

    def count_today(self, phone: str, now: datetime) -> int:
        """How many complaints this phone number has logged today (Malawi time)."""
        since = _db_time(start_of_malawi_day(now))
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM complaints WHERE phone = ? AND created_at >= ?",
                (phone, since),
            ).fetchone()
            return row[0]
        finally:
            conn.close()

    def save(
        self, complaint: Complaint, now: datetime | None = None
    ) -> ComplaintRecord:
        """Store a validated complaint and return its reference details.

        Raises ComplaintLimitReachedError if this phone number is over today's
        limit, unless the category is exempt (urgent reports always go through).

        `now` can be passed in by tests; in real use it's left empty and the
        real clock is used. This is dependency injection.
        """
        now = now or datetime.now(timezone.utc)

        # 1. Check the limit BEFORE writing anything.
        if (
            complaint.category not in _EXEMPT_FROM_LIMIT
            and self.count_today(complaint.phone, now) >= self._daily_limit
        ):
            raise ComplaintLimitReachedError(
                f"Daily limit of {self._daily_limit} reached"
            )

        # 2. Work out the details.
        created_at = _db_time(now)
        team, priority = route_complaint(complaint.category)
        reference = generate_reference(now)

        # 3. Save. The ? placeholders are PARAMETERS: SQLite receives the
        #    values separately from the SQL, so text like "'); DROP TABLE"
        #    can never run as a command. Never build SQL with f-strings.
        self._execute(
            """
            INSERT INTO complaints (reference, created_at, member_name, phone,
                                    category, description, assigned_team, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reference,
                created_at,
                complaint.member_name,
                complaint.phone,
                complaint.category,
                complaint.description,
                team,
                priority,
            ),
        )
        return ComplaintRecord(
            reference, complaint.category, team, priority, created_at
        )

    def count(self) -> int:
        """Total number of stored complaints (used by tests)."""
        conn = sqlite3.connect(self._db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        finally:
            conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        """Run one write statement in its own connection, then close it."""
        conn = sqlite3.connect(self._db_path)
        try:
            # 'with conn' commits on success and rolls back on error, but it
            # does NOT close the connection, hence the try/finally.
            with conn:
                conn.execute(sql, params)
        finally:
            conn.close()
