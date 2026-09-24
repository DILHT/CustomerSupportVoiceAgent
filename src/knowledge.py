"""Company knowledge base: load markdown documents and find relevant sections.

This is deliberately simple keyword search. The rest of the agent only uses
load_chunks() and search(), so in Phase 2 we can replace the inside of
search() with vector search (pgvector) without changing the agent's tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A markdown heading: group 1 is the '#' characters (the level), group 2 is the title.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Words are runs of letters/digits. "3.75%" becomes "3" and "75", which is fine.
_WORD = re.compile(r"[a-z0-9]+")

# Common words that appear everywhere and carry no meaning for matching.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
    "who",
    "with",
    "you",
    "your",
}

# Caps protect the agent: the query comes from the LLM, which is driven by
# the caller, so we treat it as untrusted input.
MAX_QUERY_CHARS = 300
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Chunk:
    """One section of one document. frozen=True makes it read-only."""

    source: str  # file name, e.g. "loan-product.md"
    heading: str  # section title, e.g. "Who Qualifies"
    text: str  # the section's content


def load_chunks(knowledge_dir: Path) -> list[Chunk]:
    """Read every .md file in knowledge_dir and split each one by heading.

    Raises FileNotFoundError if the folder is missing: failing at startup is
    safer than an agent that runs but knows nothing.
    """
    knowledge_dir = Path(knowledge_dir)
    if not knowledge_dir.is_dir():
        raise FileNotFoundError(f"Knowledge folder not found: {knowledge_dir}")

    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        chunks.extend(_split_by_heading(path.name, content))
    return chunks


def search(chunks: list[Chunk], query: str, top_k: int = DEFAULT_TOP_K) -> list[Chunk]:
    """Return up to top_k chunks that share the most words with the query.

    A word found in a section's heading counts double, because headings
    describe what the whole section is about.
    """
    query_words = _words(query[:MAX_QUERY_CHARS])
    if not query_words:
        return []

    scored: list[tuple[int, Chunk]] = []
    for chunk in chunks:
        heading_hits = len(query_words & _words(chunk.heading))
        body_hits = len(query_words & _words(chunk.text))
        score = heading_hits * 2 + body_hits
        if score > 0:
            scored.append((score, chunk))

    # Highest score first; keep only the best few to keep voice replies fast.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _split_by_heading(source: str, content: str) -> list[Chunk]:
    """Turn one document into chunks, starting a new chunk at each heading.

    The top-level '# Title' is remembered and prefixed to every section below
    it, e.g. "Mgodi Personal Loan > Fees". This is contextual chunking: each
    chunk carries enough context to be understood on its own.
    """
    chunks: list[Chunk] = []
    title = ""  # the document's '# Title', once we've seen it
    heading = "Overview"  # used for text that appears before any heading
    lines: list[str] = []

    for line in content.splitlines():
        match = _HEADING.match(line)
        if match:
            _add_chunk(chunks, source, heading, lines)
            level = len(match.group(1))  # '#' = 1, '##' = 2, ...
            text = match.group(2).strip()
            if level == 1:
                title = text
                heading = text
            else:
                heading = f"{title} > {text}" if title else text
            lines = []
        else:
            lines.append(line)

    _add_chunk(chunks, source, heading, lines)
    return chunks


def _add_chunk(
    chunks: list[Chunk], source: str, heading: str, lines: list[str]
) -> None:
    """Save a section as a chunk, skipping sections with no text."""
    text = "\n".join(lines).strip()
    if text:
        chunks.append(Chunk(source=source, heading=heading, text=text))


def _words(text: str) -> set[str]:
    """Lowercase, split into words, drop stopwords, and roughly remove plurals.

    The plural trim ("fees" -> "fee") is crude but applied equally to the
    query and the documents, so they still match each other.
    """
    words = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        words.add(word)
    return words
