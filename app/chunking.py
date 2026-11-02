"""Split documents into overlapping word windows that respect paragraph breaks."""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str


def split_into_chunks(text: str, max_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    paragraphs = [p.split() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    window: list[str] = []

    def flush():
        if window:
            chunks.append(Chunk(len(chunks), " ".join(window)))

    for words in paragraphs:
        # A paragraph that does not fit starts a new chunk, carrying the overlap forward.
        if window and len(window) + len(words) > max_words:
            flush()
            window = window[-overlap_words:]
        while len(words) > max_words - len(window):
            room = max_words - len(window)
            window.extend(words[:room])
            words = words[room:]
            flush()
            window = window[-overlap_words:]
        window.extend(words)

    if window and (not chunks or " ".join(window) not in chunks[-1].text):
        flush()
    return chunks
