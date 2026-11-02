"""Answers a question from retrieved chunks, with Claude's Citations API pointing back to sources."""

from dataclasses import dataclass, field
from typing import Protocol

import anthropic

from app.config import settings
from app.store import SearchHit

SYSTEM_PROMPT = (
    "Answer the user's question using only the provided documents. "
    "If the documents do not contain the answer, say that you could not find it in the documents. "
    "Be concise."
)


@dataclass
class Citation:
    source: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)


class AnswerError(Exception):
    pass


class Answerer(Protocol):
    def answer(self, question: str, hits: list[SearchHit]) -> Answer: ...


def source_label(hit: SearchHit) -> str:
    return f"{hit.chunk.document_title} (part {hit.chunk.chunk_index + 1})"


class ClaudeAnswerer:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        self.client = client or anthropic.Anthropic()
        self.model = model or settings.claude_model

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
                "title": source_label(hit),
                "citations": {"enabled": True},
            }
            for hit in hits
        ]
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
            )
        except anthropic.RateLimitError as e:
            raise AnswerError("Rate limited by the Claude API, try again shortly") from e
        except anthropic.APIStatusError as e:
            raise AnswerError(f"Claude API error {e.status_code}") from e
        except anthropic.APIConnectionError as e:
            raise AnswerError("Could not reach the Claude API") from e

        if response.stop_reason == "refusal":
            raise AnswerError("The model declined to answer this question")

        parts: list[str] = []
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            for c in block.citations or []:
                key = (c.document_title, c.cited_text)
                if key not in seen:
                    seen.add(key)
                    citations.append(Citation(c.document_title, c.cited_text.strip()))
        return Answer("".join(parts).strip(), citations)
