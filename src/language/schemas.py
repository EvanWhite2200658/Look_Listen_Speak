# src/language/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    content: str


@dataclass(frozen=True)
class ResponseRequest:
    user_text: str
    system_prompt: Optional[str] = None
    conversation_history: List[DialogueTurn] = field(default_factory=list)


@dataclass(frozen=True)
class ResponseResult:
    text: str
    prompt_tokens: Optional[int]
    generated_tokens: Optional[int]
    generation_time_s: float
    model_name: str