from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from .structured_output import (
    IncrementalTextSanitizer,
    StructuredMessageAssembler,
    StructuredTagParser,
    create_default_structured_tag_specs,
)


CHATML_CONTROL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|endoftext|>",
]
CHATML_STOP_TOKENS = ["<|im_end|>", "<|end|>", "<|endoftext|>"]


def parse_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "text"):
        return chunk.text or ""
    if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is not None and getattr(delta, "content", None):
            return delta.content
    return ""


def is_supported_chunk_type(chunk: Any) -> bool:
    if isinstance(chunk, str):
        return True
    if hasattr(chunk, "text"):
        return True
    return bool(hasattr(chunk, "choices") and chunk.choices)


@dataclass
class StreamSession:
    response_stream: Iterable
    eos_token: Optional[str]
    stream: bool
    generation_stop_event: Any
    control_tokens: list[str] = field(default_factory=lambda: CHATML_CONTROL_TOKENS)
    stop_tokens: list[str] = field(default_factory=lambda: CHATML_STOP_TOKENS)
    thinking_title: str = "Thinking"
    inline_thought_title: str = "Thinking"
    thinking_id: int = 0
    base_messages: list[Any] = field(default_factory=list)
    full_response: str = ""
    final_messages: list[Any] = field(default_factory=list)
    chunk_observer: Optional[Callable[[Any], None]] = None

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        sanitizer = IncrementalTextSanitizer(
            control_tokens=self.control_tokens,
            stop_tokens=self.stop_tokens,
            eos_token=self.eos_token,
        )
        tag_specs = create_default_structured_tag_specs(
            thinking_title=self.inline_thought_title or self.thinking_title,
        )
        parser = StructuredTagParser(tag_specs)
        assembler = StructuredMessageAssembler(tag_specs, starting_id=self.thinking_id)
        last_payload: Optional[list[Any]] = None

        for chunk in self.response_stream:
            if self.generation_stop_event.is_set():
                break
            if self.chunk_observer is not None:
                self.chunk_observer(chunk)

            chunk_text = parse_chunk_text(chunk)
            if not chunk_text:
                continue

            self.full_response += chunk_text
            sanitized = sanitizer.feed(chunk_text)
            if sanitized.text:
                assembler.consume(parser.feed(sanitized.text))
                payload = self.base_messages + assembler.snapshot(finalize=False)
                if self.stream and payload != last_payload:
                    last_payload = payload
                    yield payload

            if sanitized.stop:
                break

        tail = sanitizer.finalize()
        if tail.text or parser.buffer:
            assembler.consume(parser.feed(tail.text, final=True))
        else:
            assembler.consume(parser.feed("", final=True))

        self.final_messages = assembler.snapshot(finalize=True)
        final_payload = self.base_messages + self.final_messages
        if self.stream and final_payload != last_payload:
            yield final_payload


def generate_response_and_stream(
    model: Any,
    response_args: dict[str, Any],
    eos_token: Optional[str],
    stream: bool,
    generation_stop_event: Any,
    thinking_title: str = "Thinking",
    inline_thought_title: str = "Thinking",
    thinking_id: int = 0,
    base_messages: Optional[list[Any]] = None,
    chunk_observer: Optional[Callable[[Any], None]] = None,
) -> Iterator[list[dict[str, Any]]]:
    session = StreamSession(
        response_stream=model.generate_response(**response_args),
        eos_token=eos_token,
        stream=stream,
        generation_stop_event=generation_stop_event,
        thinking_title=thinking_title,
        inline_thought_title=inline_thought_title,
        thinking_id=thinking_id,
        base_messages=base_messages or [],
        chunk_observer=chunk_observer,
    )
    yield from session
