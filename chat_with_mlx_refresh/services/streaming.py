from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from gradio.components.chatbot import ChatMessage


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
DEFAULT_PARTIAL_PREFIXES = [
    "<",
    "<t",
    "<th",
    "<thi",
    "<thin",
    "<think",
    "</",
    "</t",
    "</th",
    "</thi",
    "</thin",
    "</think",
    "<a",
    "<an",
    "<ans",
    "<answ",
    "<answe",
    "<answer",
    "</a",
    "</an",
    "</ans",
    "</answ",
    "</answe",
    "</answer",
]


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


def trim_to_eos_if_streaming(text: str, eos_token: Optional[str], stream: bool) -> tuple[str, bool]:
    if stream and eos_token and eos_token in text:
        if text == eos_token:
            return "", True
        return text.split(eos_token)[0], True
    return text, False


def filter_chatml_tokens_and_stop(text: str, control_tokens: list[str], stop_tokens: list[str]) -> tuple[str, bool]:
    should_stop = False
    filtered = text
    for stop_token in stop_tokens:
        if stop_token in filtered:
            should_stop = True
            filtered = filtered[:filtered.find(stop_token)]
            break

    for token in control_tokens:
        filtered = filtered.replace(token, "")
    return filtered, should_stop


def strip_answer_tags(text: str) -> str:
    return text.replace("<answer>", "").replace("</answer>", "")


def merge_with_partial_buffer(prior_buffer: str, text: str, partial_prefixes: list[str]) -> tuple[str, str]:
    merged = f"{prior_buffer}{text}" if prior_buffer else text
    if not merged:
        return merged, ""

    new_buffer = ""
    for partial in sorted(partial_prefixes, key=len, reverse=True):
        if merged.endswith(partial):
            new_buffer = partial
            merged = merged[:-len(partial)]
            break
    return merged, new_buffer


@dataclass
class StreamSession:
    response_stream: Iterable
    eos_token: Optional[str]
    stream: bool
    generation_stop_event: Any
    control_tokens: list[str] = field(default_factory=lambda: CHATML_CONTROL_TOKENS)
    stop_tokens: list[str] = field(default_factory=lambda: CHATML_STOP_TOKENS)
    partial_prefixes: list[str] = field(default_factory=lambda: DEFAULT_PARTIAL_PREFIXES)
    thinking_title: str = "Thinking"
    inline_thought_title: str = "Thinking"
    thinking_id: int = 0
    base_messages: list[Any] = field(default_factory=list)
    chat_message_accumulator: Any = field(default_factory=lambda: ChatMessage(role="assistant", content=""))
    thinking_message: Optional[Any] = None
    full_response: str = ""
    final_messages: list[Any] = field(default_factory=list)
    chunk_observer: Optional[Callable[[Any], None]] = None

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        final_content_parts: list[str] = []
        thinking_content_parts: list[str] = []
        in_thinking = False
        thinking_start_time: Optional[float] = None
        chunk_buffer = ""

        for chunk in self.response_stream:
            if self.generation_stop_event.is_set():
                break
            if self.chunk_observer is not None:
                self.chunk_observer(chunk)

            chunk_text = parse_chunk_text(chunk)
            if not chunk_text:
                continue

            self.full_response += chunk_text
            chunk_text, eos_hit = trim_to_eos_if_streaming(chunk_text, self.eos_token, self.stream)
            chunk_text, chunk_buffer = merge_with_partial_buffer(chunk_buffer, chunk_text, self.partial_prefixes)
            if not chunk_text:
                if eos_hit:
                    break
                continue

            chunk_text, should_stop = filter_chatml_tokens_and_stop(chunk_text, self.control_tokens, self.stop_tokens)
            chunk_text = strip_answer_tags(chunk_text)

            if not in_thinking and "<think>" in chunk_text and "</think>" in chunk_text:
                think_match = re.search(r"<think>(.*?)</think>", chunk_text, re.DOTALL)
                if think_match:
                    extracted = think_match.group(1)
                    self.thinking_message = ChatMessage(
                        role="assistant",
                        content=extracted,
                        metadata={
                            "title": self.inline_thought_title,
                            "id": self.thinking_id,
                            "status": "done",
                            "duration": 0.1,
                        },
                    )
                    before_think = chunk_text[:chunk_text.find("<think>")]
                    after_think = chunk_text[chunk_text.find("</think>") + len("</think>"):]
                    if before_think:
                        final_content_parts.append(before_think)
                    if after_think:
                        final_content_parts.append(after_think)
                    self.chat_message_accumulator.content = "".join(final_content_parts)

                    if self.stream:
                        payload = self.base_messages + [self.thinking_message]
                        if self.chat_message_accumulator.content:
                            payload.append(self.chat_message_accumulator)
                        yield payload
                    if should_stop or eos_hit:
                        break
                    continue

            if should_stop:
                if in_thinking:
                    thinking_content_parts.append(chunk_text)
                    self._ensure_thinking_message(status="pending")
                    self.thinking_message.content = "".join(thinking_content_parts)
                    self.thinking_message.metadata["status"] = "done"
                    self.thinking_message.metadata["duration"] = time.time() - (thinking_start_time or time.time())
                    if self.stream:
                        yield self.base_messages + [self.thinking_message, self.chat_message_accumulator]
                else:
                    final_content_parts.append(chunk_text)
                    self.chat_message_accumulator.content = "".join(final_content_parts)
                    if self.stream:
                        payload = self.base_messages + [self.chat_message_accumulator]
                        if self.thinking_message and self.thinking_message.metadata.get("status") == "done":
                            payload = self.base_messages + [self.thinking_message, self.chat_message_accumulator]
                        yield payload
                break

            if "<think>" in chunk_text and not in_thinking:
                in_thinking = True
                thinking_start_time = time.time()
                self._ensure_thinking_message(status="pending")
                think_open_index = chunk_text.find("<think>")
                think_start_index = think_open_index + len("<think>")

                before_think = chunk_text[:think_open_index]
                if before_think:
                    final_content_parts.append(before_think)
                    self.chat_message_accumulator.content = "".join(final_content_parts)

                if think_start_index < len(chunk_text):
                    thinking_content_parts.append(chunk_text[think_start_index:])
                    self.thinking_message.content = "".join(thinking_content_parts)

                if self.stream:
                    yield self.base_messages + [self.thinking_message]
                continue

            if in_thinking and "</think>" not in chunk_text:
                thinking_content_parts.append(chunk_text)
                self._ensure_thinking_message(status="pending")
                self.thinking_message.content = "".join(thinking_content_parts)
                if self.stream:
                    yield self.base_messages + [self.thinking_message]
                continue

            if in_thinking and "</think>" in chunk_text:
                think_end_index = chunk_text.find("</think>")
                thinking_content_parts.append(chunk_text[:think_end_index])
                self._ensure_thinking_message(status="pending")
                self.thinking_message.content = "".join(thinking_content_parts)
                self.thinking_message.metadata["status"] = "done"
                self.thinking_message.metadata["duration"] = time.time() - (thinking_start_time or time.time())

                remaining = chunk_text[think_end_index + len("</think>"):]
                if remaining.strip():
                    final_content_parts.append(remaining)
                    self.chat_message_accumulator.content = "".join(final_content_parts)

                if self.stream:
                    payload = self.base_messages + [self.thinking_message]
                    if self.chat_message_accumulator.content:
                        payload.append(self.chat_message_accumulator)
                    yield payload
                in_thinking = False
                continue

            final_content_parts.append(chunk_text)
            self.chat_message_accumulator.content = "".join(final_content_parts)
            if self.stream:
                payload = self.base_messages + [self.chat_message_accumulator]
                if self.thinking_message and self.thinking_message.metadata.get("status") == "done":
                    payload = self.base_messages + [self.thinking_message, self.chat_message_accumulator]
                yield payload

            if self.stream and eos_hit:
                break

        if self.thinking_message and self.thinking_message.metadata.get("status") == "done":
            self.final_messages = [self.thinking_message]
            if self.chat_message_accumulator.content:
                self.final_messages.append(self.chat_message_accumulator)
        elif self.chat_message_accumulator.content:
            self.final_messages = [self.chat_message_accumulator]

    def _ensure_thinking_message(self, status: str) -> None:
        if self.thinking_message:
            return
        self.thinking_message = ChatMessage(
            role="assistant",
            content="",
            metadata={"title": self.thinking_title, "id": self.thinking_id, "status": status},
        )


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
