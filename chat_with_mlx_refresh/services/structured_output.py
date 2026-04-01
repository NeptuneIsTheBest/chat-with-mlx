from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

from gradio.components.chatbot import ChatMessage


def build_partial_prefixes(tokens: Sequence[str]) -> list[str]:
    prefixes = {token[:index] for token in tokens for index in range(1, len(token))}
    return sorted(prefixes, key=len, reverse=True)


def merge_with_partial_buffer(prior_buffer: str, text: str, partial_prefixes: Sequence[str]) -> tuple[str, str]:
    merged = f"{prior_buffer}{text}" if prior_buffer else text
    if not merged:
        return "", ""

    for partial in partial_prefixes:
        if merged.endswith(partial):
            return merged[:-len(partial)], partial
    return merged, ""


def filter_chatml_tokens_and_stop(text: str, control_tokens: Sequence[str], stop_tokens: Sequence[str]) -> tuple[str, bool]:
    should_stop = False
    filtered = text
    for stop_token in stop_tokens:
        if stop_token in filtered:
            filtered = filtered.split(stop_token, 1)[0]
            should_stop = True
            break

    for token in control_tokens:
        filtered = filtered.replace(token, "")
    return filtered, should_stop


@dataclass(frozen=True)
class SanitizedChunk:
    text: str
    stop: bool = False


class IncrementalTextSanitizer:
    def __init__(
        self,
        *,
        control_tokens: Sequence[str],
        stop_tokens: Sequence[str],
        eos_token: Optional[str] = None,
    ) -> None:
        special_tokens = [token for token in [eos_token, *control_tokens, *stop_tokens] if token]
        self.control_tokens = list(control_tokens)
        self.stop_tokens = list(stop_tokens)
        self.eos_token = eos_token
        self.partial_prefixes = build_partial_prefixes(special_tokens)
        self.partial_buffer = ""

    def feed(self, text: str) -> SanitizedChunk:
        merged, self.partial_buffer = merge_with_partial_buffer(self.partial_buffer, text, self.partial_prefixes)
        if not merged:
            return SanitizedChunk("")

        filtered, should_stop = self._sanitize_text(merged)
        return SanitizedChunk(filtered, should_stop)

    def finalize(self) -> SanitizedChunk:
        if not self.partial_buffer:
            return SanitizedChunk("")

        merged = self.partial_buffer
        self.partial_buffer = ""
        filtered, should_stop = self._sanitize_text(merged)
        return SanitizedChunk(filtered, should_stop)

    def _sanitize_text(self, text: str) -> tuple[str, bool]:
        should_stop = False
        filtered = text
        if self.eos_token and self.eos_token in filtered:
            filtered = filtered.split(self.eos_token, 1)[0]
            should_stop = True

        filtered, stop_hit = filter_chatml_tokens_and_stop(
            filtered,
            self.control_tokens,
            self.stop_tokens,
        )
        return filtered, should_stop or stop_hit


@dataclass(frozen=True)
class StructuredTagSpec:
    name: str
    title: str
    transparent: bool = False


@dataclass(frozen=True)
class TagToken:
    kind: Literal["open", "close"]
    name: str
    raw: str
    attrs: dict[str, str] = field(default_factory=dict)
    self_closing: bool = False


@dataclass(frozen=True)
class TextToken:
    text: str


StreamToken = TextToken | TagToken


_INCOMPLETE = object()
_TAG_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-")
_LOG_ATTR_KEYS = ("name", "tool", "server", "method")


def create_default_structured_tag_specs(thinking_title: str = "Thinking") -> dict[str, StructuredTagSpec]:
    specs = [
        StructuredTagSpec(name="answer", title="Answer", transparent=True),
        StructuredTagSpec(name="think", title=thinking_title),
        StructuredTagSpec(name="mcp_call", title="MCP Call"),
        StructuredTagSpec(name="mcp_result", title="MCP Result"),
        StructuredTagSpec(name="tool_call", title="Tool Call"),
        StructuredTagSpec(name="tool_result", title="Tool Result"),
    ]
    return {spec.name: spec for spec in specs}


def is_structured_ui_message(history_item: dict[str, object]) -> bool:
    if not isinstance(history_item, dict):
        return False

    metadata = history_item.get("metadata")
    if not isinstance(metadata, dict):
        return False

    if metadata.get("_ui_block_type") == "structured_tag":
        return True
    return metadata.get("title") == "Thinking"


class StructuredTagParser:
    def __init__(self, tag_specs: dict[str, StructuredTagSpec]) -> None:
        self.tag_specs = tag_specs
        self.known_tag_names = tuple(sorted(tag_specs))
        self.buffer = ""

    def feed(self, text: str = "", *, final: bool = False) -> list[StreamToken]:
        if text:
            self.buffer += text

        tokens: list[StreamToken] = []
        while self.buffer:
            open_index = self.buffer.find("<")
            if open_index < 0:
                tokens.append(TextToken(self.buffer))
                self.buffer = ""
                break

            if open_index > 0:
                tokens.append(TextToken(self.buffer[:open_index]))
                self.buffer = self.buffer[open_index:]
                continue

            parsed = self._parse_tag_token(self.buffer, final=final)
            if parsed is _INCOMPLETE:
                break

            if parsed is None:
                tokens.append(TextToken(self.buffer[0]))
                self.buffer = self.buffer[1:]
                continue

            tokens.append(parsed)
            self.buffer = self.buffer[len(parsed.raw) :]

        if final and self.buffer:
            tokens.append(TextToken(self.buffer))
            self.buffer = ""

        return [token for token in tokens if not isinstance(token, TextToken) or token.text]

    def _parse_tag_token(self, value: str, *, final: bool) -> TagToken | object | None:
        if not value.startswith("<"):
            return None

        if len(value) == 1 and not final:
            return _INCOMPLETE

        closing = value.startswith("</")
        index = 2 if closing else 1
        if index >= len(value):
            return None if final else _INCOMPLETE

        if value[index].isspace() or value[index] in ">!/":
            return None

        name_start = index
        while index < len(value) and value[index] in _TAG_NAME_CHARS:
            index += 1

        candidate = value[name_start:index].lower()
        if not candidate:
            return None

        if index == len(value):
            return self._resolve_partial(candidate, final)

        if candidate not in self.tag_specs:
            return self._resolve_partial(candidate, final)

        if closing:
            return self._parse_close_tag(value, index, candidate, final)
        return self._parse_open_tag(value, index, candidate, final)

    def _resolve_partial(self, candidate: str, final: bool) -> object | None:
        if not final and any(name.startswith(candidate) for name in self.known_tag_names):
            return _INCOMPLETE
        return None

    @staticmethod
    def _skip_whitespace(value: str, index: int) -> int:
        while index < len(value) and value[index].isspace():
            index += 1
        return index

    def _parse_close_tag(self, value: str, index: int, candidate: str, final: bool) -> TagToken | object | None:
        index = self._skip_whitespace(value, index)
        if index >= len(value):
            return None if final else _INCOMPLETE
        if value[index] != ">":
            return None
        return TagToken(kind="close", name=candidate, raw=value[: index + 1])

    def _parse_open_tag(self, value: str, index: int, candidate: str, final: bool) -> TagToken | object | None:
        tag_end = self._find_tag_end(value, index, final)
        if tag_end is _INCOMPLETE:
            return _INCOMPLETE
        if tag_end is None:
            return None

        raw = value[: tag_end + 1]
        attr_source = value[index:tag_end]
        stripped = attr_source.rstrip()
        self_closing = stripped.endswith("/")
        if self_closing:
            attr_source = stripped[:-1]

        attrs = self._parse_attributes(attr_source)
        return TagToken(
            kind="open",
            name=candidate,
            raw=raw,
            attrs=attrs,
            self_closing=self_closing,
        )

    def _find_tag_end(self, value: str, index: int, final: bool) -> int | object | None:
        in_quote: Optional[str] = None
        cursor = index
        while cursor < len(value):
            char = value[cursor]
            if in_quote:
                if char == in_quote:
                    in_quote = None
            elif char in {'"', "'"}:
                in_quote = char
            elif char == ">":
                return cursor
            cursor += 1
        return None if final else _INCOMPLETE

    @staticmethod
    def _parse_attributes(source: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        index = 0
        length = len(source)
        while index < length:
            while index < length and source[index].isspace():
                index += 1
            if index >= length:
                break

            name_start = index
            while index < length and source[index] in _TAG_NAME_CHARS:
                index += 1
            name = source[name_start:index].lower()
            if not name:
                break

            while index < length and source[index].isspace():
                index += 1
            if index >= length or source[index] != "=":
                attrs[name] = ""
                continue

            index += 1
            while index < length and source[index].isspace():
                index += 1
            if index >= length:
                attrs[name] = ""
                break

            quote = source[index]
            if quote in {'"', "'"}:
                index += 1
                value_start = index
                while index < length and source[index] != quote:
                    index += 1
                attrs[name] = source[value_start:index]
                if index < length:
                    index += 1
                continue

            value_start = index
            while index < length and not source[index].isspace():
                index += 1
            attrs[name] = source[value_start:index]
        return attrs


@dataclass
class TagFrame:
    spec: StructuredTagSpec
    raw_open: str
    attrs: dict[str, str]
    visible_id: Optional[int]
    parent_visible_id: Optional[int]
    started_at: float = field(default_factory=time.time)
    raw_close: str = ""
    closed: bool = False
    closed_at: Optional[float] = None
    children: list[str | TagFrame] = field(default_factory=list)

    def append_text(self, text: str) -> None:
        if not text:
            return
        if self.children and isinstance(self.children[-1], str):
            self.children[-1] += text
        else:
            self.children.append(text)

    def close(self, raw_close: str = "") -> None:
        self.closed = True
        self.raw_close = raw_close
        self.closed_at = time.time()

    def serialize_raw(self) -> str:
        content = "".join(
            child if isinstance(child, str) else child.serialize_raw()
            for child in self.children
        )
        return f"{self.raw_open}{content}{self.raw_close}"


class StructuredMessageAssembler:
    def __init__(
        self,
        tag_specs: dict[str, StructuredTagSpec],
        *,
        starting_id: int = 0,
    ) -> None:
        self.tag_specs = tag_specs
        self.root_items: list[str | TagFrame] = []
        self.stack: list[TagFrame] = []
        self.next_visible_id = starting_id

    def consume(self, tokens: Iterable[StreamToken]) -> None:
        for token in tokens:
            if isinstance(token, TextToken):
                self._append_text(token.text)
                continue

            if token.kind == "open":
                self._open_frame(token)
                continue

            self._close_frame(token)

    def snapshot(self, *, finalize: bool = False) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        self._build_messages(
            items=self.root_items,
            finalize=finalize,
            visible_parent=None,
            messages=messages,
        )
        return messages

    def _append_text(self, text: str) -> None:
        if not text:
            return
        items = self._current_items()
        if items and isinstance(items[-1], str):
            items[-1] += text
        else:
            items.append(text)

    def _current_items(self) -> list[str | TagFrame]:
        return self.stack[-1].children if self.stack else self.root_items

    def _nearest_visible_frame(self) -> Optional[TagFrame]:
        for frame in reversed(self.stack):
            if not frame.spec.transparent:
                return frame
        return None

    def _open_frame(self, token: TagToken) -> None:
        spec = self.tag_specs[token.name]
        visible_parent = self._nearest_visible_frame()
        frame = TagFrame(
            spec=spec,
            raw_open=token.raw,
            attrs=token.attrs,
            visible_id=None if spec.transparent else self._next_visible_id(),
            parent_visible_id=None if visible_parent is None else visible_parent.visible_id,
        )
        self._current_items().append(frame)
        if token.self_closing:
            frame.close()
            return
        self.stack.append(frame)

    def _close_frame(self, token: TagToken) -> None:
        if self.stack and self.stack[-1].spec.name == token.name:
            self.stack.pop().close(token.raw)
            return
        self._append_text(token.raw)

    def _next_visible_id(self) -> int:
        current = self.next_visible_id
        self.next_visible_id += 1
        return current

    def _build_messages(
        self,
        *,
        items: list[str | TagFrame],
        finalize: bool,
        visible_parent: Optional[TagFrame],
        messages: list[ChatMessage],
    ) -> None:
        for item in items:
            if isinstance(item, str):
                if visible_parent is None:
                    self._append_message_text(messages, item)
                continue

            if finalize and not item.closed:
                if visible_parent is None:
                    self._append_message_text(messages, item.serialize_raw())
                continue

            if item.spec.transparent:
                self._build_messages(
                    items=item.children,
                    finalize=finalize,
                    visible_parent=visible_parent,
                    messages=messages,
                )
                continue

            messages.append(self._build_block_message(item, finalize=finalize))
            self._build_messages(
                items=item.children,
                finalize=finalize,
                visible_parent=item,
                messages=messages,
            )

    @staticmethod
    def _append_message_text(messages: list[ChatMessage], text: str) -> None:
        if not text:
            return
        if messages and not messages[-1].metadata:
            if isinstance(messages[-1].content, str):
                messages[-1].content += text
                return
        messages.append(ChatMessage(role="assistant", content=text))

    def _build_block_message(self, frame: TagFrame, *, finalize: bool) -> ChatMessage:
        metadata: dict[str, object] = {
            "title": frame.spec.title,
            "id": frame.visible_id,
            "status": "done" if frame.closed else "pending",
            "_ui_block_type": "structured_tag",
            "_tag_name": frame.spec.name,
        }
        if frame.parent_visible_id is not None:
            metadata["parent_id"] = frame.parent_visible_id

        log_value = self._extract_log_value(frame.attrs)
        if log_value:
            metadata["log"] = log_value
        if frame.closed and frame.closed_at is not None:
            metadata["duration"] = max(0.0, frame.closed_at - frame.started_at)

        return ChatMessage(
            role="assistant",
            content=self._collect_visible_text(frame.children, finalize=finalize),
            metadata=metadata,
        )

    def _collect_visible_text(self, items: list[str | TagFrame], *, finalize: bool) -> str:
        parts: list[str] = []
        for item in items:
            if isinstance(item, str):
                parts.append(item)
                continue

            if finalize and not item.closed:
                parts.append(item.serialize_raw())
                continue

            if item.spec.transparent:
                parts.append(self._collect_visible_text(item.children, finalize=finalize))
        return "".join(parts)

    @staticmethod
    def _extract_log_value(attrs: dict[str, str]) -> Optional[str]:
        for key in _LOG_ATTR_KEYS:
            value = attrs.get(key)
            if value:
                return value
        return None
