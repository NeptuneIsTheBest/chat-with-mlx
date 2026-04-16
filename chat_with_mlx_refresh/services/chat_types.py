from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


AttachmentKind = Literal["document", "image", "audio"]


@dataclass(slots=True, frozen=True)
class AttachmentRef:
    kind: AttachmentKind
    path: str


@dataclass(slots=True)
class TurnMedia:
    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "images": list(self.images),
            "audios": list(self.audios),
        }


@dataclass(slots=True)
class NormalizedTurn:
    role: str
    text: str
    attachments: list[AttachmentRef] = field(default_factory=list)


@dataclass(slots=True)
class PreparedPrompt:
    formatted_prompt: str
    prompt_token_ids: list[int]
    prepared_inputs: dict[str, Any] | None
    flat_images: list[str]
    flat_audios: list[str]
    turn_media_sequence: list[TurnMedia]


def flatten_turn_media(turn_media_sequence: Sequence[TurnMedia]) -> tuple[list[str], list[str]]:
    images = [path for turn_media in turn_media_sequence for path in turn_media.images]
    audios = [path for turn_media in turn_media_sequence for path in turn_media.audios]
    return images, audios


def normalize_turn_media_sequence(
    turn_media_sequence: Sequence[TurnMedia],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    return [
        (
            tuple(str(path) for path in turn_media.images),
            tuple(str(path) for path in turn_media.audios),
        )
        for turn_media in turn_media_sequence
    ]


def turn_media_sequence_to_dicts(turn_media_sequence: Sequence[TurnMedia]) -> list[dict[str, list[str]]]:
    return [turn_media.to_dict() for turn_media in turn_media_sequence]
