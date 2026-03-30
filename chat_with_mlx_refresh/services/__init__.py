from __future__ import annotations

from importlib import import_module
from typing import Any


_SERVICE_EXPORTS = {
    "ChatService": ".chat",
    "CompletionService": ".completion",
    "FileService": ".files",
    "ModelManagementService": ".model_management",
    "RAGService": ".rag",
    "RuntimeService": ".runtime",
}

__all__ = [
    "ChatService",
    "CompletionService",
    "FileService",
    "ModelManagementService",
    "RAGService",
    "RuntimeService",
]


def __getattr__(name: str) -> Any:
    module_path = _SERVICE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
