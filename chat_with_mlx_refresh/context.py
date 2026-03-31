from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from gradio.utils import get_cache_folder, get_upload_folder

from .model import ModelManager
from .services.files import FileService
from .services.rag import RAGService


def _build_allowed_file_roots(model_manager: ModelManager) -> tuple[Path, ...]:
    roots = {
        Path(get_upload_folder()),
        Path(get_cache_folder()),
        model_manager.config_store.base_dir,
    }
    return tuple(sorted((root.expanduser().resolve() for root in roots), key=str))


@dataclass
class AppContext:
    model_manager: ModelManager = field(default_factory=ModelManager)
    file_service: FileService = field(default_factory=FileService)
    rag_service: RAGService = field(default_factory=RAGService)
    generation_stop_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        if not self.file_service.allowed_roots:
            self.file_service.set_allowed_roots(_build_allowed_file_roots(self.model_manager))

    def close(self) -> None:
        self.model_manager.close_model()
