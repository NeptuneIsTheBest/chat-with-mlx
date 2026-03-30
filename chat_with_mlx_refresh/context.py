from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .model import ModelManager
from .services.files import FileService
from .services.rag import RAGService


@dataclass
class AppContext:
    model_manager: ModelManager = field(default_factory=ModelManager)
    file_service: FileService = field(default_factory=FileService)
    rag_service: RAGService = field(default_factory=RAGService)
    generation_stop_event: threading.Event = field(default_factory=threading.Event)

    def close(self) -> None:
        self.model_manager.close_model()
