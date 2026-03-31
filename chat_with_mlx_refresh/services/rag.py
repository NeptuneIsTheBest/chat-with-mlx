from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .error_handling import (
    get_exception_message,
    gradio_error_boundary,
    log_service_exception,
)
from .files import FileService, get_canonical_file_path


logger = logging.getLogger(__name__)


class RAGService:
    COLLECTION_NAME = "rag_collection"
    COLLECTION_METADATA = {"hnsw:space": "cosine"}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persistence: bool = False, db_path: str = "./chromadb"):
        self._lock = threading.RLock()
        if persistence:
            self.client = chromadb.PersistentClient(
                db_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(settings=chromadb.Settings(anonymized_telemetry=False))

        self.model_name = model_name
        self.embedding_model = None
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.n_results = 5
        self.similarity_threshold = 0.0
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.collection: Optional[Any] = self._create_collection()
        self.indexed_sources: dict[str, str] = {}
        self._rehydrate_indexed_sources()
        self.enabled = False

    def _create_collection(self) -> Any:
        with self._lock:
            return self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata=self.COLLECTION_METADATA,
            )

    def _get_collection(self) -> Any:
        with self._lock:
            if self.collection is None:
                self.collection = self._create_collection()
            return self.collection

    def _get_status_text_safe(self) -> str:
        try:
            return self.get_status_text()
        except Exception as exc:
            log_service_exception(
                logger,
                "read the RAG status text",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return "RAG index status unavailable."

    def _ensure_embedding_model(self) -> Any:
        with self._lock:
            if self.embedding_model is None:
                logger.info("Lazy loading embedding model for RAG...")
                self.embedding_model = SentenceTransformer(self.model_name)
            return self.embedding_model

    @staticmethod
    def _get_source_path(file_path: Path) -> str:
        return get_canonical_file_path(file_path)

    @classmethod
    def _build_source_id(cls, source_path: str, file_content: str) -> str:
        content_hash = hashlib.sha1(file_content.encode("utf-8")).hexdigest()
        source_key = f"{source_path}:{content_hash}"
        return hashlib.sha1(source_key.encode("utf-8")).hexdigest()

    def _delete_stale_source_ids(self, collection: Any, source_ids: Iterable[str], source_label: str) -> None:
        for stale_source_id in source_ids:
            try:
                collection.delete(where={"source_id": stale_source_id})
            except Exception as exc:
                log_service_exception(
                    logger,
                    f"remove stale RAG chunks for '{source_label}'",
                    exc,
                    level=logging.WARNING,
                    include_traceback=False,
                )

    def _rehydrate_indexed_sources(self) -> None:
        try:
            collection = self._get_collection()
            payload = collection.get(include=["metadatas"])
        except Exception as exc:
            log_service_exception(
                logger,
                "rehydrate the RAG index state",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return

        indexed_sources: dict[str, str] = {}
        for metadata in payload.get("metadatas") or []:
            if not isinstance(metadata, dict):
                continue
            source_name = metadata.get("source_name")
            source_path = metadata.get("source_path")
            source_id = metadata.get("source_id")
            source_key = source_path or source_name
            if source_key and source_id:
                indexed_sources[source_key] = source_id

        with self._lock:
            self.indexed_sources = indexed_sources

    def update_parameters(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        n_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> bool:
        with self._lock:
            updated = False
            if chunk_size is not None and chunk_size != self.chunk_size:
                self.chunk_size = chunk_size
                updated = True
            if chunk_overlap is not None and chunk_overlap != self.chunk_overlap:
                self.chunk_overlap = chunk_overlap
                updated = True
            if n_results is not None:
                self.n_results = n_results
            if similarity_threshold is not None:
                self.similarity_threshold = similarity_threshold

            if updated:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            return updated

    def add_document(self, file_path: Path, file_content: str) -> tuple[bool, str]:
        source_name = file_path.name
        source_path = self._get_source_path(file_path)
        source_id = self._build_source_id(source_path, file_content)
        with self._lock:
            existing_source_id = self.indexed_sources.get(source_path)
            legacy_source_id = self.indexed_sources.get(source_name)
            stale_source_ids = {
                candidate
                for candidate in (existing_source_id, legacy_source_id)
                if candidate is not None and candidate != source_id
            }

            if existing_source_id == source_id:
                if stale_source_ids:
                    self._delete_stale_source_ids(self._get_collection(), stale_source_ids, source_name)
                    self.indexed_sources[source_path] = source_id
                    self.indexed_sources.pop(source_name, None)
                return False, f"Document '{file_path.name}' is already indexed."

            chunks = self.text_splitter.split_text(file_content)
            if not chunks:
                return False, f"No content to index in document '{file_path.name}'."

            embeddings = self._ensure_embedding_model().encode(chunks, show_progress_bar=True, batch_size=32)
            ids = [f"{source_id}_{index}" for index in range(len(chunks))]
            metadata = [
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_path": source_path,
                    "chunk_id": index,
                }
                for index in range(len(chunks))
            ]

            collection = self._get_collection()
            collection.add(embeddings=embeddings, documents=chunks, metadatas=metadata, ids=ids)

            if stale_source_ids:
                self._delete_stale_source_ids(collection, stale_source_ids, source_name)

            self.indexed_sources[source_path] = source_id
            self.indexed_sources.pop(source_name, None)
            return True, f"Document '{file_path.name}' indexed with {len(chunks)} chunks."

    def retrieve(self, query: str, n_results: Optional[int] = None) -> str:
        with self._lock:
            result_count = n_results if n_results is not None else self.n_results
            similarity_threshold = self.similarity_threshold

        collection = self._get_collection()
        collection_count = collection.count()
        if collection_count == 0:
            return ""

        query_embeddings = self._ensure_embedding_model().encode([query], show_progress_bar=False)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(result_count, collection_count),
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        distances = results.get("distances")
        if not distances or distances[0] is None:
            return "\n\n".join(documents)

        filtered_docs = []
        for index, document in enumerate(documents):
            distance = distances[0][index] if index < len(distances[0]) else None
            if distance is None:
                filtered_docs.append(document)
                continue

            similarity = 1 - distance
            if similarity >= similarity_threshold:
                filtered_docs.append(document)

        return "\n\n".join(filtered_docs) if filtered_docs else ""

    def clear_index(self) -> None:
        with self._lock:
            if self.collection is None:
                self.collection = self._create_collection()
                self.indexed_sources.clear()
                return

            try:
                self.client.delete_collection(name=self.collection.name)
            except Exception as exc:
                raise RuntimeError(f"Failed to delete RAG collection '{self.collection.name}': {exc}") from exc

            self.indexed_sources.clear()

            try:
                self.collection = self._create_collection()
            except Exception as exc:
                self.collection = None
                raise RuntimeError(f"Failed to recreate RAG collection '{self.COLLECTION_NAME}': {exc}") from exc

    def enable(self) -> None:
        with self._lock:
            self._ensure_embedding_model()
            self.enabled = True

    def disable(self) -> None:
        with self._lock:
            self.enabled = False

    def is_enabled(self) -> bool:
        with self._lock:
            return self.enabled

    def get_status(self) -> tuple[bool, str]:
        with self._lock:
            indexed_count = len(self.indexed_sources)

        total_chunks = self._get_collection().count()
        if indexed_count == 0 and total_chunks > 0:
            self._rehydrate_indexed_sources()
            with self._lock:
                indexed_count = len(self.indexed_sources)

        if indexed_count == 0:
            return False, "RAG Index is empty."
        return True, f"Indexed {indexed_count} documents with {total_chunks} chunks."

    def get_status_text(self) -> str:
        try:
            return self.get_status()[1]
        except Exception as exc:
            log_service_exception(
                logger,
                "read the RAG status",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return "RAG index status unavailable."

    def get_parameters(self) -> dict[str, float]:
        with self._lock:
            return {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "n_results": self.n_results,
                "similarity_threshold": self.similarity_threshold,
            }

    def get_parameter_tuple(self) -> tuple[int, int, int, float]:
        params = self.get_parameters()
        return (
            params["chunk_size"],
            params["chunk_overlap"],
            params["n_results"],
            params["similarity_threshold"],
        )

    @gradio_error_boundary("upload files into the RAG index", logger)
    def upload_and_index_files(self, files: Optional[Iterable[Any]], file_service: FileService) -> tuple[str, str]:
        if not files:
            return "No files selected.", self.get_status_text()

        results = []
        for file_ref in files:
            file_path = Path(getattr(file_ref, "name", file_ref))
            try:
                file_content = file_service.load_file_uncached(file_path, raw_content_only=True)
                if file_content:
                    _, message = self.add_document(file_path, file_content)
                    results.append(f"{file_path.name}: {message}")
                else:
                    results.append(f"{file_path.name}: Failed to load content")
            except Exception as exc:
                log_service_exception(
                    logger,
                    f"index RAG document '{file_path.name}'",
                    exc,
                    level=logging.WARNING,
                    include_traceback=False,
                )
                results.append(f"{file_path.name}: Error - {get_exception_message(exc)}")
        return "\n".join(results), self.get_status_text()

    def clear_index_with_status(self) -> tuple[str, str]:
        try:
            self.clear_index()
            return "RAG index cleared successfully.", self._get_status_text_safe()
        except Exception as exc:
            log_service_exception(logger, "clear the RAG index", exc)
            return f"Error clearing RAG index: {get_exception_message(exc)}", self._get_status_text_safe()

    @gradio_error_boundary("toggle RAG mode", logger)
    def toggle_enabled(self, enabled: bool) -> str:
        if enabled:
            self.enable()
        else:
            self.disable()
        return f"RAG {'enabled' if enabled else 'disabled'}."

    def update_ui_parameters(
        self,
        chunk_size: int,
        chunk_overlap: int,
        n_results: int,
        similarity_threshold: float,
    ) -> str:
        try:
            updated = self.update_parameters(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                n_results=n_results,
                similarity_threshold=similarity_threshold,
            )
        except Exception as exc:
            log_service_exception(logger, "update RAG parameters", exc)
            return f"Error updating RAG parameters: {get_exception_message(exc)}"

        if updated:
            return "RAG parameters updated. Consider re-indexing documents for optimal performance."
        return "RAG parameters updated."

    def enhance_message(self, message_text: str, rag_enabled: bool, n_results: int = 5) -> str:
        if not rag_enabled:
            return message_text

        try:
            retrieved_docs = self.retrieve(message_text, n_results=n_results)
        except Exception as exc:
            log_service_exception(
                logger,
                "retrieve documents from the RAG index",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return message_text

        if not retrieved_docs:
            return message_text

        return """
            Based on the following relevant documents: {}

            ---

            User question: {}
            """.format(retrieved_docs, message_text)
