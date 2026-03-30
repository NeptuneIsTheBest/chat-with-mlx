from __future__ import annotations

import base64
import functools
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=128)
def _get_file_md5_cached(file_path: str, mtime: float) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_file_md5(file_name: Path) -> str:
    return _get_file_md5_cached(str(file_name), file_name.stat().st_mtime)


def get_canonical_file_path(file_name: Path) -> str:
    return str(file_name.expanduser().resolve())


class FileService:
    def __init__(self) -> None:
        self.files: Dict[str, Dict[str, str]] = {}
        self.loader_map = {
            ".pdf": self.load_pdf,
            ".docx": self.load_docx,
            ".pptx": self.load_pptx,
            **dict.fromkeys([".txt", ".csv", ".md"], self.load_txt_like),
            **dict.fromkeys([".xlsx", ".xls"], self.load_excel),
        }

    def format_content(self, file_name: Path, content: str) -> str:
        boundary_start = f"<<<BEGIN FILE:{file_name}>>>"
        boundary_end = f"<<<END FILE:{file_name}>>>"
        sanitized = content.replace(boundary_start, "").replace(boundary_end, "")
        return f"{boundary_start}\n{sanitized}\n{boundary_end}"

    def _get_cached_content(self, file_name: Path, raw_content_only: bool) -> Optional[str]:
        file_key = get_canonical_file_path(file_name)
        cached = self.files.get(file_key)
        if not cached:
            return None

        if cached["md5"] != get_file_md5(file_name):
            return None
        return cached["content" if raw_content_only else "formatted_content"]

    def _store_content(self, file_name: Path, content: str) -> None:
        self.files[get_canonical_file_path(file_name)] = {
            "md5": get_file_md5(file_name),
            "formatted_content": self.format_content(file_name, content),
            "content": content,
        }

    def load_file(self, file_name: Path, raw_content_only: bool = False) -> Optional[str]:
        cached_content = self._get_cached_content(file_name, raw_content_only)
        if cached_content is not None:
            return cached_content

        loader = self.loader_map.get(file_name.suffix.lower())
        return loader(file_name, raw_content_only) if loader else None

    def load_pdf(self, file_name: Path, raw_content_only: bool = False) -> Optional[str]:
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf not found.")
            return None

        pdf = PdfReader(file_name)
        content = "".join(text for page in pdf.pages if (text := page.extract_text()))
        self._store_content(file_name, content)
        cached = self.files[get_canonical_file_path(file_name)]
        return cached["content" if raw_content_only else "formatted_content"]

    def load_txt_like(self, file_name: Path, raw_content_only: bool = False) -> str:
        with open(file_name, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
        self._store_content(file_name, content)
        cached = self.files[get_canonical_file_path(file_name)]
        return cached["content" if raw_content_only else "formatted_content"]

    def load_docx(self, file_name: Path, raw_content_only: bool = False) -> Optional[str]:
        try:
            from docx import Document
        except ImportError:
            logger.warning("docx not found.")
            return None

        document = Document(str(file_name))
        content = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self._store_content(file_name, content)
        cached = self.files[get_canonical_file_path(file_name)]
        return cached["content" if raw_content_only else "formatted_content"]

    def load_pptx(self, file_name: Path, raw_content_only: bool = False) -> Optional[str]:
        try:
            from pptx import Presentation
        except ImportError:
            logger.warning("pptx not found.")
            return None

        presentation = Presentation(str(file_name))
        content_parts = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    content_parts.append(shape.text)
        content = "\n".join(content_parts)
        self._store_content(file_name, content)
        cached = self.files[get_canonical_file_path(file_name)]
        return cached["content" if raw_content_only else "formatted_content"]

    def load_excel(self, file_name: Path, raw_content_only: bool = False) -> Optional[str]:
        try:
            from pandas import ExcelFile, read_excel
        except ImportError:
            logger.warning("pandas not found.")
            return None

        excel_file = ExcelFile(file_name)
        content_parts = []
        for sheet_name in excel_file.sheet_names:
            data_frame = read_excel(excel_file, sheet_name=sheet_name)
            content_parts.append(f"Sheet: {sheet_name}\n")
            content_parts.append(data_frame.to_csv(index=False))
        content = "".join(content_parts)
        self._store_content(file_name, content)
        cached = self.files[get_canonical_file_path(file_name)]
        return cached["content" if raw_content_only else "formatted_content"]

    def encode_image(self, image_path: Path) -> str:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".heif": "image/heif",
            ".heic": "image/heic",
        }
        suffix = image_path.suffix.lower()
        if suffix not in mime_types:
            raise RuntimeError(f"Unsupported image type: {suffix}")

        with open(image_path, "rb") as file_obj:
            encoded = base64.b64encode(file_obj.read()).decode("utf-8")
        return f"data:{mime_types[suffix]};base64,{encoded}"

    def clear(self) -> None:
        self.files.clear()
