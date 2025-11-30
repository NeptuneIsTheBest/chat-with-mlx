import argparse
import atexit
import base64
import copy
import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

import chromadb
import gradio as gr
from gradio.components.chatbot import ChatMessage
from huggingface_hub import HfApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from .language import get_text
from .model import Message, MessageRole, ModelManager, TextModel, VisionModel, BaseLocalModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_manager = ModelManager()
generation_stop_event = threading.Event()


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: Optional[Callable] = None
    enabled: bool = True

    def to_openai_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "enabled": self.enabled
        }


class FunctionManager:
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.enabled = False
        self.execution_mode = "execute"
        self._initialize_builtin_functions()

    def _initialize_builtin_functions(self):
        calculator_function = FunctionDefinition(
            name="calculate",
            description="Calculate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            implementation=self._calculate
        )
        self.add_function(calculator_function)

    @staticmethod
    def _calculate(expression: str) -> float:
        import ast
        import operator as op

        ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Mod: op.mod,
            ast.Pow: op.pow,
            ast.UAdd: lambda x: +x,
            ast.USub: lambda x: -x,
        }

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value!r}")

            if hasattr(ast, "Num") and isinstance(node, ast.Constant):
                return node.n

            if isinstance(node, ast.BinOp):
                if type(node.op) not in ops:
                    raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")
                left = eval_node(node.left)
                right = eval_node(node.right)
                return ops[type(node.op)](left, right)

            if isinstance(node, ast.UnaryOp):
                if type(node.op) not in ops:
                    raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
                operand = eval_node(node.operand)
                return ops[type(node.op)](operand)

            forbidden = (
                ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.List, ast.Tuple,
                ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                ast.BoolOp, ast.Compare, ast.IfExp, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom
            )
            if isinstance(node, forbidden):
                raise ValueError(f"Unsupported expression: {ast.dump(node)}")

            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = eval_node(tree)
            return float(result)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}") from e

    def set_execution_mode(self, mode: str):
        mode = (mode or "").lower().strip()
        if mode not in ("execute", "simulate"):
            raise ValueError("Invalid execution mode. Use 'execute' or 'simulate'.")
        self.execution_mode = mode

    def get_execution_mode(self) -> str:
        return self.execution_mode

    def get_function_names(self) -> List[str]:
        return sorted(self.functions.keys())

    def toggle_all(self, enabled: bool) -> int:
        changed = 0
        for func in self.functions.values():
            if func.enabled != enabled:
                func.enabled = enabled
                changed += 1
        return changed

    def export_custom_functions(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.functions.values() if f.implementation is None]

    def import_functions(self, functions_payload: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> Tuple[bool, str]:
        try:
            if isinstance(functions_payload, str):
                data = json.loads(functions_payload)
            else:
                data = functions_payload

            if isinstance(data, dict) and "functions" in data:
                items = data["functions"]
            elif isinstance(data, list):
                items = data
            else:
                return False, "Invalid import format. Expect list or {'functions': [...]}"

            added, skipped = 0, 0
            for item in items:
                name = item.get("name")
                desc = item.get("description", "")
                params = item.get("parameters")
                if not name or not params:
                    skipped += 1
                    continue
                ok, _ = self.add_custom_function(name, desc, json.dumps(params, ensure_ascii=False))
                if ok:
                    added += 1
                else:
                    skipped += 1

            return True, f"Import done. Added: {added}, Skipped: {skipped}"
        except Exception as e:
            return False, f"Import error: {e}"

    def get_execution_history_rows(self, limit: int = 50) -> List[List[str]]:
        rows = []
        for rec in reversed(self.execution_history[-limit:]):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.get("timestamp", time.time())))
            func = rec.get("function", "")
            args = json.dumps(rec.get("arguments", {}), ensure_ascii=False)
            if "result" in rec:
                res = json.dumps(rec.get("result"), ensure_ascii=False)
                err = ""
            else:
                res = ""
                err = str(rec.get("error", ""))
            rows.append([ts, func, args, res, err])
        return rows

    def add_function(self, function: FunctionDefinition) -> bool:
        if function.name in self.functions:
            return False
        self.functions[function.name] = function
        return True

    def add_custom_function(self, name: str, description: str, parameters_json: str) -> Tuple[bool, str]:
        try:
            parameters = json.loads(parameters_json)
            function = FunctionDefinition(
                name=name,
                description=description,
                parameters=parameters,
                implementation=None
            )
            if self.add_function(function):
                return True, f"Function '{name}' added successfully"
            else:
                return False, f"Function '{name}' already exists"
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON parameters: {e}"
        except Exception as e:
            return False, f"Error adding function: {e}"

    def remove_function(self, name: str) -> bool:
        if name in self.functions:
            del self.functions[name]
            return True
        return False

    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        return self.functions.get(name)

    def get_enabled_functions(self) -> List[FunctionDefinition]:
        return [f for f in self.functions.values() if f.enabled]

    def execute_function(self, name: str, arguments: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        function = self.get_function(name)
        if not function:
            return {"error": f"Function {name} not found"}

        if not function.enabled:
            return {"error": f"Function {name} is disabled"}

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON arguments: {arguments}"}

        try:
            if self.execution_mode == "simulate":
                result = f"[Simulated] Executed {name} with arguments: {json.dumps(arguments, ensure_ascii=False)}"
            else:
                if function.implementation:
                    result = function.implementation(**arguments)
                else:
                    result = f"[Simulated] Executed {name} with arguments: {json.dumps(arguments, ensure_ascii=False)}"

            execution_record = {
                "function": name,
                "arguments": arguments,
                "result": result,
                "timestamp": time.time()
            }
            self.execution_history.append(execution_record)
            return {"result": result}
        except Exception as e:
            execution_record = {
                "function": name,
                "arguments": arguments,
                "error": str(e),
                "timestamp": time.time()
            }
            self.execution_history.append(execution_record)
            return {"error": str(e)}

    def toggle_function(self, name: str, enabled: bool) -> bool:
        if name in self.functions:
            self.functions[name].enabled = enabled
            return True
        return False

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        return [f.to_openai_format() for f in self.get_enabled_functions()]

    def get_functions_list(self) -> List[List[Union[str, bool]]]:
        return [[f.name, f.description, f.enabled] for f in self.functions.values()]

    def clear_history(self):
        self.execution_history.clear()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def format_function_call_for_local_model(self, message: str) -> str:
        if not self.enabled or not self.get_enabled_functions():
            return message

        functions_str = "Available functions:\n"
        for func in self.get_enabled_functions():
            functions_str += f"- {func.name}: {func.description}\n"
            functions_str += f"  Parameters: {json.dumps(func.parameters, indent=2)}\n"

        return f"""{functions_str}

To call a function, use the format:
<function_call>
{{"name": "function_name", "arguments": {{"param": "value"}}}}
</function_call>

User message: {message}"""

    def parse_function_call_from_response(self, response: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        pattern = r'<function_call>\s*(\{.*?\})\s*</function_call>'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            try:
                function_data = json.loads(match.group(1))
                return function_data.get("name"), function_data.get("arguments", {})
            except json.JSONDecodeError:
                pass

        return None


function_manager = FunctionManager()


def get_loaded_model() -> Union[BaseLocalModel]:
    model = model_manager.get_loaded_model()
    if model is None:
        raise RuntimeError("No model loaded.")
    return model


def get_file_md5(file_name: Path) -> str:
    md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


class FileManager:
    def __init__(self):
        self.files = {}

    def format_content(self, file_name, content):
        boundary_start = f"<<<BEGIN FILE:{file_name}>>>"
        boundary_end = f"<<<END FILE:{file_name}>>>"
        content = content.replace(boundary_start, '')
        content = content.replace(boundary_end, '')
        return f"{boundary_start}\n{content}\n{boundary_end}"

    def load_file(self, file_name: Path, raw_content_only: bool = False):
        if file_name.name in self.files:
            if self.files[file_name.name]["md5"] == get_file_md5(file_name):
                return self.files[file_name.name]["formatted_content"] if not raw_content_only else self.files[file_name.name]["content"]
        suffix = file_name.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(file_name, raw_content_only)
        elif suffix in [".txt", ".csv", ".md"]:
            return self.load_txt_like(file_name, raw_content_only)
        elif suffix == ".docx":
            return self.load_docx(file_name, raw_content_only)
        elif suffix == ".pptx":
            return self.load_pptx(file_name, raw_content_only)
        elif suffix in [".xlsx", ".xls"]:
            return self.load_excel(file_name, raw_content_only)
        else:
            return None

    def load_pdf(self, file_name: Path, raw_content_only: bool = False):
        try:
            from pypdf import PdfReader

            pdf = PdfReader(file_name)
            content_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            content = ''.join(content_parts)
            formatted_content = self.format_content(file_name, content)
            self.files[file_name.name] = {
                "md5": get_file_md5(file_name),
                "formatted_content": formatted_content,
                "content": content
            }
            return formatted_content if not raw_content_only else content
        except ImportError:
            logger.warning("pypdf not found.")
            return None

    def load_txt_like(self, file_name: Path, raw_content_only: bool = False):
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()
        formatted_content = self.format_content(file_name, content)
        self.files[file_name.name] = {
            "md5": get_file_md5(file_name),
            "formatted_content": formatted_content,
            "content": content
        }
        return formatted_content if not raw_content_only else content

    def load_docx(self, file_name: Path, raw_content_only: bool = False):
        try:
            from docx import Document

            doc = Document(str(file_name))
            content = "\n".join([para.text for para in doc.paragraphs])
            formatted_content = self.format_content(file_name, content)
            self.files[file_name.name] = {
                "md5": get_file_md5(file_name),
                "formatted_content": formatted_content,
                "content": content
            }
            return formatted_content if not raw_content_only else content
        except ImportError:
            logger.warning("docx not found.")
            return None

    def load_pptx(self, file_name: Path, raw_content_only: bool = False):
        try:
            from pptx import Presentation

            prs = Presentation(str(file_name))
            content_parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        content_parts.append(shape.text)
            content = "\n".join(content_parts)
            formatted_content = self.format_content(file_name, content)
            self.files[file_name.name] = {
                "md5": get_file_md5(file_name),
                "formatted_content": formatted_content,
                "content": content
            }
            return formatted_content if not raw_content_only else content
        except ImportError:
            logger.warning("pptx not found.")
            return None

    def load_excel(self, file_name: Path, raw_content_only: bool = False):
        try:
            from pandas import ExcelFile, read_excel
            excel_file = ExcelFile(file_name)
            content_parts = []

            for sheet_name in excel_file.sheet_names:
                df = read_excel(excel_file, sheet_name=sheet_name)
                content_parts.append(f"Sheet: {sheet_name}\n")
                content_parts.append(df.to_csv(index=False))

            content = ''.join(content_parts)
            formatted_content = self.format_content(file_name, content)
            self.files[file_name.name] = {
                "md5": get_file_md5(file_name),
                "formatted_content": formatted_content,
                "content": content
            }
            return formatted_content if not raw_content_only else content
        except ImportError:
            logger.warning("pandas not found.")
            return None

    def clear(self):
        self.files.clear()


class RAGManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persistence=False, db_path="./chromadb"):
        if persistence:
            self.client = chromadb.PersistentClient(db_path, settings=chromadb.Settings(anonymized_telemetry=False))
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
            chunk_overlap=self.chunk_overlap
        )
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self.indexed_files = set()
        self.enabled = False

    def update_parameters(self,
                          chunk_size: int = None,
                          chunk_overlap: int = None,
                          n_results: int = None,
                          similarity_threshold: float = None):
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
                chunk_overlap=self.chunk_overlap
            )

        return updated

    def add_document(self, file_path: Path, file_content: str) -> Tuple[bool, str]:
        if file_path.name in self.indexed_files:
            return False, "Document '{}' is already indexed.".format(file_path.name)

        chunks = self.text_splitter.split_text(file_content)
        if not chunks:
            return False, "No content to index in document '{}'.".format(file_path.name)

        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)

        ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]
        metadata = [{"source": file_path.name, "chunk_id": i} for i in range(len(chunks))]

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        self.indexed_files.add(file_path.name)
        return True, "Document '{}' indexed with {} chunks.".format(file_path.name, len(chunks))

    def retrieve(self, query: str, n_results: int = None) -> str:
        if self.collection.count() == 0:
            return ""

        results_count = n_results if n_results is not None else self.n_results

        query_embeddings = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=min(results_count, self.collection.count())
        )

        if not results or not results['documents'] or not results['documents'][0]:
            return ""

        filtered_docs = []
        documents = results['documents'][0]
        distances = results.get('distances', [None])[0] if 'distances' in results else [None] * len(documents)

        for doc, distance in zip(documents, distances):
            if distance is not None and distance >= self.similarity_threshold:
                filtered_docs.append(doc)

        if not filtered_docs:
            return ""

        retrieved_docs = "\n\n".join(filtered_docs)
        return retrieved_docs

    def clear_index(self):
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self.indexed_files.clear()

    def enable(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.model_name)
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def get_status(self) -> Tuple[bool, str]:
        if not self.indexed_files:
            return False, "RAG Index is empty."
        total_chunks = self.collection.count()
        return True, "Indexed {} documents with {} chunks.".format(len(self.indexed_files), total_chunks)

    def get_parameters(self) -> Dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "n_results": self.n_results,
            "similarity_threshold": self.similarity_threshold
        }


file_manager = FileManager()
rag_manager = RAGManager()


def preprocess_file(message: Dict, history: List[Dict]) -> Tuple[str, List[Dict]]:
    processed_message_parts = []
    if "files" in message and message["files"]:
        for file_path_str in message["files"]:
            if file_path_str:
                file_content = file_manager.load_file(Path(file_path_str))
                if file_content:
                    processed_message_parts.append(file_content)

    text_content = message.get("text", "")
    if not isinstance(text_content, str):
        text_content = str(text_content) if text_content is not None else ""
    processed_message_parts.append(text_content)

    processed_message = "".join(processed_message_parts)

    preprocessed_history = []
    i = 0
    while i < len(history):
        current_hist_item_original = history[i]
        current_processed_item = copy.deepcopy(current_hist_item_original)

        current_content = current_hist_item_original.get("content")

        if isinstance(current_content, tuple):
            combined_files_text_parts = []
            for file_path_in_tuple_str in current_content:
                if file_path_in_tuple_str:
                    file_actual_content = file_manager.load_file(Path(file_path_in_tuple_str))
                    if file_actual_content:
                        combined_files_text_parts.append(file_actual_content)

            final_combined_content = "".join(combined_files_text_parts)

            text_to_merge_from_next = ""
            consumed_next_item_flag = False
            if i + 1 < len(history):
                next_original_hist_item = history[i + 1]
                next_content = next_original_hist_item.get("content")
                if isinstance(next_content, str):
                    text_to_merge_from_next = next_content
                    consumed_next_item_flag = True

            current_processed_item["content"] = final_combined_content + text_to_merge_from_next
            preprocessed_history.append(current_processed_item)

            i += 1
            if consumed_next_item_flag:
                i += 1
        else:
            if not isinstance(current_content, str):
                current_processed_item["content"] = str(current_content) if current_content is not None else ""

            preprocessed_history.append(current_processed_item)
            i += 1

    return processed_message, preprocessed_history


def encode_image(image_path: Path):
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".heif": "image/heif",
        ".heic": "image/heic"
    }
    if suffix not in mime_types:
        raise RuntimeError(f"Unsupported imgae type: {suffix}")
    with open(image_path, 'rb') as f:
        return f"data:{mime_types[suffix]};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def prepare_openai_message_content(text_input: Optional[str], file_paths: Optional[List[str]]) -> List[Dict]:
    content_parts = []
    current_text = str(text_input) if text_input is not None else ""

    if file_paths:
        for file_path_str in file_paths:
            file = Path(file_path_str)
            if not file.is_file():
                logger.warning(f"File not found: {file_path_str}, skipping.")
                continue

            suffix = file.suffix.lower()
            if suffix in [".png", ".jpg", ".jpeg", ".heif", ".heic"]:
                try:
                    image_base64 = encode_image(file)
                    content_parts.append({"type": "image_url", "image_url": {"url": image_base64}})
                except Exception as e:
                    logger.error(f"Failed to encode image {file_path_str}: {e}")
            else:
                file_content = file_manager.load_file(file)
                if file_content:
                    current_text += "\n" + file_content
                else:
                    logger.warning(f"Could not load content from file: {file_path_str}")

    if current_text or not content_parts:
        content_parts.insert(0, {"type": "text", "text": current_text.strip()})

    return content_parts


def prepare_generic_model_inputs(current_message_dict: Dict, history_list: List[Dict], system_prompt: Optional[str], model_instance: Union[TextModel, VisionModel]) -> Tuple[str, List[Dict], List[str]]:
    effective_history = []
    if system_prompt and system_prompt.strip() != "":
        effective_history.append(Message(MessageRole.SYSTEM, content=system_prompt).to_dict())
    effective_history.extend(history_list)

    effective_history = [
        history_item for history_item in effective_history
        if not (isinstance(history_item, dict) and
                isinstance(history_item.get("metadata"), dict) and
                history_item["metadata"].get("title") in ["Thinking"])
    ]

    image_paths = []
    if isinstance(model_instance, VisionModel):
        for hist_item in effective_history:
            files_to_check = []
            if isinstance(hist_item.get("content"), tuple):
                files_to_check.extend(list(hist_item["content"]))
            elif isinstance(hist_item.get("files"), list):
                files_to_check.extend(hist_item["files"])

            for file_path_str in files_to_check:
                path = Path(file_path_str)
                if path.is_file() and path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    image_paths.append(file_path_str)

        if "files" in current_message_dict and current_message_dict["files"]:
            for file_path_str in current_message_dict["files"]:
                path = Path(file_path_str)
                if path.is_file() and path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    image_paths.append(file_path_str)

        image_paths = list(set(image_paths))

    processed_message_text, processed_history_list = preprocess_file(current_message_dict, effective_history)

    return processed_message_text, processed_history_list, image_paths


from typing import Dict, List, Optional, Iterable, Iterator
from dataclasses import dataclass, field
import time
import json
import re

CHATML_CONTROL_TOKENS = [
    '<|im_start|>', '<|im_end|>',
    '<|system|>', '<|user|>', '<|assistant|>',
    '<|end|>', '<|endoftext|>'
]
CHATML_STOP_TOKENS = ['<|im_end|>', '<|end|>', '<|endoftext|>']

DEFAULT_PARTIAL_PREFIXES = [
    "<", "<t", "<th", "<thi", "<thin", "<think",
    "</", "</t", "</th", "</thi", "</thin", "</think",
    "<a", "<an", "<ans", "<answ", "<answe", "<answer",
    "</a", "</an", "</ans", "</answ", "</answe", "</answer",
    "<f", "<fu", "<fun", "<func", "<funct", "<functi", "<functio", "<function",
    "<function_", "<function_c", "<function_ca", "<function_cal", "<function_call"
]

def apply_rag_if_needed(message: Dict, rag_enabled: bool, rag_n_results: int) -> Dict:
    if rag_enabled and message.get("text"):
        original_text = message.get("text", "")
        enhanced_text = enhance_message_with_rag(original_text, rag_enabled, rag_n_results)
        new_message = dict(message)
        new_message["text"] = enhanced_text
        return new_message
    return message


def ensure_model_has_chat_template(model) -> None:
    tokenizer_to_check = None
    if isinstance(model, VisionModel):
        if getattr(model, "processor", None) and hasattr(model.processor, "tokenizer"):
            tokenizer_to_check = model.processor.tokenizer
    elif hasattr(model, "tokenizer"):
        tokenizer_to_check = model.tokenizer

    if tokenizer_to_check and getattr(tokenizer_to_check, "chat_template", None) is None:
        model_name = getattr(model, "model_name", type(model).__name__)
        raise RuntimeError(
            f"Model {model_name} does not have a chat template. "
            "Please use the 'Completion' tab or set a chat template."
        )


def normalize_sampling_params(
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float
) -> Dict[str, float]:
    return {
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "repetition_penalty": float(repetition_penalty),
    }


def get_eos_token_from_model(model) -> Optional[str]:
    if isinstance(model, VisionModel):
        if getattr(model, "processor", None) and getattr(model.processor, "tokenizer", None):
            return model.processor.tokenizer.eos_token
        return None
    return model.tokenizer.eos_token if getattr(model, "tokenizer", None) else None


def parse_chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "text"):
        return chunk.text
    if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is not None and getattr(delta, "content", None):
            return delta.content
    logger.warning(f"Unexpected chunk type from model: {type(chunk)}")
    return ""


def trim_to_eos_if_streaming(text: str, eos_token: Optional[str], stream: bool) -> (str, bool):
    if stream and eos_token and eos_token in text:
        if text == eos_token:
            return "", True
        else:
            return text.split(eos_token)[0], True
    return text, False


def filter_chatml_tokens_and_stop(
    text: str,
    control_tokens: List[str],
    stop_tokens: List[str]
) -> (str, bool):
    should_stop = False
    filtered = text

    for stop_token in stop_tokens:
        if stop_token in filtered:
            should_stop = True
            idx = filtered.find(stop_token)
            filtered = filtered[:idx]
            break

    for tok in control_tokens:
        filtered = filtered.replace(tok, "")

    return filtered, should_stop


def strip_answer_tags(text: str) -> str:
    return text.replace("<answer>", "").replace("</answer>", "")


def merge_with_partial_buffer(
    prior_buffer: str,
    text: str,
    partial_prefixes: List[str]
) -> (str, str):
    text = f"{prior_buffer}{text}" if prior_buffer else text
    new_buffer = ""
    if not text:
        return text, new_buffer

    for partial in partial_prefixes:
        if text.endswith(partial):
            new_buffer = partial
            text = text[:-len(partial)]
            break
    return text, new_buffer

@dataclass
class StreamSession:
    response_stream: Iterable
    eos_token: Optional[str]
    stream: bool
    generation_stop_event: any
    control_tokens: List[str] = field(default_factory=lambda: CHATML_CONTROL_TOKENS)
    stop_tokens: List[str] = field(default_factory=lambda: CHATML_STOP_TOKENS)
    partial_prefixes: List[str] = field(default_factory=lambda: DEFAULT_PARTIAL_PREFIXES)
    thinking_title: str = "Thinking"
    inline_thought_title: str = "Thinking"
    thinking_id: int = 0
    base_messages: List = field(default_factory=list)
    chat_message_accumulator: any = field(default_factory=lambda: ChatMessage(role="assistant", content=""))
    thinking_message: Optional[any] = None
    full_response: str = ""
    final_messages: List = field(default_factory=list)

    def __iter__(self) -> Iterator:
        final_content_parts: List[str] = []
        thinking_content_parts: List[str] = []
        in_thinking = False
        thinking_start_time: Optional[float] = None
        chunk_buffer = ""

        for chunk in self.response_stream:
            if self.generation_stop_event.is_set():
                break

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

            if should_stop:
                if in_thinking:
                    thinking_content_parts.append(chunk_text)
                    self._ensure_thinking_message(thinking_title=self.thinking_title, status="pending")
                    self.thinking_message.content = "".join(thinking_content_parts)
                    self.thinking_message.metadata["status"] = "done"
                    self.thinking_message.metadata["duration"] = time.time() - (thinking_start_time or time.time())
                    if self.stream:
                        yield self.base_messages + [self.thinking_message, self.chat_message_accumulator]
                else:
                    final_content_parts.append(chunk_text)
                    self.chat_message_accumulator.content = "".join(final_content_parts)
                    if self.stream:
                        if self.thinking_message and self.thinking_message.metadata.get("status") == "done":
                            yield self.base_messages + [self.thinking_message, self.chat_message_accumulator]
                        else:
                            yield self.base_messages + [self.chat_message_accumulator]
                break

            if "<think>" in chunk_text and not in_thinking:
                in_thinking = True
                thinking_start_time = time.time()
                self._ensure_thinking_message(thinking_title=self.thinking_title, status="pending")

                think_open_idx = chunk_text.find("<think>")
                think_start_idx = think_open_idx + len("<think>")

                before_think = chunk_text[:think_open_idx]
                if before_think:
                    final_content_parts.append(before_think)
                    self.chat_message_accumulator.content = "".join(final_content_parts)

                if think_start_idx < len(chunk_text):
                    thinking_content_parts.append(chunk_text[think_start_idx:])
                    self.thinking_message.content = "".join(thinking_content_parts)

                if self.stream:
                    yield self.base_messages + [self.thinking_message]
                continue

            if in_thinking and "</think>" not in chunk_text:
                thinking_content_parts.append(chunk_text)
                self._ensure_thinking_message(thinking_title=self.thinking_title, status="pending")
                self.thinking_message.content = "".join(thinking_content_parts)
                if self.stream:
                    yield self.base_messages + [self.thinking_message]
                continue

            if in_thinking and "</think>" in chunk_text:
                think_end_idx = chunk_text.find("</think>")
                thinking_content_parts.append(chunk_text[:think_end_idx])
                self._ensure_thinking_message(thinking_title=self.thinking_title, status="pending")
                self.thinking_message.content = "".join(thinking_content_parts)
                self.thinking_message.metadata["status"] = "done"
                self.thinking_message.metadata["duration"] = time.time() - (thinking_start_time or time.time())

                remaining = chunk_text[think_end_idx + len("</think>"):]
                if remaining.strip():
                    final_content_parts.append(remaining)
                    self.chat_message_accumulator.content = "".join(final_content_parts)

                if self.stream:
                    payload = self.base_messages + [self.thinking_message]
                    if self.chat_message_accumulator.content:
                        payload += [self.chat_message_accumulator]
                    yield payload
                in_thinking = False
                continue

            if not in_thinking:
                if "<think>" in chunk_text and "</think>" in chunk_text:
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
                                "duration": 0.1
                            }
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
                                payload += [self.chat_message_accumulator]
                            yield payload
                        continue

                final_content_parts.append(chunk_text)
                self.chat_message_accumulator.content = "".join(final_content_parts)
                if self.stream:
                    if self.thinking_message and self.thinking_message.metadata.get("status") == "done":
                        yield self.base_messages + [self.thinking_message, self.chat_message_accumulator]
                    else:
                        yield self.base_messages + [self.chat_message_accumulator]

            if self.stream and eos_hit:
                break

        if self.thinking_message and self.thinking_message.metadata.get("status") == "done":
            self.final_messages = [self.thinking_message]
            if self.chat_message_accumulator.content:
                self.final_messages.append(self.chat_message_accumulator)
        else:
            self.final_messages = [self.chat_message_accumulator] if self.chat_message_accumulator.content else []

    def _ensure_thinking_message(self, thinking_title: str, status: str):
        if not self.thinking_message:
            self.thinking_message = ChatMessage(
                role="assistant",
                content="",
                metadata={"title": thinking_title, "id": self.thinking_id, "status": status}
            )

def handle_chat(message: Dict,
                history: List[Dict],
                system_prompt: str = None,
                temperature: float = 0.7,
                top_k: int = 20,
                top_p: float = 0.9,
                min_p: float = 0.0,
                max_tokens: int = 512,
                repetition_penalty: float = 1.0,
                rag_enabled: bool = False,
                rag_n_results: int = 5,
                function_calling_enabled: bool = False,
                stream: bool = True):
    try:
        message = apply_rag_if_needed(message, rag_enabled, rag_n_results)

        model = get_loaded_model()
        ensure_model_has_chat_template(model)

        processed_message_text, processed_history_list, image_paths = prepare_generic_model_inputs(
            message, history, system_prompt, model
        )

        if function_calling_enabled and function_manager.is_enabled():
            processed_message_text = function_manager.format_function_call_for_local_model(processed_message_text)

        sampling = normalize_sampling_params(temperature, top_k, top_p, min_p, repetition_penalty)

        response_args = {
            "message": processed_message_text,
            "history": processed_history_list,
            "stream": stream,
            "max_tokens": max_tokens,
            **sampling
        }
        if isinstance(model, VisionModel):
            response_args["images"] = image_paths
        eos_token = get_eos_token_from_model(model)

        response_stream = model.generate_response(**response_args)
        session = StreamSession(
            response_stream=response_stream,
            eos_token=eos_token,
            stream=stream,
            generation_stop_event=generation_stop_event,
            thinking_title="Thinking",
            inline_thought_title="Thinking",
            thinking_id=0,
            base_messages=[]
        )
        for payload in session:
            yield payload

        final_messages = session.final_messages
        full_response = session.full_response

        if function_calling_enabled and function_manager.is_enabled():
            function_call = function_manager.parse_function_call_from_response(full_response)
            if function_call:
                function_name, function_args = function_call
                result = function_manager.execute_function(function_name, function_args)

                function_message = ChatMessage(
                    role="assistant",
                    content=(
                        f"**Function Call:** `{function_name}`\n"
                        f"**Arguments:** `{json.dumps(function_args)}`\n"
                        f"**Result:** {json.dumps(result)}"
                    ),
                    metadata={"title": "Function Call", "id": 1, "status": "done"}
                )

                current_messages = list(final_messages) + [function_message]
                if stream:
                    yield current_messages

                if "result" in result:
                    assistant_full_response = ""
                    if session.thinking_message and session.thinking_message.content:
                        assistant_full_response = "<think>{}</think>\n{}".format(
                            session.thinking_message.content,
                            session.chat_message_accumulator.content or f"I'll call the {function_name} function."
                        )
                    else:
                        assistant_full_response = session.chat_message_accumulator.content or f"I'll call the {function_name} function."

                    enhanced_history = processed_history_list + [
                        {"role": "assistant", "content": assistant_full_response},
                        {"role": "tool", "content": json.dumps(result["result"]), "name": function_name}
                    ]

                    follow_up_prompt = (
                        f"Based on the function '{function_name}' result: "
                        f"{json.dumps(result['result'])}, please provide a helpful response to the user's original question."
                    )

                    follow_up_args = {
                        "message": follow_up_prompt,
                        "history": enhanced_history,
                        "stream": stream,
                        "max_tokens": max_tokens,
                        **sampling
                    }
                    if isinstance(model, VisionModel) and image_paths:
                        follow_up_args["images"] = image_paths

                    follow_up_response_stream = model.generate_response(**follow_up_args)

                    follow_up_session = StreamSession(
                        response_stream=follow_up_response_stream,
                        eos_token=eos_token,
                        stream=stream,
                        generation_stop_event=generation_stop_event,
                        thinking_title="Follow-up Thinking",
                        inline_thought_title="Follow-up Thought",
                        thinking_id=2,
                        base_messages=list(final_messages) + [function_message]
                    )

                    for payload in follow_up_session:
                        yield payload

                    if not stream:
                        all_messages = follow_up_session.base_messages + follow_up_session.final_messages
                        yield all_messages
    except Exception as e:
        logger.exception("Error in handle_chat:")
        raise gr.Error(str(e))


def managed_chat_generator(
        message: Dict,
        history: List[Dict],
        system_prompt: str = None,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        rag_enabled: bool = False,
        rag_n_results: int = 5,
        function_calling_enabled: bool = False,
        stream: bool = True):
    g = handle_chat(
        message=message,
        history=history,
        system_prompt=system_prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        rag_enabled=rag_enabled,
        rag_n_results=rag_n_results,
        function_calling_enabled=function_calling_enabled,
        stream=stream,
    )

    generation_stop_event.clear()
    model_manager.close_active_generator()
    model_manager.set_active_generator(g)

    try:
        yield from g
    finally:
        model_manager.close_active_generator()


def handle_completion(prompt: str,
                      temperature: float = 0.7,
                      top_k: int = 20,
                      top_p: float = 0.9,
                      min_p: float = 0.0,
                      max_tokens: int = 512,
                      repetition_penalty: float = 1.0,
                      stream: bool = True):
    try:
        model = get_loaded_model()
        if isinstance(model, VisionModel):
            raise RuntimeError("Not supported yet.")

        temperature = float(temperature)
        top_p = float(top_p)
        repetition_penalty = float(repetition_penalty)

        if not stream:
            completion_text = model.generate_completion(
                prompt=prompt,
                stream=stream,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty)
            return ''.join([prompt, completion_text])
        else:
            response_parts = [prompt]
            eos_token = model.tokenizer.eos_token

            for chunk in model.generate_completion(
                    prompt=prompt,
                    stream=stream,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty):
                if generation_stop_event.is_set():
                    break

                chunk_text = ""
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, "text"):
                    chunk_text = chunk.text
                elif hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                else:
                    logger.warning(f"Unexpected chunk type from model: {type(chunk)}")

                if chunk_text:
                    if eos_token not in chunk_text:
                        response_parts.append(chunk_text)
                        yield ''.join(response_parts)
                    else:
                        if eos_token in chunk_text:
                            before_eos = chunk_text.split(eos_token)[0]
                            if before_eos:
                                response_parts.append(before_eos)
                        yield ''.join(response_parts)
                        break
            return None
    except Exception as e:
        raise gr.Error(str(e))


def managed_completion_generator(prompt: str,
                                 temperature: float = 0.7,
                                 top_k: int = 20,
                                 top_p: float = 0.9,
                                 min_p: float = 0.0,
                                 max_tokens: int = 512,
                                 repetition_penalty: float = 1.0,
                                 stream: bool = True):
    g = handle_completion(
        prompt=prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        stream=stream
    )

    generation_stop_event.clear()
    model_manager.close_active_generator()
    model_manager.set_active_generator(g)

    try:
        yield from g
    finally:
        model_manager.close_active_generator()


def get_load_model_status():
    if model_manager.get_loaded_model_config():
        return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.loaded_value").format(model_manager.get_loaded_model_config().get("display_name"))
    else:
        return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")


def load_model(model_name: str) -> Tuple[str, str]:
    try:
        model_manager.load_model(model_name)
        return get_load_model_status(), model_manager.get_system_prompt(default=True)
    except Exception as e:
        raise gr.Error(str(e))


def stop_generation() -> None:
    generation_stop_event.set()


def chat_load_model_callback(model_name: str):
    stop_generation()
    return load_model(model_name)


def completion_load_model_callback(model_name: str):
    stop_generation()
    return load_model(model_name)[0]


def get_default_system_prompt_callback():
    if model_manager.get_loaded_model_config():
        return model_manager.get_system_prompt(default=True)
    else:
        raise gr.Error("No model loaded.")



def search_huggingface_models(query: str) -> DataFrame:
    if not query:
        return DataFrame(columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))

    api = HfApi()
    try:
        models = api.list_models(search=query, filter="mlx", sort="likes", direction=-1, limit=50)
        data = []
        for model in models:
            data.append([model.modelId, model.likes, model.downloads])
        return DataFrame(data, columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))
    except Exception as e:
        logger.error(f"Error searching HuggingFace: {e}")
        return DataFrame(columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))


def auto_fill_model_info(evt: gr.SelectData, df: DataFrame):
    if evt.index[0] < 0 or evt.index[0] >= len(df):
        return gr.update(), gr.update(), gr.update()

    model_id = df.iloc[evt.index[0]]["Model ID"]
    model_name = model_id.split("/")[-1]

    quantize = "None"
    lower_name = model_name.lower()
    if "4bit" in lower_name:
        quantize = "4bit"
    elif "8bit" in lower_name:
        quantize = "8bit"
    elif "q4" in lower_name:
        quantize = "4bit"
    elif "q8" in lower_name:
        quantize = "8bit"

    return model_name, model_id, quantize


def update_model_management_models_list():
    return DataFrame({get_text("Page.ModelManagement.Dataframe.model_list.headers"): model_manager.get_model_list()})


def update_select_model_dropdown_value():
    if model_manager.get_loaded_model_config():
        return model_manager.get_loaded_model_config().get("display_name")
    else:
        return model_manager.get_model_list()[0] if len(model_manager.get_model_list()) > 0 else None


def update_model_selector_choices():
    return gr.update(choices=model_manager.get_model_list(), value=update_select_model_dropdown_value())


def update_delete_model_selector_choices():
    return gr.update(choices=model_manager.get_model_list())


def add_model(model_name: Optional[str], mlx_repo: str, quantize: str, default_language: str, default_system_prompt: Optional[str], multimodal_ability: List[str]):
    try:
        model_manager.add_config(mlx_repo, model_name, quantize, default_language, default_system_prompt, multimodal_ability)
    except Exception as e:
        raise gr.Error(str(e))


def delete_model(model_name: str, delete_files: bool) -> str:
    try:
        if not model_name:
            raise gr.Error(get_text("Page.ModelManagement.DeleteModelBlock.Messages.no_model_selected"))
        
        model_manager.delete_config(model_name, delete_model_files=delete_files)
        
        if delete_files:
            return get_text("Page.ModelManagement.DeleteModelBlock.Messages.config_and_files_deleted").format(model_name)
        else:
            return get_text("Page.ModelManagement.DeleteModelBlock.Messages.config_deleted").format(model_name)
    except Exception as e:
        raise gr.Error(str(e))


def update_slider_config(slider_new_min: Union[int, float], slider_new_max: Union[int, float], slider_value: Union[int, float]):
    if not slider_new_min or not slider_new_max:
        return gr.update()

    if slider_new_min > slider_new_max:
        return gr.update()

    value_to_set = slider_value if slider_value is not None else (slider_new_min + slider_new_max) / 2

    if isinstance(slider_new_min, int) and isinstance(slider_new_max, int):
        value_to_set = int(value_to_set)

    if value_to_set < slider_new_min:
        value_to_set = slider_new_min

    if value_to_set > slider_new_max:
        value_to_set = slider_new_max

    return gr.update(minimum=slider_new_min, maximum=slider_new_max, value=value_to_set)


def update_model_max_length(slider_value: Union[int, float]):
    try:
        model = get_loaded_model()
        if model is not None:
            try:
                model_max_length = 32768
                if isinstance(model, TextModel) or isinstance(model, VisionModel):
                    model_max_length = model.max_position_embeddings
                if model_max_length is None or model_max_length <= 0:
                    model_max_length = 32768
            except Exception as e:
                model_max_length = 32768
                logger.error("Error while updating model max length.")
            return update_slider_config(1, model_max_length, slider_value)
        else:
            return gr.update()
    except RuntimeError:
        return gr.update()
    except Exception as e:
        raise gr.Error(str(e))


def bytes_to_gigabytes(value):
    return value / 1024 ** 3


def get_memory_usage() -> str:
    memory_usage_bytes = model_manager.get_system_memory_usage()
    total_memory_bytes = model_manager.get_device_info()["memory_size"]

    memory_usage_gb = bytes_to_gigabytes(memory_usage_bytes) if isinstance(memory_usage_bytes, (int, float)) else "N/A"
    total_memory_gb = bytes_to_gigabytes(total_memory_bytes) if isinstance(total_memory_bytes, (int, float)) else "N/A"

    return "{:.2f} GB | {:.2f} GB".format(memory_usage_gb, total_memory_gb)


def update_all_memory_usage() -> list[str]:
    memory_usage = get_memory_usage()
    return [memory_usage, memory_usage]


def get_rag_enabled_status() -> bool:
    return rag_manager.is_enabled()


def get_rag_status() -> str:
    return rag_manager.get_status()[1]


def get_function_calling_enabled_status() -> bool:
    return function_manager.is_enabled()


def get_function_status() -> str:
    if not function_manager.functions:
        return "No functions configured."
    enabled_count = len(function_manager.get_enabled_functions())
    total_count = len(function_manager.functions)
    mode = function_manager.get_execution_mode()
    mode_label = "Simulate" if mode == "simulate" else "Execute"
    return f"{enabled_count}/{total_count} functions enabled. Mode: {mode_label}."


def get_functions_df() -> DataFrame:
    return DataFrame(function_manager.get_functions_list(), columns=["Name", "Description", "Enabled"])


def build_sample_args_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = (schema or {}).get("properties", {})
    required = (schema or {}).get("required", [])

    def sample_for(t: str, node: Dict[str, Any]) -> Any:
        if t == "string":
            return ""
        if t in ("integer", "number"):
            return 0
        if t == "boolean":
            return False
        if t == "array":
            item = (node or {}).get("items", {})
            itype = item.get("type", "string")
            return [sample_for(etype, item)] if (etype := itype) else []
        if t == "object":
            return {}
        return None

    out = {}
    for k in required:
        node = props.get(k, {})
        t = node.get("type", "string")
        out[k] = sample_for(t, node)
    return out


def get_function_schema_and_sample(name: str) -> Tuple[str, str]:
    func = function_manager.get_function(name)
    if not func:
        return "", "{}"
    schema_pretty = json.dumps(func.parameters or {}, indent=2, ensure_ascii=False)
    sample = build_sample_args_from_schema(func.parameters or {})
    return schema_pretty, json.dumps(sample, indent=2, ensure_ascii=False)


def execute_function_test(name: str, args_json: str) -> Tuple[str, str, DataFrame]:
    result = function_manager.execute_function(name, args_json)
    if "result" in result:
        status = f"Executed '{name}' successfully."
        pretty = json.dumps(result["result"], indent=2, ensure_ascii=False)
    else:
        status = f"Error executing '{name}': {result.get('error')}"
        pretty = json.dumps(result, indent=2, ensure_ascii=False)
    hist_df = DataFrame(
        function_manager.get_execution_history_rows(limit=50),
        columns=["Time", "Function", "Arguments", "Result", "Error"]
    )
    return pretty, status, hist_df


def clear_function_history() -> Tuple[DataFrame, str]:
    function_manager.clear_history()
    hist_df = DataFrame(
        function_manager.get_execution_history_rows(limit=50),
        columns=["Time", "Function", "Arguments", "Result", "Error"]
    )
    return hist_df, "Function execution history cleared."


def apply_function_list_enabled_from_df(df: Union[DataFrame, List[List[Any]]]) -> Tuple[str, str, DataFrame]:
    try:
        rows = df.values.tolist() if isinstance(df, DataFrame) else df
        changed = 0
        for name, _, enabled in rows:
            if isinstance(enabled, str):
                enabled_bool = enabled.strip().lower() in ("true", "1", "✓", "yes")
            else:
                enabled_bool = bool(enabled)
            if function_manager.get_function(name):
                changed += 1 if function_manager.toggle_function(name, enabled_bool) else 0
        msg = f"Synchronized {len(rows)} functions."
    except Exception as e:
        msg = f"Sync error: {e}"

    return msg, get_function_status(), get_functions_df()


def enable_all_functions() -> Tuple[str, DataFrame, str]:
    changed = function_manager.toggle_all(True)
    return f"Enabled {changed} function(s).", get_functions_df(), get_function_status()


def disable_all_functions() -> Tuple[str, DataFrame, str]:
    changed = function_manager.toggle_all(False)
    return f"Disabled {changed} function(s).", get_functions_df(), get_function_status()


def set_function_execution_mode(mode_label: str) -> str:
    mode = "simulate" if (mode_label or "").lower().startswith("sim") else "execute"
    function_manager.set_execution_mode(mode)
    return get_function_status()


def get_openai_tools_preview() -> str:
    return json.dumps(function_manager.get_openai_tools(), indent=2, ensure_ascii=False)


def export_custom_functions_json() -> str:
    return json.dumps(function_manager.export_custom_functions(), indent=2, ensure_ascii=False)


def import_custom_functions_from_json(json_text: str) -> Tuple[str, DataFrame, str, gr.update]:
    ok, message = function_manager.import_functions(json_text or "[]")
    return (
        message,
        get_functions_df(),
        get_function_status(),
        gr.update(choices=function_manager.get_function_names())
    )


def update_test_function_choices() -> gr.update:
    return gr.update(choices=function_manager.get_function_names())


def get_initial_history_df() -> DataFrame:
    return DataFrame(
        function_manager.get_execution_history_rows(limit=50),
        columns=["Time", "Function", "Arguments", "Result", "Error"]
    )


def upload_and_index_file(files) -> Tuple[str, str]:
    if not files:
        return "No files selected.", get_rag_status()

    results = []
    for file_path in files:
        try:
            file_content = file_manager.load_file(Path(file_path), raw_content_only=True)
            if file_content:
                success, message = rag_manager.add_document(Path(file_path), file_content)
                results.append(f"{Path(file_path).name}: {message}")
            else:
                results.append(f"{Path(file_path).name}: Failed to load content")
        except Exception as e:
            results.append(f"{Path(file_path).name}: Error - {str(e)}")

    return "\n".join(results), get_rag_status()


def clear_rag_index() -> Tuple[str, str]:
    try:
        rag_manager.clear_index()
        return "RAG index cleared successfully.", get_rag_status()
    except Exception as e:
        return f"Error clearing RAG index: {str(e)}", get_rag_status()


def toggle_rag_enabled(enabled: bool) -> str:
    if enabled:
        rag_manager.enable()
        return "RAG enabled."
    else:
        rag_manager.disable()
        return "RAG disabled."


def toggle_function_calling_enabled(enabled: bool) -> str:
    if enabled:
        function_manager.enable()
        return "Function calling enabled."
    else:
        function_manager.disable()
        return "Function calling disabled."


def update_rag_parameters(chunk_size: int, chunk_overlap: int, similarity_threshold: float) -> str:
    try:
        updated = rag_manager.update_parameters(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold
        )

        if updated:
            return "RAG parameters updated. Consider re-indexing documents for optimal performance."
        else:
            return "RAG parameters updated."
    except Exception as e:
        return f"Error updating RAG parameters: {str(e)}"


def get_rag_parameters() -> Tuple[int, int, int, float]:
    params = rag_manager.get_parameters()
    return (
        params["chunk_size"],
        params["chunk_overlap"],
        params["n_results"],
        params["similarity_threshold"]
    )


def add_custom_function(name: str, description: str, parameters_json: str) -> Tuple[str, DataFrame]:
    success, message = function_manager.add_custom_function(name, description, parameters_json)
    return message, get_functions_df()


def toggle_function(name: str, enabled: bool) -> Tuple[str, DataFrame]:
    if function_manager.toggle_function(name, enabled):
        status = "enabled" if enabled else "disabled"
        message = f"Function '{name}' {status}."
    else:
        message = f"Function '{name}' not found."
    return message, get_functions_df()


def remove_function(name: str) -> Tuple[str, DataFrame]:
    if function_manager.remove_function(name):
        message = f"Function '{name}' removed."
    else:
        message = f"Function '{name}' not found."
    return message, get_functions_df()


def enhance_message_with_rag(message_text: str, rag_enabled: bool, n_results: int = 5) -> str:
    if not rag_enabled:
        return message_text

    try:
        retrieved_docs = rag_manager.retrieve(message_text, n_results=n_results)

        if retrieved_docs:
            enhanced_message = """
            Based on the following relevant documents: {}

            ---

            User question: {}
            """.format(retrieved_docs, message_text)
            return enhanced_message
        else:
            return message_text
    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
        return message_text


def create_slider(min_val, max_val, default_val, label_key, **kwargs):
    return gr.Slider(
        minimum=min_val,
        maximum=max_val,
        value=default_val,
        label=get_text(label_key),
        render=False,
        interactive=True,
        **kwargs
    )


def create_textbox(label_key, placeholder_key=None, **kwargs):
    params = {
        'label': get_text(label_key),
        'render': False,
        'interactive': True,
        **kwargs
    }
    if placeholder_key:
        params['placeholder'] = get_text(placeholder_key)
    return gr.Textbox(**params)


def create_model_controls():
    memory_usage = gr.Textbox(
        label=get_text("Page.Chat.SystemStatusBlock.Textbox.memory_usage.label"),
        interactive=False,
        render=False
    )

    model_selector = gr.Dropdown(
        label=get_text("Page.Chat.LoadModelBlock.Dropdown.model_selector.label"),
        choices=model_manager.get_model_list(),
        render=False,
        interactive=True
    )

    model_status = gr.Textbox(
        value=get_load_model_status,
        show_label=False,
        render=False,
        interactive=False
    )

    load_button = gr.Button(
        value=get_text("Page.Chat.LoadModelBlock.Button.load_model.value"),
        render=False,
        interactive=True
    )

    return memory_usage, model_selector, model_status, load_button


def create_generation_params():
    slider_configs = {
        'temperature': (0.0, 2.0, 0.6),
        'top_k': (0, 100, 20),
        'top_p': (0.0, 1.0, 0.95),
        'min_p': (0.0, 1.0, 0.0),
        'max_tokens': (1, 32768, 4096),
        'repetition_penalty': (0.0, 2.0, 1.0)
    }

    sliders = {}
    for param, (min_val, max_val, default) in slider_configs.items():
        sliders[param] = create_slider(
            min_val, max_val, default,
            f"Page.Chat.Accordion.AdvancedSetting.Slider.{param}.label"
        )

    return sliders


def create_rag_params():
    chunk_size, chunk_overlap, n_results, similarity_threshold = get_rag_parameters()

    rag_params = {
        'chunk_size': gr.Slider(
            minimum=100,
            maximum=2000,
            value=chunk_size,
            step=50,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.chunk_size.label"),
            render=False,
            interactive=True
        ),
        'chunk_overlap': gr.Slider(
            minimum=0,
            maximum=500,
            value=chunk_overlap,
            step=10,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.chunk_overlap.label"),
            render=False,
            interactive=True
        ),
        'n_results': gr.Slider(
            minimum=1,
            maximum=10,
            value=n_results,
            step=1,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.n_results.label"),
            render=False,
            interactive=True
        ),
        'similarity_threshold': gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=similarity_threshold,
            step=0.05,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.similarity_threshold.label"),
            render=False,
            interactive=True
        )
    }

    return rag_params


def setup_model_sync_events(chat_selector, completion_selector, chat_load_btn, completion_load_btn,
                            chat_status, completion_status, chat_system_prompt, chat_max_tokens, completion_max_tokens):
    chat_selector.select(
        fn=lambda x: x,
        inputs=[chat_selector],
        outputs=[completion_selector]
    )

    completion_selector.select(
        fn=lambda x: x,
        inputs=[completion_selector],
        outputs=[chat_selector]
    )

    chat_load_btn.click(
        fn=chat_load_model_callback,
        inputs=[chat_selector],
        outputs=[chat_status, chat_system_prompt]
    ).then(
        fn=lambda x: x,
        inputs=[chat_status],
        outputs=[completion_status]
    ).then(
        fn=update_model_max_length,
        inputs=[chat_max_tokens],
        outputs=[chat_max_tokens]
    ).then(
        fn=update_model_max_length,
        inputs=[completion_max_tokens],
        outputs=[completion_max_tokens]
    )

    completion_load_btn.click(
        fn=completion_load_model_callback,
        inputs=[completion_selector],
        outputs=[completion_status]
    ).then(
        fn=lambda x: x,
        inputs=[completion_status],
        outputs=[chat_status]
    ).then(
        fn=update_model_max_length,
        inputs=[chat_max_tokens],
        outputs=[chat_max_tokens]
    ).then(
        fn=update_model_max_length,
        inputs=[completion_max_tokens],
        outputs=[completion_max_tokens]
    )


def setup_model_management_events(local_form, model_list, chat_selector, completion_selector):
    local_form['search_button'].click(
        fn=search_huggingface_models,
        inputs=[local_form['search_query']],
        outputs=[local_form['search_results']]
    )

    local_form['search_results'].select(
        fn=auto_fill_model_info,
        inputs=[local_form['search_results']],
        outputs=[
            local_form['model_name'],
            local_form['mlx_repo'],
            local_form['quantize']
        ]
    )

    local_form['add_button'].click(
        fn=add_model,
        inputs=[
            local_form['model_name'],
            local_form['mlx_repo'],
            local_form['quantize'],
            local_form['default_language'],
            local_form['system_prompt'],
            local_form['multimodal']
        ]
    ).then(
        fn=update_model_management_models_list,
        outputs=[model_list]
    ).then(
        fn=update_model_selector_choices,
        outputs=[chat_selector]
    ).then(
        fn=update_model_selector_choices,
        outputs=[completion_selector]
    ).then(
        fn=update_delete_model_selector_choices,
        outputs=[local_form['delete_model_selector']]
    )

    local_form['delete_button'].click(
        fn=delete_model,
        inputs=[
            local_form['delete_model_selector'],
            local_form['delete_files_checkbox']
        ],
        outputs=[local_form['delete_status']]
    ).then(
        fn=update_model_management_models_list,
        outputs=[model_list]
    ).then(
        fn=update_model_selector_choices,
        outputs=[chat_selector]
    ).then(
        fn=update_model_selector_choices,
        outputs=[completion_selector]
    ).then(
        fn=update_delete_model_selector_choices,
        outputs=[local_form['delete_model_selector']]
    )


def clear_cache():
    stop_generation()
    file_manager.clear()
    model_manager.close_active_generator()


with gr.Blocks(fill_height=True, fill_width=True, title="Chat with MLX") as app:
    update_memory_usage_timer = gr.Timer(value=1, active=True)

    chat_memory, chat_model_selector, chat_model_status, chat_load_button = create_model_controls()
    chat_params = create_generation_params()
    rag_params = create_rag_params()

    chat_system_prompt_textbox = gr.Textbox(
        label=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.label"),
        placeholder=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.placeholder"),
        value=model_manager.get_system_prompt,
        lines=3,
        max_lines=5,
        show_copy_button=True,
        render=False,
        scale=9
    )

    chat_default_system_prompt_button = gr.Button(
        value=get_text("Page.Chat.ChatSystemPromptBlock.Button.default_system_prompt.value"),
        render=False,
        scale=1
    )

    chat_rag_form = {
        'rag_enabled': gr.Checkbox(
            label=get_text("Page.Chat.Accordion.RAGSetting.Checkbox.rag_enabled.label"),
            value=get_rag_enabled_status,
            interactive=True,
            render=False
        ),
        'file_upload': gr.File(
            label=get_text("Page.Chat.Accordion.RAGSetting.File.file_upload.label"),
            file_count="multiple",
            file_types=[".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".md", ".csv"],
            render=False
        ),
        'upload_button': gr.Button(
            value=get_text("Page.Chat.Accordion.RAGSetting.Button.upload.value"),
            render=False
        ),
        'clear_button': gr.Button(
            value=get_text("Page.Chat.Accordion.RAGSetting.Button.clear.value"),
            render=False
        ),
        'upload_status': create_textbox("Page.Chat.Accordion.RAGSetting.Textbox.upload_status.label", interactive=False, render=False),
        'rag_status': create_textbox("Page.Chat.Accordion.RAGSetting.Textbox.rag_status.label", interactive=False, render=False),
        'update_params_button': gr.Button(
            value=get_text("Page.Chat.Accordion.RAGSetting.Button.update_params.value"),
            render=False
        ),
        'params_status': create_textbox("Page.Chat.Accordion.RAGSetting.Textbox.params_status.label", interactive=False, render=False)
    }

    function_form = {
        'function_calling_enabled': gr.Checkbox(
            label="Enable Function Calling",
            value=get_function_calling_enabled_status,
            interactive=True,
            render=False
        ),
        'execution_mode': gr.Radio(
            label="Execution Mode",
            choices=["Execute", "Simulate"],
            value=lambda: "Simulate" if function_manager.get_execution_mode() == "simulate" else "Execute",
            interactive=True,
            render=False
        ),
        'function_status': gr.Textbox(
            label="Function Status",
            value=get_function_status,
            interactive=False,
            render=False
        ),
        'enable_all_button': gr.Button(
            value="Enable All",
            render=False
        ),
        'disable_all_button': gr.Button(
            value="Disable All",
            render=False
        ),
        'function_name': gr.Textbox(
            label="Function Name",
            placeholder="e.g., get_weather",
            render=False
        ),
        'function_description': gr.Textbox(
            label="Function Description",
            placeholder="Describe what the function does",
            render=False
        ),
        'function_parameters': gr.Code(
            label="Function Parameters (JSON Schema)",
            language="json",
            value='{\n  "type": "object",\n  "properties": {\n    "param1": {\n      "type": "string",\n      "description": "Description of param1"\n    }\n  },\n  "required": ["param1"]\n}',
            render=False
        ),
        'add_function_button': gr.Button(
            value="Add Function",
            render=False
        ),
        'function_list': gr.Dataframe(
            headers=["Name", "Description", "Enabled"],
            value=get_functions_df(),
            datatype=["str", "str", "bool"],
            row_count=(5, "dynamic"),
            render=False,
            interactive=True
        ),
        'toggle_function_name': gr.Textbox(
            label="Function Name to Toggle",
            render=False
        ),
        'toggle_function_enabled': gr.Checkbox(
            label="Enable",
            value=True,
            render=False
        ),
        'toggle_function_button': gr.Button(
            value="Toggle Function",
            render=False
        ),
        'remove_function_name': gr.Textbox(
            label="Function Name to Remove",
            render=False
        ),
        'remove_function_button': gr.Button(
            value="Remove Function",
            render=False
        ),
        'function_operation_status': gr.Textbox(
            label="Operation Status",
            interactive=False,
            render=False
        ),
        'test_function_select': gr.Dropdown(
            label="Select Function to Test",
            choices=function_manager.get_function_names(),
            value=None,
            render=False,
            interactive=True
        ),
        'test_parameters_preview': gr.Code(
            label="Parameters Schema (read-only)",
            language="json",
            value="",
            render=False
        ),
        'test_arguments': gr.Code(
            label="Arguments (JSON)",
            language="json",
            value="{}",
            render=False
        ),
        'execute_test_button': gr.Button(
            value="Execute Test",
            render=False
        ),
        'test_result': gr.Code(
            label="Result",
            language="json",
            value="",
            render=False
        ),
        'history_table': gr.Dataframe(
            headers=["Time", "Function", "Arguments", "Result", "Error"],
            value=get_initial_history_df(),
            datatype=["str", "str", "str", "str", "str"],
            row_count=(5, "dynamic"),
            render=False,
            interactive=False
        ),
        'clear_history_button': gr.Button(
            value="Clear History",
            render=False
        ),
        'export_button': gr.Button(
            value="Export Custom Functions JSON",
            render=False
        ),
        'export_json': gr.Code(
            label="Exported JSON",
            language="json",
            value="",
            render=False
        ),
        'import_json': gr.Code(
            label="Import JSON",
            language="json",
            value="[]",
            render=False
        ),
        'import_button': gr.Button(
            value="Import Functions",
            render=False
        ),
        'tools_preview_button': gr.Button(
            value="Preview OpenAI Tools JSON",
            render=False
        ),
        'tools_preview': gr.Code(
            label="OpenAI Tools (Preview)",
            language="json",
            value="",
            render=False
        )
    }

    completion_memory, completion_model_selector, completion_model_status, completion_load_button = create_model_controls()
    completion_params = create_generation_params()

    local_model_form = {
        'search_query': gr.Textbox(label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.search_query.label"), placeholder=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.search_query.placeholder"), render=False),
        'search_button': gr.Button(value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.search.value"), render=False),
        'search_results': gr.Dataframe(headers=get_text("Page.ModelManagement.Dataframe.search_results.headers"), datatype=["str", "number", "number"], interactive=False, render=False),
        'model_name': create_textbox("Page.ModelManagement.AddLocalModelBlock.Textbox.model_name.label",
                                     "Page.ModelManagement.AddLocalModelBlock.Textbox.model_name.placeholder"),
        'mlx_repo': create_textbox("Page.ModelManagement.AddLocalModelBlock.Textbox.mlx_repo.label",
                                   "Page.ModelManagement.AddLocalModelBlock.Textbox.mlx_repo.placeholder"),
        'quantize': gr.Dropdown(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.quantize.label"),
            choices=["None", "2bit", "3bit", "4bit", "6bit", "8bit", "bf16", "bf32"],
            value="None",
            interactive=True,
            render=False
        ),
        'default_language': gr.Dropdown(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.default_language.label"),
            choices=["multi"],
            interactive=True,
            render=False
        ),
        'system_prompt': create_textbox("Page.ModelManagement.AddLocalModelBlock.Textbox.default_system_prompt.label"),
        'multimodal': gr.Dropdown(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.multimodal_ability.label"),
            choices=["None", "vision"],
            value="None",
            multiselect=True,
            render=False
        ),
        'add_button': gr.Button(
            value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.add.value"),
            render=False
        ),
        'delete_model_selector': gr.Dropdown(
            label=get_text("Page.ModelManagement.DeleteModelBlock.Dropdown.model_selector.label"),
            choices=model_manager.get_model_list(),
            value=None,
            interactive=True,
            render=False
        ),
        'delete_files_checkbox': gr.Checkbox(
            label=get_text("Page.ModelManagement.DeleteModelBlock.Checkbox.delete_files.label"),
            value=False,
            interactive=True,
            render=False
        ),
        'delete_button': gr.Button(
            value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.delete.value"),
            variant="stop",
            render=False
        ),
        'delete_status': gr.Textbox(
            label=get_text("Page.ModelManagement.DeleteModelBlock.Textbox.delete_status.label"),
            interactive=False,
            render=False
        )
    }

    model_list = gr.Dataframe(
        headers=[get_text("Page.ModelManagement.Dataframe.model_list.headers")],
        value=update_model_management_models_list(),
        datatype="str",
        row_count=(10, "dynamic"),
        render=False,
        interactive=False
    )

    update_memory_usage_timer.tick(
        fn=update_all_memory_usage,
        outputs=[chat_memory, completion_memory]
    )

    gr.HTML("<h2>Chat with MLX</h2>")

    with gr.Tab(get_text("Tab.chat")):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    chat_memory.render()

                with gr.Row():
                    gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                    chat_model_selector.render()
                    chat_model_status.render()
                    chat_load_button.render()

                with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=False):
                    with gr.Group():
                        for slider in chat_params.values():
                            slider.render()

                with gr.Accordion(label=get_text("Page.Chat.Accordion.RAGSetting.label"), open=False):
                    chat_rag_form['rag_enabled'].render()
                    chat_rag_form['rag_status'].render()

                    with gr.Row():
                        chat_rag_form['file_upload'].render()

                    with gr.Row():
                        chat_rag_form['upload_button'].render()
                        chat_rag_form['clear_button'].render()

                    chat_rag_form['upload_status'].render()

                    with gr.Group():
                        with gr.Row():
                            rag_params['chunk_size'].render()
                            rag_params['chunk_overlap'].render()

                        with gr.Row():
                            rag_params['n_results'].render()
                            rag_params['similarity_threshold'].render()

                    chat_rag_form['update_params_button'].render()
                    chat_rag_form['params_status'].render()

                with gr.Accordion(label="Function Calling", open=False):
                    with gr.Row():
                        function_form['function_calling_enabled'].render()
                        function_form['execution_mode'].render()

                    function_form['function_status'].render()

                    with gr.Row():
                        function_form['enable_all_button'].render()
                        function_form['disable_all_button'].render()

                    gr.Markdown("### Available Functions")
                    function_form['function_list'].render()

                    gr.Markdown("### Add Custom Function")
                    function_form['function_name'].render()
                    function_form['function_description'].render()
                    function_form['function_parameters'].render()
                    function_form['add_function_button'].render()

                    gr.Markdown("### Manage Functions")
                    with gr.Row():
                        function_form['toggle_function_name'].render()
                        function_form['toggle_function_enabled'].render()
                        function_form['toggle_function_button'].render()

                    with gr.Row():
                        function_form['remove_function_name'].render()
                        function_form['remove_function_button'].render()

                    function_form['function_operation_status'].render()

                    gr.Markdown("### Test & History")
                    with gr.Row():
                        function_form['test_function_select'].render()
                    with gr.Row():
                        function_form['test_parameters_preview'].render()
                    with gr.Row():
                        function_form['test_arguments'].render()
                    with gr.Row():
                        function_form['execute_test_button'].render()
                    with gr.Row():
                        function_form['test_result'].render()
                    with gr.Row():
                        function_form['history_table'].render()
                        function_form['clear_history_button'].render()

                    gr.Markdown("### Import / Export & Tools Preview")
                    with gr.Row():
                        function_form['export_button'].render()
                        function_form['import_button'].render()
                        function_form['tools_preview_button'].render()
                    with gr.Row():
                        function_form['export_json'].render()
                    with gr.Row():
                        function_form['import_json'].render()
                    with gr.Row():
                        function_form['tools_preview'].render()

            with gr.Column(scale=8):
                with gr.Row(equal_height=True):
                    chat_system_prompt_textbox.render()
                    chat_default_system_prompt_button.render()

                    chat_default_system_prompt_button.click(
                        fn=get_default_system_prompt_callback,
                        outputs=[chat_system_prompt_textbox]
                    )
                    chat_system_prompt_textbox.change(
                        fn=model_manager.set_custom_prompt,
                        inputs=[chat_system_prompt_textbox]
                    )

                chatbot = gr.Chatbot(
                    type="messages",
                    show_copy_button=True,
                    render=False,
                    latex_delimiters=[
                        {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                        {"left": "\\begin{equation*}", "right": "\\end{equation*}", "display": True},
                        {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                        {"left": "\\begin{align*}", "right": "\\end{align*}", "display": True},
                        {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
                        {"left": "\\begin{alignat*}", "right": "\\end{alignat*}", "display": True},
                        {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
                        {"left": "\\begin{gather*}", "right": "\\end{gather*}", "display": True},
                        {"left": "\\begin{eqnarray}", "right": "\\end{eqnarray}", "display": True},
                        {"left": "\\begin{eqnarray*}", "right": "\\end{eqnarray*}", "display": True},
                        {"left": "\\begin{multline}", "right": "\\end{multline}", "display": True},
                        {"left": "\\begin{multline*}", "right": "\\end{multline*}", "display": True},
                        {"left": "\\begin{split}", "right": "\\end{split}", "display": True},
                        {"left": "\\begin{cases}", "right": "\\end{cases}", "display": True},
                        {"left": "\\begin{matrix}", "right": "\\end{matrix}", "display": True},
                        {"left": "\\begin{pmatrix}", "right": "\\end{pmatrix}", "display": True},
                        {"left": "\\begin{bmatrix}", "right": "\\end{bmatrix}", "display": True},
                        {"left": "\\begin{vmatrix}", "right": "\\end{vmatrix}", "display": True},
                        {"left": "\\begin{Vmatrix}", "right": "\\end{Vmatrix}", "display": True},
                        {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
                        {"left": "\\[", "right": "\\]", "display": True},
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "\\(", "right": "\\)", "display": False},
                        {"left": "$", "right": "$", "display": False}
                    ]
                )

                chatbot.clear(
                    fn=clear_cache
                )

                gr.ChatInterface(
                    multimodal=True,
                    chatbot=chatbot,
                    type="messages",
                    fn=managed_chat_generator,
                    title=None,
                    autofocus=False,
                    fill_height=True,
                    fill_width=True,
                    save_history=True,
                    additional_inputs=[chat_system_prompt_textbox] + list(chat_params.values()) + [chat_rag_form['rag_enabled'], rag_params['n_results'], function_form['function_calling_enabled']]
                )

    with gr.Tab(get_text("Tab.completion"), interactive=True):
        with gr.Row():
            with gr.Column(scale=2):
                completion_memory.render()

                gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                completion_model_selector.render()
                completion_model_status.render()
                completion_load_button.render()

                with gr.Row(visible=False):
                    with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=True):
                        for slider in completion_params.values():
                            slider.render()

            with gr.Column(scale=8):
                completion_interface = gr.Interface(
                    clear_btn=None,
                    flagging_mode="never",
                    fn=managed_completion_generator,
                    inputs=[
                               gr.Textbox(lines=10, show_copy_button=True, render=True,
                                          label=get_text("Page.Completion.Textbox.prompt.label")),
                           ] + list(completion_params.values()),
                    outputs=[
                        gr.Textbox(lines=25, show_copy_button=True, render=True,
                                   label=get_text("Page.Completion.Textbox.output.label"))
                    ],
                    submit_btn=get_text("Page.Completion.Button.submit.value"),
                    stop_btn=get_text("Page.Completion.Button.stop.value"),
                )

    with gr.Tab(get_text("Tab.model_management"), interactive=True):
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                model_list.render()

            with gr.Column(scale=5):
                gr.Markdown(f"## {get_text('Page.ModelManagement.DeleteModelBlock.Markdown.add_model')}")
                local_model_form['search_query'].render()
                local_model_form['search_button'].render()
                local_model_form['search_results'].render()
                local_model_form['model_name'].render()
                local_model_form['mlx_repo'].render()
                local_model_form['quantize'].render()
                local_model_form['default_language'].render()
                local_model_form['system_prompt'].render()
                local_model_form['multimodal'].render()
                local_model_form['add_button'].render()
                
                gr.Markdown(f"## {get_text('Page.ModelManagement.DeleteModelBlock.Markdown.delete_model')}")
                local_model_form['delete_model_selector'].render()
                local_model_form['delete_files_checkbox'].render()
                local_model_form['delete_button'].render()
                local_model_form['delete_status'].render()

    setup_model_sync_events(
        chat_model_selector, completion_model_selector,
        chat_load_button, completion_load_button,
        chat_model_status, completion_model_status,
        chat_system_prompt_textbox,
        chat_params['max_tokens'], completion_params['max_tokens']
    )

    setup_model_management_events(
        local_model_form, model_list,
        chat_model_selector, completion_model_selector
    )

    chat_rag_form['rag_enabled'].change(
        fn=toggle_rag_enabled,
        inputs=[chat_rag_form['rag_enabled']],
        outputs=[chat_rag_form['upload_status']]
    )

    chat_rag_form['upload_button'].click(
        fn=upload_and_index_file,
        inputs=[chat_rag_form['file_upload']],
        outputs=[chat_rag_form['upload_status'], chat_rag_form['rag_status']]
    )

    chat_rag_form['clear_button'].click(
        fn=clear_rag_index,
        outputs=[chat_rag_form['upload_status'], chat_rag_form['rag_status']]
    )

    chat_rag_form['update_params_button'].click(
        fn=update_rag_parameters,
        inputs=[
            rag_params['chunk_size'],
            rag_params['chunk_overlap'],
            rag_params['similarity_threshold']
        ],
        outputs=[chat_rag_form['params_status']]
    )

    function_form['function_calling_enabled'].change(
        fn=toggle_function_calling_enabled,
        inputs=[function_form['function_calling_enabled']],
        outputs=[function_form['function_operation_status']]
    ).then(
        fn=get_function_status,
        outputs=[function_form['function_status']]
    )

    function_form['execution_mode'].change(
        fn=set_function_execution_mode,
        inputs=[function_form['execution_mode']],
        outputs=[function_form['function_status']]
    )

    function_form['enable_all_button'].click(
        fn=enable_all_functions,
        outputs=[function_form['function_operation_status'], function_form['function_list'], function_form['function_status']]
    )

    function_form['disable_all_button'].click(
        fn=disable_all_functions,
        outputs=[function_form['function_operation_status'], function_form['function_list'], function_form['function_status']]
    )

    function_form['function_list'].change(
        fn=apply_function_list_enabled_from_df,
        inputs=[function_form['function_list']],
        outputs=[function_form['function_operation_status'], function_form['function_status'], function_form['function_list']]
    )

    function_form['add_function_button'].click(
        fn=add_custom_function,
        inputs=[
            function_form['function_name'],
            function_form['function_description'],
            function_form['function_parameters']
        ],
        outputs=[function_form['function_operation_status'], function_form['function_list']]
    ).then(
        fn=get_function_status,
        outputs=[function_form['function_status']]
    ).then(
        fn=update_test_function_choices,
        outputs=[function_form['test_function_select']]
    )

    function_form['toggle_function_button'].click(
        fn=toggle_function,
        inputs=[
            function_form['toggle_function_name'],
            function_form['toggle_function_enabled']
        ],
        outputs=[function_form['function_operation_status'], function_form['function_list']]
    ).then(
        fn=get_function_status,
        outputs=[function_form['function_status']]
    ).then(
        fn=update_test_function_choices,
        outputs=[function_form['test_function_select']]
    )

    function_form['remove_function_button'].click(
        fn=remove_function,
        inputs=[function_form['remove_function_name']],
        outputs=[function_form['function_operation_status'], function_form['function_list']]
    ).then(
        fn=get_function_status,
        outputs=[function_form['function_status']]
    ).then(
        fn=update_test_function_choices,
        outputs=[function_form['test_function_select']]
    )

    function_form['test_function_select'].change(
        fn=get_function_schema_and_sample,
        inputs=[function_form['test_function_select']],
        outputs=[function_form['test_parameters_preview'], function_form['test_arguments']]
    )

    function_form['execute_test_button'].click(
        fn=execute_function_test,
        inputs=[function_form['test_function_select'], function_form['test_arguments']],
        outputs=[function_form['test_result'], function_form['function_operation_status'], function_form['history_table']]
    )

    function_form['clear_history_button'].click(
        fn=clear_function_history,
        outputs=[function_form['history_table'], function_form['function_operation_status']]
    )

    function_form['export_button'].click(
        fn=export_custom_functions_json,
        outputs=[function_form['export_json']]
    )

    function_form['import_button'].click(
        fn=import_custom_functions_from_json,
        inputs=[function_form['import_json']],
        outputs=[function_form['function_operation_status'], function_form['function_list'], function_form['function_status'], function_form['test_function_select']]
    )

    function_form['tools_preview_button'].click(
        fn=get_openai_tools_preview,
        outputs=[function_form['tools_preview']]
    )

    app.load(
        fn=update_model_management_models_list,
        outputs=[model_list]
    ).then(
        fn=update_model_selector_choices,
        outputs=[chat_model_selector]
    ).then(
        fn=update_model_selector_choices,
        outputs=[completion_model_selector]
    ).then(
        fn=update_model_max_length,
        inputs=[chat_params['max_tokens']],
        outputs=[chat_params['max_tokens']]
    ).then(
        fn=update_model_max_length,
        inputs=[completion_params['max_tokens']],
        outputs=[completion_params['max_tokens']]
    ).then(
        fn=update_all_memory_usage,
        outputs=[chat_memory, completion_memory]
    ).then(
        fn=get_rag_status,
        outputs=[chat_rag_form['rag_status']]
    ).then(
        fn=get_function_status,
        outputs=[function_form['function_status']]
    ).then(
        fn=update_test_function_choices,
        outputs=[function_form['test_function_select']]
    )


def exit_handler():
    model_manager.close_model()


atexit.register(exit_handler)


def start(port: int, share: bool = False, in_browser: bool = True) -> None:
    logger.info(f"Starting the app on port {port} with share={share} and in_browser={in_browser}")
    app.launch(server_port=port, inbrowser=in_browser, share=share, pwa=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with MLX")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="The port number to run the application on (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable sharing the application link externally"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the application in the default web browser"
    )
    args = parser.parse_args()

    start(port=args.port, share=args.share, in_browser=not args.no_browser)


if __name__ == "__main__":
    main()
