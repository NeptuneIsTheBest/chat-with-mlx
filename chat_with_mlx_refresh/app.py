import argparse
import atexit
import base64
import copy
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import gradio as gr
import mlx.core.metal
from gradio.components.chatbot import ChatMessage

from .chat import ChatSystemPromptBlock, LoadModelBlock, AdvancedSettingBlock, RAGSettingBlock, SystemStatusBlock
from .language import get_text
from .model import Message, MessageRole, ModelManager, TextModel, VisionModel, OpenAIModel
from .model_management import AddLocalModelBlock, AddAPIModelBlock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_manager = ModelManager()

chat_system_status_block = SystemStatusBlock(model_manager)
chat_system_prompt_block = ChatSystemPromptBlock(model_manager=model_manager)
chat_load_model_block = LoadModelBlock(model_manager=model_manager)
chat_advanced_setting_block = AdvancedSettingBlock()
chat_rag_setting_block = RAGSettingBlock()

completion_system_status_block = SystemStatusBlock(model_manager=model_manager)
completion_load_model_block = LoadModelBlock(model_manager=model_manager)

completion_advanced_setting_block = AdvancedSettingBlock()

model_management_add_loacl_model_block = AddLocalModelBlock(model_manager=model_manager)
model_management_add_api_model_block = AddAPIModelBlock(model_manager=model_manager)


def get_loaded_model() -> Union[TextModel, VisionModel, OpenAIModel]:
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


try:
    import pypdf
except ImportError:
    pypdf = None
    logger.warning("pypdf not found.")

try:
    import docx
except ImportError:
    docx = None
    logger.warning("docx not found.")

try:
    from pptx import Presentation
except ImportError:
    Presentation = None
    logger.warning("pptx not found.")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas not found.")


class FileManager:
    def __init__(self):
        self.files = {}

    def format_content(self, file_name, content):
        boundary_start = f"<<<BEGIN FILE:{file_name}>>>"
        boundary_end = f"<<<END FILE:{file_name}>>>"
        content = content.replace(boundary_start, '')
        content = content.replace(boundary_end, '')
        return f"{boundary_start}\n{content}\n{boundary_end}"

    def load_file(self, file_name: Path):
        if file_name.name in self.files:
            if self.files[file_name.name]["md5"] == get_file_md5(file_name):
                return self.files[file_name.name]["content"]
        suffix = file_name.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(file_name)
        elif suffix in [".txt", ".csv", ".md"]:
            return self.load_txt_like(file_name)
        elif suffix == ".docx":
            return self.load_docx(file_name)
        elif suffix == ".pptx":
            return self.load_pptx(file_name)
        elif suffix in [".xlsx", ".xls"]:
            return self.load_excel(file_name)
        else:
            return None

    def load_pdf(self, file_name: Path):
        if not pypdf:
            logger.warning("pypdf not found.")
            return None
        pdf = pypdf.PdfReader(file_name)
        content = ''
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content += text
        formatted_content = self.format_content(file_name, content)
        self.files[file_name.name] = {
            "md5": get_file_md5(file_name),
            "content": formatted_content
        }
        return formatted_content

    def load_txt_like(self, file_name: Path):
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()
        formatted_content = self.format_content(file_name, content)
        self.files[file_name.name] = {
            "md5": get_file_md5(file_name),
            "content": formatted_content
        }
        return formatted_content

    def load_docx(self, file_name: Path):
        if not docx:
            logger.warning("docx not found.")
            return None
        doc = docx.Document(str(file_name))
        content = "\n".join([para.text for para in doc.paragraphs])
        formatted_content = self.format_content(file_name, content)
        self.files[file_name.name] = {
            "md5": get_file_md5(file_name),
            "content": formatted_content
        }
        return formatted_content

    def load_pptx(self, file_name: Path):
        if not Presentation:
            logger.warning("pptx not found.")
            return None
        prs = Presentation(str(file_name))
        content = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    content += shape.text + "\n"
        formatted_content = self.format_content(file_name, content)
        self.files[file_name.name] = {
            "md5": get_file_md5(file_name),
            "content": formatted_content
        }
        return formatted_content

    def load_excel(self, file_name: Path):
        if not pd:
            logger.warning("pandas not found.")
            return None
        excel_file = pd.ExcelFile(file_name)
        content = ""
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            content += f"Sheet: {sheet_name}\n"
            content += df.to_csv(index=False)
        formatted_content = self.format_content(file_name, content)
        self.files[file_name.name] = {
            "md5": get_file_md5(file_name),
            "content": formatted_content
        }
        return formatted_content


file_manager = FileManager()


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
            if suffix in [".png", ".jpg", ".jpeg", ".heif", ".heic"]:  # Image
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


def build_openai_api_messages(current_message_dict: Dict, history_list: List[Dict], system_prompt: Optional[str], ) -> List[Dict]:
    api_messages = []
    if system_prompt and system_prompt.strip():
        api_messages.append({"role": "system", "content": system_prompt})

    i = 0
    while i < len(history_list):
        hist_item = history_list[i]
        role = hist_item.get("role")
        content = hist_item.get("content")

        if role not in ("user", "assistant", "tool"):
            logger.warning(f"Skipping history item with unexpected role: {role}")
            i += 1
            continue

        if isinstance(content, tuple):
            files_in_hist_item = list(content)
            text_for_files = None
            if i + 1 < len(history_list) and \
                    history_list[i + 1].get("role") == role and \
                    isinstance(history_list[i + 1].get("content"), str):
                text_for_files = history_list[i + 1]["content"]
                i += 1

            processed_content_parts = prepare_openai_message_content(text_for_files, files_in_hist_item)
        elif isinstance(content, str):
            processed_content_parts = prepare_openai_message_content(content, None)
        elif isinstance(content, list):
            processed_content_parts = content
        else:
            logger.warning(f"Skipping history item with unexpected content type: {type(content)}")
            i += 1
            continue

        if processed_content_parts:
            api_messages.append({"role": role, "content": processed_content_parts})
        i += 1

    user_text = current_message_dict.get("text")
    user_files = current_message_dict.get("files")
    current_user_content_parts = prepare_openai_message_content(user_text, user_files)
    if current_user_content_parts:
        api_messages.append({"role": "user", "content": current_user_content_parts})

    return api_messages


def prepare_generic_model_inputs(current_message_dict: Dict, history_list: List[Dict], system_prompt: Optional[str], model_instance: Union[TextModel, VisionModel]) -> Tuple[str, List[Dict], List[str]]:
    effective_history = []
    if system_prompt and system_prompt.strip() != "":
        effective_history.append(Message(MessageRole.SYSTEM, content=system_prompt).to_dict())
    effective_history.extend(history_list)

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


def handle_chat(message: Dict,
                history: List[Dict],
                system_prompt: str = None,
                temperature: float = 0.7,
                top_k: int = 20,
                top_p: float = 0.9,
                min_p: float = 0.0,
                max_tokens: int = 512,
                repetition_penalty: float = 1.0,
                stream: bool = True):
    CHATML_CONTROL_TOKENS = [
        '<|im_start|>', '<|im_end|>',
        '<|system|>', '<|user|>', '<|assistant|>',
        '<|end|>', '<|endoftext|>'
    ]

    CHATML_STOP_TOKENS = ['<|im_end|>', '<|end|>', '<|endoftext|>']

    def filter_chatml_tokens(text: str) -> tuple[str, bool]:
        should_stop = False
        filtered_text = text

        for stop_token in CHATML_STOP_TOKENS:
            if stop_token in text:
                should_stop = True
                stop_index = text.find(stop_token)
                filtered_text = text[:stop_index]
                break

        for token in CHATML_CONTROL_TOKENS:
            filtered_text = filtered_text.replace(token, '')

        return filtered_text, should_stop

    try:
        model = get_loaded_model()

        if isinstance(model, OpenAIModel):
            api_messages = build_openai_api_messages(message, history, system_prompt)

            response_stream = model.generate_response(messages=api_messages, stream=stream, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=1)

            chat_message_accumulator = ChatMessage(role="assistant", content="")
            thinking_message = None
            thinking_content = ""
            final_content = ""
            in_thinking = False
            thinking_start_time = None
            chunk_buffer = ""

            if stream:
                for chunk in response_stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            chunk_text = chunk_buffer + delta.content
                            chunk_buffer = ""

                            for partial in ["<", "<t", "<th", "<thi", "<thin", "<think", "</", "</t", "</th", "</thi", "</thin", "</think"]:
                                if chunk_text.endswith(partial):
                                    chunk_buffer = partial
                                    chunk_text = chunk_text[:-len(partial)]
                                    break

                            if not chunk_text:
                                continue

                            chunk_text, should_stop = filter_chatml_tokens(chunk_text)

                            if should_stop:
                                if in_thinking:
                                    thinking_content += chunk_text
                                    thinking_message.content = thinking_content
                                    thinking_message.metadata["title"] = "Thought for"
                                    thinking_message.metadata["status"] = "done"
                                    thinking_message.metadata["duration"] = time.time() - thinking_start_time
                                    yield [thinking_message, chat_message_accumulator]
                                else:
                                    final_content += chunk_text
                                    chat_message_accumulator.content = final_content
                                    if thinking_message and thinking_message.metadata.get("status") == "done":
                                        yield [thinking_message, chat_message_accumulator]
                                    else:
                                        yield chat_message_accumulator
                                break

                            if "<think>" in chunk_text and not in_thinking:
                                in_thinking = True
                                thinking_start_time = time.time()
                                thinking_message = ChatMessage(
                                    role="assistant",
                                    content="",
                                    metadata={"title": "Thinking", "id": 0, "status": "pending"}
                                )
                                think_start_idx = chunk_text.find("<think>") + len("<think>")
                                if think_start_idx < len(chunk_text):
                                    thinking_content += chunk_text[think_start_idx:]
                                    thinking_message.content = thinking_content
                                yield thinking_message
                                continue

                            elif in_thinking and "</think>" not in chunk_text:
                                thinking_content += chunk_text
                                thinking_message.content = thinking_content
                                yield thinking_message
                                continue

                            elif in_thinking and "</think>" in chunk_text:
                                think_end_idx = chunk_text.find("</think>")
                                thinking_content += chunk_text[:think_end_idx]
                                thinking_message.content = thinking_content
                                thinking_message.metadata["title"] = "Thought for"
                                thinking_message.metadata["status"] = "done"
                                thinking_message.metadata["duration"] = time.time() - thinking_start_time

                                remaining_content = chunk_text[think_end_idx + len("</think>"):]
                                if remaining_content.strip():
                                    final_content += remaining_content
                                    chat_message_accumulator.content = final_content

                                yield [thinking_message, chat_message_accumulator]
                                in_thinking = False
                                continue

                            elif not in_thinking:
                                if "<think>" in chunk_text and "</think>" in chunk_text:
                                    think_match = re.search(r'<think>(.*?)</think>', chunk_text, re.DOTALL)
                                    if think_match:
                                        thinking_start_time = time.time()
                                        extracted_thinking = think_match.group(1)
                                        thinking_message = ChatMessage(
                                            role="assistant",
                                            content=extracted_thinking,
                                            metadata={
                                                "title": "Thought for",
                                                "id": 0,
                                                "status": "done",
                                                "duration": 0.1
                                            }
                                        )

                                        before_think = chunk_text[:chunk_text.find("<think>")]
                                        after_think = chunk_text[chunk_text.find("</think>") + len("</think>"):]
                                        final_content += before_think + after_think
                                        chat_message_accumulator.content = final_content

                                        yield [thinking_message, chat_message_accumulator]
                                        continue
                                else:
                                    final_content += chunk_text
                                    chat_message_accumulator.content = final_content
                                    if thinking_message and thinking_message.metadata.get("status") == "done":
                                        yield [thinking_message, chat_message_accumulator]
                                    else:
                                        yield chat_message_accumulator
            else:
                if response_stream.choices:
                    full_content = response_stream.choices[0].message.content

                    full_content, _ = filter_chatml_tokens(full_content)

                    think_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
                    if think_match:
                        thinking_content = think_match.group(1)
                        thinking_message = ChatMessage(
                            role="assistant",
                            content=thinking_content,
                            metadata={
                                "title": "Thought for",
                                "id": 0,
                                "status": "done",
                                "duration": 0.1
                            }
                        )

                        final_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
                        chat_message_accumulator.content = final_content

                        yield [thinking_message, chat_message_accumulator]
                    else:
                        chat_message_accumulator.content = full_content
                        yield chat_message_accumulator
                else:
                    logger.error("OpenAI non-stream response had no choices.")
                    yield ChatMessage(role="assistant", content="Error: No response from model.")
        else:
            tokenizer_to_check = None
            if isinstance(model, VisionModel):
                if model.processor and hasattr(model.processor, 'tokenizer'):
                    tokenizer_to_check = model.processor.tokenizer
            elif hasattr(model, 'tokenizer'):
                tokenizer_to_check = model.tokenizer

            if tokenizer_to_check and tokenizer_to_check.chat_template is None:
                raise RuntimeError("Model {} does not have a chat template. Please use the 'Completion' tab or set a chat template.".format(model.model_name if hasattr(model, 'model_name') else type(model).__name__))

            processed_message_text, processed_history_list, image_paths = prepare_generic_model_inputs(message, history, system_prompt, model)

            temperature = float(temperature)
            top_k = int(top_k)
            top_p = float(top_p)
            min_p = float(min_p)
            repetition_penalty = float(repetition_penalty)

            response_args = {
                "message": processed_message_text,
                "history": processed_history_list,
                "stream": stream,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
            }
            if isinstance(model, VisionModel):
                response_args["images"] = image_paths
                eos_token = model.processor.tokenizer.eos_token if model.processor and model.processor.tokenizer else None
            else:
                eos_token = model.tokenizer.eos_token if model.tokenizer else None

            response_stream = model.generate_response(**response_args)

            chat_message_accumulator = ChatMessage(role="assistant", content="")
            thinking_message = None
            thinking_content = ""
            final_content = ""
            in_thinking = False
            thinking_start_time = None
            chunk_buffer = ""

            for chunk in response_stream:
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
                    if stream and eos_token and eos_token in chunk_text:
                        if chunk_text == eos_token:
                            break
                        chunk_text = chunk_text.split(eos_token)[0]

                    chunk_text = chunk_buffer + chunk_text
                    chunk_buffer = ""

                    for partial in ["<", "<t", "<th", "<thi", "<thin", "<think", "</", "</t", "</th", "</thi", "</thin", "</think"]:
                        if chunk_text.endswith(partial):
                            chunk_buffer = partial
                            chunk_text = chunk_text[:-len(partial)]
                            break

                    if not chunk_text:
                        continue

                    chunk_text, should_stop = filter_chatml_tokens(chunk_text)

                    if should_stop:
                        if in_thinking:
                            thinking_content += chunk_text
                            thinking_message.content = thinking_content
                            thinking_message.metadata["title"] = "Thought for"
                            thinking_message.metadata["status"] = "done"
                            thinking_message.metadata["duration"] = time.time() - thinking_start_time
                            if stream:
                                yield [thinking_message, chat_message_accumulator]
                        else:
                            final_content += chunk_text
                            chat_message_accumulator.content = final_content
                            if thinking_message and thinking_message.metadata.get("status") == "done":
                                if stream:
                                    yield [thinking_message, chat_message_accumulator]
                            else:
                                if stream:
                                    yield chat_message_accumulator
                        break

                    if "<think>" in chunk_text and not in_thinking:
                        in_thinking = True
                        thinking_start_time = time.time()
                        thinking_message = ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"title": "Thinking", "id": 0, "status": "pending"}
                        )
                        think_start_idx = chunk_text.find("<think>") + len("<think>")
                        if think_start_idx < len(chunk_text):
                            thinking_content += chunk_text[think_start_idx:]
                            thinking_message.content = thinking_content
                        if stream:
                            yield thinking_message
                        continue

                    elif in_thinking and "</think>" not in chunk_text:
                        thinking_content += chunk_text
                        thinking_message.content = thinking_content
                        if stream:
                            yield thinking_message
                        continue

                    elif in_thinking and "</think>" in chunk_text:
                        think_end_idx = chunk_text.find("</think>")
                        thinking_content += chunk_text[:think_end_idx]
                        thinking_message.content = thinking_content
                        thinking_message.metadata['title'] = "Thought for"
                        thinking_message.metadata["status"] = "done"
                        thinking_message.metadata["duration"] = time.time() - thinking_start_time

                        remaining_content = chunk_text[think_end_idx + len("</think>"):]
                        if remaining_content.strip():
                            final_content += remaining_content
                            chat_message_accumulator.content = final_content

                        if stream:
                            yield [thinking_message, chat_message_accumulator]
                        in_thinking = False
                        continue

                    elif not in_thinking:
                        if "<think>" in chunk_text and "</think>" in chunk_text:
                            think_match = re.search(r'<think>(.*?)</think>', chunk_text, re.DOTALL)
                            if think_match:
                                thinking_start_time = time.time()
                                extracted_thinking = think_match.group(1)
                                thinking_message = ChatMessage(
                                    role="assistant",
                                    content=extracted_thinking,
                                    metadata={
                                        "title": "Thought for",
                                        "id": 0,
                                        "status": "done",
                                        "duration": 0.1
                                    }
                                )

                                before_think = chunk_text[:chunk_text.find("<think>")]
                                after_think = chunk_text[chunk_text.find("</think>") + len("</think>"):]
                                final_content += before_think + after_think
                                chat_message_accumulator.content = final_content

                                if stream:
                                    yield [thinking_message, chat_message_accumulator]
                                continue
                        else:
                            final_content += chunk_text
                            chat_message_accumulator.content = final_content
                            if thinking_message and thinking_message.metadata.get("status") == "done":
                                if stream:
                                    yield [thinking_message, chat_message_accumulator]
                            else:
                                if stream:
                                    yield chat_message_accumulator

                    if stream and eos_token and eos_token in chunk_text:
                        break

            if not stream:
                if thinking_message and thinking_message.metadata.get("status") == "done":
                    yield [thinking_message, chat_message_accumulator]
                else:
                    yield chat_message_accumulator
    except Exception as e:
        logger.exception("Error in handle_chat:")
        raise gr.Error(str(e))


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
        if isinstance(model, VisionModel) or isinstance(model, OpenAIModel):
            raise RuntimeError("Not supported yet.")

        temperature = float(temperature)
        top_p = float(top_p)
        repetition_penalty = float(repetition_penalty)

        if not stream:
            return prompt + model.generate_completion(
                prompt=prompt,
                stream=stream,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty)
        else:
            response = prompt
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
                chunk_text = ""
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, "text"):
                    chunk_text = chunk.text
                elif hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                else:
                    logger.warning(f"Unexpected chunk type from model: {type(chunk)}")
                if eos_token not in chunk_text:
                    response += chunk_text
                    yield response
                else:
                    yield response
                    break
            return None
    except Exception as e:
        raise gr.Error(str(e))


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


def chat_load_model_callback(model_name: str):
    return load_model(model_name)


def completion_load_model_callback(model_name: str):
    return load_model(model_name)[0]


def get_default_system_prompt_callback():
    if model_manager.get_loaded_model_config():
        return model_manager.get_system_prompt(default=True)
    else:
        raise gr.Error("No model loaded.")


def update_model_management_models_list():
    return pd.DataFrame({get_text("Page.ModelManagement.Dataframe.model_list.headers"): model_manager.get_model_list()})


def update_model_selector_choices():
    return gr.update(choices=model_manager.get_model_list(), value=chat_load_model_block.update_select_model_dropdown_value())


def add_model(model_name: Optional[str], original_repo: str, mlx_repo: str, quantize: str, default_language: str, default_system_prompt: Optional[str], multimodal_ability: List[str]):
    try:
        model_manager.add_config(original_repo, mlx_repo, model_name, quantize, default_language, default_system_prompt, multimodal_ability)
    except Exception as e:
        raise gr.Error(str(e))


def add_api_model(model_name: str, api_key: str, nick_name: Optional[str] = None, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
    try:
        model_manager.add_api_config(model_name, api_key, nick_name=nick_name, base_url=base_url, system_prompt=system_prompt)
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
    model = model_manager.get_loaded_model()
    if model is not None:
        if isinstance(model, OpenAIModel):
            return update_slider_config(1, 1048576, slider_value)
        try:
            model_max_length = model.tokenizer.model_max_length
            if model_max_length is None or model_max_length <= 0:
                model_max_length = 32768
        except Exception as e:
            model_max_length = 32768
            logger.error("Error while updating model max length.")
        return update_slider_config(1, model_max_length, slider_value)
    else:
        return gr.update()


def bytes_to_gigabytes(value):
    return value / 1024 ** 3


def update_memory_usage() -> str:
    memory_usage_bytes = model_manager.get_system_memory_usage()
    total_memory_bytes = mlx.core.metal.device_info()["memory_size"]

    memory_usage_gb = bytes_to_gigabytes(memory_usage_bytes) if isinstance(memory_usage_bytes, (int, float)) else "N/A"
    total_memory_gb = bytes_to_gigabytes(total_memory_bytes) if isinstance(total_memory_bytes, (int, float)) else "N/A"

    return "{:.2f} GB | {:.2f} GB".format(memory_usage_gb, total_memory_gb)


with gr.Blocks(fill_height=True, fill_width=True, title="Chat with MLX") as app:
    timer = gr.Timer(value=1, active=True)
    timer.tick(
        fn=lambda: update_memory_usage(),
        outputs=[chat_system_status_block.memory_usage_textbox]
    ).then(
        fn=lambda: update_memory_usage(),
        outputs=[completion_system_status_block.memory_usage_textbox]
    )

    gr.HTML("<h1>Chat with MLX</h1>")

    with gr.Tab(get_text("Tab.chat")):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    chat_system_status_block.render_all()

                with gr.Row():
                    gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                    chat_load_model_block.render_all()
                    chat_load_model_block.model_selector_dropdown.select(
                        fn=lambda x: x,
                        inputs=[chat_load_model_block.model_selector_dropdown],
                        outputs=[completion_load_model_block.model_selector_dropdown]
                    )
                    chat_load_model_block.load_model_button.click(
                        fn=chat_load_model_callback,
                        inputs=[chat_load_model_block.model_selector_dropdown],
                        outputs=[
                            chat_load_model_block.model_status_textbox,
                            chat_system_prompt_block.system_prompt_textbox
                        ]
                    ).then(
                        fn=lambda x: x,
                        inputs=[chat_load_model_block.model_status_textbox],
                        outputs=[completion_load_model_block.model_status_textbox]
                    ).then(
                        fn=update_model_max_length,
                        inputs=[chat_advanced_setting_block.max_tokens_slider],
                        outputs=[chat_advanced_setting_block.max_tokens_slider]
                    ).then(
                        fn=update_model_max_length,
                        inputs=[completion_advanced_setting_block.max_tokens_slider],
                        outputs=[completion_advanced_setting_block.max_tokens_slider]
                    )

                with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=False):
                    chat_advanced_setting_block.render_all()

                with gr.Accordion(label=get_text("Page.Chat.Accordion.RAGSetting.label"), open=False):
                    chat_rag_setting_block.render_all()

            with gr.Column(scale=8):
                with gr.Row(equal_height=True):
                    chat_system_prompt_block.render_all()
                    chat_system_prompt_block.default_system_prompt_button.click(
                        fn=get_default_system_prompt_callback,
                        outputs=[chat_system_prompt_block.system_prompt_textbox]
                    )
                    chat_system_prompt_block.system_prompt_textbox.change(
                        fn=model_manager.set_custom_prompt,
                        inputs=[chat_system_prompt_block.system_prompt_textbox]
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

                gr.ChatInterface(
                    multimodal=True,
                    chatbot=chatbot,
                    type="messages",
                    fn=handle_chat,
                    title=None,
                    autofocus=False,
                    additional_inputs=[
                        chat_system_prompt_block.system_prompt_textbox,
                        chat_advanced_setting_block.temperature_slider,
                        chat_advanced_setting_block.top_k_slider,
                        chat_advanced_setting_block.top_p_slider,
                        chat_advanced_setting_block.min_p_slider,
                        chat_advanced_setting_block.max_tokens_slider,
                        chat_advanced_setting_block.repetition_penalty_slider
                    ]
                )

    with gr.Tab(get_text("Tab.completion"), interactive=True):
        with gr.Row():
            with gr.Column(scale=2):
                completion_system_status_block.render_all()

                gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                completion_load_model_block.render_all()
                completion_load_model_block.model_selector_dropdown.select(
                    fn=lambda x: x,
                    inputs=[completion_load_model_block.model_selector_dropdown],
                    outputs=[chat_load_model_block.model_selector_dropdown]
                )
                completion_load_model_block.load_model_button.click(
                    fn=completion_load_model_callback,
                    inputs=[completion_load_model_block.model_selector_dropdown],
                    outputs=[
                        completion_load_model_block.model_status_textbox
                    ]
                ).then(
                    fn=lambda x: x,
                    inputs=[completion_load_model_block.model_status_textbox],
                    outputs=[chat_load_model_block.model_status_textbox]
                ).then(
                    fn=update_model_max_length,
                    inputs=[chat_advanced_setting_block.max_tokens_slider],
                    outputs=[chat_advanced_setting_block.max_tokens_slider]
                ).then(
                    fn=update_model_max_length,
                    inputs=[completion_advanced_setting_block.max_tokens_slider],
                    outputs=[completion_advanced_setting_block.max_tokens_slider]
                )

                with gr.Row(visible=False):
                    with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=True):
                        completion_advanced_setting_block.render_all()

            with gr.Column(scale=8):
                completion_textbox = gr.Textbox(lines=25, render=False)
                completion_interface = gr.Interface(
                    clear_btn=None,
                    flagging_mode="never",
                    fn=handle_completion,
                    inputs=[
                        gr.Textbox(lines=10, show_copy_button=True, render=True, label=get_text("Page.Completion.Textbox.prompt.label")),
                        completion_advanced_setting_block.temperature_slider,
                        completion_advanced_setting_block.top_k_slider,
                        completion_advanced_setting_block.top_p_slider,
                        completion_advanced_setting_block.min_p_slider,
                        completion_advanced_setting_block.max_tokens_slider,
                        completion_advanced_setting_block.repetition_penalty_slider
                    ],
                    outputs=[
                        gr.Textbox(lines=25, show_copy_button=True, render=True, label=get_text("Page.Completion.Textbox.output.label"))
                    ],
                    submit_btn=get_text("Page.Completion.Button.submit.value"),
                    stop_btn=get_text("Page.Completion.Button.stop.value"),
                )

    model_list = gr.Dataframe(
        headers=[get_text("Page.ModelManagement.Dataframe.model_list.headers")],
        value=update_model_management_models_list(),
        datatype=["str"],
        row_count=(10, "dynamic"),
        render=False,
        interactive=False
    )

    with gr.Tab(get_text("Tab.model_management"), interactive=True):
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                model_list.render()

            with gr.Column(scale=5):
                with gr.Tab(get_text("Page.ModelManagement.Tab.local_model")):
                    model_management_add_loacl_model_block.render_all()
                    model_management_add_loacl_model_block.add_button.click(
                        fn=add_model,
                        inputs=[
                            model_management_add_loacl_model_block.model_name_textbox,
                            model_management_add_loacl_model_block.original_repo_textbox,
                            model_management_add_loacl_model_block.mlx_repo_textbox,
                            model_management_add_loacl_model_block.quantize_dropdown,
                            model_management_add_loacl_model_block.default_language_dropdown,
                            model_management_add_loacl_model_block.default_system_prompt_textbox,
                            model_management_add_loacl_model_block.multimodal_ability_dropdown
                        ]
                    ).then(
                        fn=update_model_management_models_list,
                        outputs=[
                            model_list
                        ]
                    ).then(
                        fn=update_model_selector_choices,
                        outputs=[
                            chat_load_model_block.model_selector_dropdown
                        ]
                    ).then(
                        fn=update_model_selector_choices,
                        outputs=[
                            completion_load_model_block.model_selector_dropdown
                        ]
                    )

                with gr.Tab(get_text("Page.ModelManagement.Tab.openai_api")):
                    model_management_add_api_model_block.render_all()
                    model_management_add_api_model_block.add_button.click(
                        fn=add_api_model,
                        inputs=[
                            model_management_add_api_model_block.model_name_textbox,
                            model_management_add_api_model_block.api_key_textbox,
                            model_management_add_api_model_block.nick_name_textbox,
                            model_management_add_api_model_block.base_url_textbox,
                            model_management_add_api_model_block.default_system_prompt_textbox
                        ]
                    ).then(
                        fn=update_model_management_models_list,
                        outputs=[
                            model_list
                        ]
                    ).then(
                        fn=update_model_selector_choices,
                        outputs=[
                            chat_load_model_block.model_selector_dropdown
                        ]
                    ).then(
                        fn=update_model_selector_choices,
                        outputs=[
                            completion_load_model_block.model_selector_dropdown
                        ]
                    )

    app.load(
        fn=update_model_management_models_list,
        outputs=[
            model_list
        ]
    ).then(
        fn=update_model_selector_choices,
        outputs=[
            chat_load_model_block.model_selector_dropdown
        ]
    ).then(
        fn=update_model_selector_choices,
        outputs=[
            completion_load_model_block.model_selector_dropdown
        ]
    ).then(
        fn=update_model_max_length,
        inputs=[chat_advanced_setting_block.max_tokens_slider],
        outputs=[chat_advanced_setting_block.max_tokens_slider]
    ).then(
        fn=update_model_max_length,
        inputs=[completion_advanced_setting_block.max_tokens_slider],
        outputs=[completion_advanced_setting_block.max_tokens_slider]
    ).then(
        fn=lambda: update_memory_usage(),
        outputs=[chat_system_status_block.memory_usage_textbox]
    ).then(
        fn=lambda: update_memory_usage(),
        outputs=[completion_system_status_block.memory_usage_textbox]
    )


def exit_handler():
    if model_manager.get_loaded_model():
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
