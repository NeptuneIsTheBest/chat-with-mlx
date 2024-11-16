import enum
import gc
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List, Tuple, Optional

from huggingface_hub import snapshot_download
from mlx_lm import load, generate, stream_generate

from language import get_text


class MessageRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> Dict:
        return {"role": self.role.value, "content": self.content}


class Model:
    def __init__(self, model_path: str, max_tokens: int = 4096, ):
        self.model_path = model_path
        self.max_tokens = max_tokens

        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        try:
            # Replace with actual model loading code
            model, tokenizer = load(self.model_path)
            return model, tokenizer
        except Exception as e:
            raise e

    def generate_completion(
            self,
            prompt: str,
            stream: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 512,
            repetition_penalty: float = 1.0
    ):
        try:
            if stream:
                return self._stream_generate(
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
            else:
                return self._generate(
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
        except Exception as e:
            raise e

    def generate_response(
            self,
            message: Dict,
            history: Union[str, List[Dict]],
            stream: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 512,
            repetition_penalty: float = 1.0
    ):
        message = [Message(MessageRole.USER, message["text"]).to_dict()]

        conversation = history + message

        formatted_prompt = self.tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=True)

        try:
            if stream:
                return self._stream_generate(
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
            else:
                return self._generate(
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
        except Exception as e:
            raise e

    def _generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            temp=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

    def _stream_generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        return stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            temp=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()


class LoadModelStatus(enum.Enum):
    NOT_LOADED = get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")
    LOADED = get_text("Page.Chat.LoadModelBlock.Textbox.model_status.loaded_value")


class ModelManager:
    def __init__(self, models_folder: Optional[str] = None):
        self.models_folder = Path(models_folder) if models_folder else Path(os.getenv("MODELS_PATH", "./models"))

        self.configs_folder = self.models_folder / "configs"
        self.models_folder_path = self.models_folder / "models"
        self.models_folder.mkdir(parents=True, exist_ok=True)
        self.configs_folder.mkdir(parents=True, exist_ok=True)
        self.models_folder_path.mkdir(parents=True, exist_ok=True)

        if not (self.models_folder.is_dir() and self.configs_folder.is_dir() and self.models_folder_path.is_dir()):
            raise IOError("Models folder exists but is not a directory structure as expected.")

        self.model: Optional[Model] = None
        self.model_configs: Dict[str, Dict[str, str]] = self.scan_models()

    def get_config_path(self, model_config: Dict[str, str]) -> Path:
        mlx_repo = model_config.get("mlx_repo")
        if mlx_repo is None:
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_config.get("model_name")}'.")
        return Path(self.configs_folder, "{}.json".format(mlx_repo))

    def get_model_path(self, model_config: Dict[str, str]) -> Path:
        mlx_repo = model_config.get("mlx_repo")
        if mlx_repo is None:
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_config.get("model_name")}'.")
        mlx_repo = mlx_repo.strip()
        mlx_repo_name = mlx_repo.split("/")[-1]
        model_path = Path(self.models_folder_path, mlx_repo_name)
        return model_path

    def create_config(self, model_config: Dict[str, str]):
        model_config_json = json.dumps(model_config)
        with open(self.get_config_path(model_config), "w") as f:
            f.write(model_config_json)

    def scan_models(self) -> Dict[str, Dict[str, str]]:
        model_configs = {}
        for config_file in self.configs_folder.glob("*.json"):
            if config_file.is_file():
                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        model_config = json.load(f)
                    model_name = model_config.get("model_name")
                    default_language = model_config.get("default_language")
                    quantize = model_config.get("quantize")
                    if not all([model_name, default_language, quantize]):
                        logging.info(f"Skipping incomplete config: {config_file}")
                        continue
                    key = f"{model_name}({default_language},{quantize})"
                    model_configs[key] = model_config
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from {config_file}: {e}")
        return model_configs

    def load_model(self, model_name: str) -> Tuple[str, Optional[str]]:
        if self.model:
            self.model.close()
            self.model = None

        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise RuntimeError(f"Model '{model_name}' not found.")

        local_model_path = self.get_model_path(model_config)
        if not local_model_path.exists():
            mlx_repo = model_config.get("mlx_repo")
            try:
                local_model_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=mlx_repo, local_dir=str(local_model_path))
            except Exception as e:
                raise RuntimeError(f"Failed to download model from '{mlx_repo}': {e}")

        try:
            self.model = Model(str(local_model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{local_model_path}': {e}")

        system_prompt = model_config.get("system_prompt")
        return model_name, system_prompt

    def get_loaded_model(self) -> Model:
        if self.model:
            return self.model
        else:
            raise RuntimeError("No model loaded.")

    def get_model_list(self) -> list:
        return list(self.model_configs.keys())
