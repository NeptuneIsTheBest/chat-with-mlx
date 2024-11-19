import enum
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List, Optional

from huggingface_hub import snapshot_download
from mlx_lm import load, generate, stream_generate


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
            message: str,
            history: Union[str, List[Dict]],
            stream: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 512,
            repetition_penalty: float = 1.0
    ):
        message = [Message(MessageRole.USER, message).to_dict()]

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


class ModelManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("./models")
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if not (self.base_dir.is_dir() and self.configs_dir.is_dir() and self.models_dir.is_dir()):
            raise IOError("Failed to create the required directory structure.")

        self.model: Optional[Model] = None
        self.model_config: Optional[Dict[str, str]] = None
        self.model_configs: Dict[str, Dict[str, str]] = self.scan_models()

    def get_config_path(self, model_config: Dict[str, str]) -> Path:
        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            model_name = model_config.get('model_name', 'unknown')
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'.")
        mlx_repo_name = mlx_repo.strip().split('/')[-1]
        return self.configs_dir / f"{mlx_repo_name}.json"

    def get_model_path(self, model_config: Dict[str, str]) -> Path:
        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            model_name = model_config.get('model_name', 'unknown')
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'.")
        mlx_repo_name = mlx_repo.strip().split('/')[-1]
        return self.models_dir / mlx_repo_name

    def get_model_list(self) -> List[str]:
        self.model_configs: Dict[str, Dict[str, str]] = self.scan_models()
        return sorted(self.model_configs.keys())

    def create_config_json(self, model_config: Dict[str, str]):
        config_path = self.get_config_path(model_config)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(model_config, f, ensure_ascii=False, indent=4)

    def add_config(
            self,
            original_repo: str,
            mlx_repo: str,
            model_name: Optional[str] = None,
            quantize: Optional[str] = None,
            default_language: str = "multi",
            system_prompt: Optional[str] = None,
            multimodal_ability: Optional[List[str]] = None,
    ):
        if len(original_repo.strip().split("/")) != 2 or len(mlx_repo.strip().split("/")) != 2:
            raise RuntimeError("'original_repo' or 'mlx_repo' not in compliance with the specification.")
        if quantize not in ["None", "4bit", "8bit", "bf16", "bf32"]:
            raise RuntimeError("quantize must be one of 'None', '4bit', '8bit', 'bf16', 'bf32'")
        if default_language not in ["multi"]:
            raise RuntimeError("default_language must be one of 'multi'")
        if system_prompt and system_prompt.strip() == "":
            system_prompt = None
        if multimodal_ability:
            for ability in multimodal_ability:
                if ability not in ["None", "vision"]:
                    raise RuntimeError("multimodal_ability must be one of 'None', 'vision'")
        model_config = {
            "original_repo": original_repo.strip(),
            "mlx_repo": mlx_repo.strip(),
            "model_name": model_name.strip() if model_name and model_name.strip() != "" else mlx_repo.strip().split("/")[-1],
            "quantize": None if quantize == "None" else quantize,
            "default_language": default_language,
            "system_prompt": system_prompt,
            "multimodal_ability": [] if multimodal_ability == ["None"] else multimodal_ability
        }
        self.create_config_json(model_config)
        display_name = f"{model_config['model_name']}({default_language},{quantize})"
        model_config["display_name"] = display_name
        self.model_configs[display_name] = model_config

    def scan_models(self) -> Dict[str, Dict[str, str]]:
        model_configs = {}
        for config_file in self.configs_dir.glob("*.json"):
            if config_file.is_file():
                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        model_config = json.load(f)
                    model_name = model_config.get("model_name")
                    default_language = model_config.get("default_language")
                    quantize = model_config.get("quantize") if model_config.get("quantize") else "None"
                    if not all([model_name, default_language, quantize]):
                        logging.info(f"Skipping incomplete config: {config_file}")
                        continue
                    display_name = f"{model_name}({default_language},{quantize})"
                    model_config["display_name"] = display_name
                    model_configs[display_name] = model_config
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from {config_file}: {e}")
        return model_configs

    def load_model(self, model_name: str):
        if self.model:
            self.close_model()

        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise RuntimeError(f"Model '{model_name}' not found.")

        local_model_path = self.get_model_path(model_config)
        if not local_model_path.exists():
            mlx_repo = model_config.get("mlx_repo")
            try:
                snapshot_download(repo_id=mlx_repo, local_dir=str(local_model_path))
            except Exception as e:
                raise RuntimeError(f"Failed to download model from '{mlx_repo}': {e}")

        try:
            self.model = Model(str(local_model_path))
            self.model_config = model_config
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{local_model_path}': {e}")

    def close_model(self):
        if self.model:
            self.model.close()
            del self.model
            gc.collect()
            self.model = None
            self.model_config = None

    def get_loaded_model(self) -> Optional[Model]:
        return self.model

    def get_loaded_model_config(self) -> Optional[Dict[str, str]]:
        return self.model_config

    def get_system_prompt(self, default=False) -> Optional[str]:
        if self.model_config:
            if not default and "custom_system_prompt" in self.model_config:
                return self.model_config.get("custom_system_prompt")
            return self.model_config.get("system_prompt")
        return None

    def set_custom_prompt(self, custom_system_prompt: str):
        if self.model_config:
            self.model_config["custom_system_prompt"] = custom_system_prompt
