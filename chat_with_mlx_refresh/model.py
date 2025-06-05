import enum
import gc
import json
import logging
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List, Optional

import mlx
import mlx_vlm
from mlx_lm import load, generate, stream_generate, sample_utils
from openai import OpenAI


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
    def __init__(self, model_path: str):
        self.model_path = model_path

        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        try:
            model, tokenizer = load(self.model_path)
            return model, tokenizer
        except Exception as e:
            raise RuntimeError("Failed to load model {}: {}".format(self.model_path, e))

    def generate_completion(self, prompt: str, stream: bool = False, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512, repetition_penalty: float = 1.0):
        try:
            if stream:
                return self._stream_generate(prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
            else:
                return self._generate(prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
        except Exception as e:
            raise e

    def generate_response(self, message: str, history: Union[str, List[Dict]], stream: bool = False, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512, repetition_penalty: float = 1.0):
        message = [Message(MessageRole.USER, message).to_dict()]

        conversation = history + message

        formatted_prompt = self.tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=True)

        try:
            if stream:
                return self._stream_generate(prompt=formatted_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
            else:
                return self._generate(prompt=formatted_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
        except Exception as e:
            raise e

    def _generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        sampler = sample_utils.make_sampler(temp=temperature, top_p=top_p)
        logits_processors = sample_utils.make_logits_processors(repetition_penalty=repetition_penalty)
        return generate(model=self.model, tokenizer=self.tokenizer, prompt=prompt, sampler=sampler, logits_processors=logits_processors, max_tokens=max_tokens)

    def _stream_generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        sampler = sample_utils.make_sampler(temp=temperature, top_p=top_p)
        logits_processors = sample_utils.make_logits_processors(repetition_penalty=repetition_penalty)
        return stream_generate(model=self.model, tokenizer=self.tokenizer, prompt=prompt, sampler=sampler, logits_processors=logits_processors, max_tokens=max_tokens)

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        mlx.core.clear_cache()


class VisionModel:
    def __init__(self, model_path: str):
        self.model_path = model_path

        self.config, self.model, self.processor, self.image_processor = self._load_model()

    def _load_model(self):
        try:
            config = mlx_vlm.utils.load_config(self.model_path)
            model, processor = mlx_vlm.load(self.model_path, processor_config={"trust_remote_code": True})
            image_processor = mlx_vlm.utils.load_image_processor(self.model_path)
            return config, model, processor, image_processor
        except Exception as e:
            raise RuntimeError("Failed to load {}: {}".format(self.model_path, e))

    def generate_completion(self, prompt: str, images: List[str], stream: bool = False, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512, repetition_penalty: float = 1.0):
        if not images or len(images) == 0:
            raise RuntimeError("Text only chat is not supported.")

        try:
            if stream:
                return self._stream_generate(prompt=prompt, images=images, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
            else:
                return self._generate(prompt=prompt, images=images, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
        except Exception as e:
            raise e

    def generate_response(self, message: str, images: List[str], history: Union[str, List[Dict]], stream: bool = False, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512, repetition_penalty: float = 1.0):
        if not images or len(images) == 0:
            raise RuntimeError("Text only chat is not supported.")

        message = [Message(MessageRole.USER, message).to_dict()]

        conversation = history + message

        formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(processor=self.processor, config=self.config, prompt=conversation, add_generation_prompt=True, num_images=len(images))

        try:
            if stream:
                return self._stream_generate(prompt=formatted_prompt, images=images, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
            else:
                return self._generate(prompt=formatted_prompt, images=images, temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
        except Exception as e:
            raise e

    def _generate(self, prompt: str, images: list[str], temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        return mlx_vlm.utils.generate(model=self.model, processor=self.processor, image_processor=self.image_processor, prompt=prompt, image=images, temp=temperature, top_p=top_p, max_tokens=max_tokens,
            repetition_penalty=repetition_penalty)

    def _stream_generate(self, prompt: str, images: list[str], temperature: float, top_p: float, max_tokens: int, repetition_penalty: float):
        return mlx_vlm.utils.stream_generate(model=self.model, processor=self.processor, image_processor=self.image_processor, prompt=prompt, image=images, temp=temperature, top_p=top_p, max_tokens=max_tokens,
            repetition_penalty=repetition_penalty)

    def close(self):
        del self.model
        del self.processor
        del self.image_processor
        del self.config
        gc.collect()
        mlx.core.clear_cache()


class OpenAIModel:
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model_name = model_name

    def generate_response(self, messages: List, reasoning_effort: str = None, stream: bool = False, **kwargs):
        return self.client.chat.completions.create(model=self.model_name, messages=messages, reasoning_effort=reasoning_effort, stream=stream, **kwargs)

    def close(self):
        del self.client
        del self.api_key


class ModelType(enum.Enum):
    LOCAL = "local"
    OPENAI_API = "openai_api"


class ModelManager:
    VALID_QUANTIZE_TYPES = {"None", "4bit", "8bit", "bf16", "bf32"}
    VALID_LANGUAGES = {"multi"}
    VALID_MULTIMODAL_ABILITIES = {"None", "vision"}
    CONFIG_EXTENSION = ".json"

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "models"
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"

        self._setup_directories()

        self.model: Optional[Model] = None
        self.model_config: Optional[Dict[str, Union[str, List[str]]]] = None
        self.model_configs: Dict[str, Dict[str, Union[str, List[str]]]] = self.scan_models()

    def _setup_directories(self) -> None:
        directories = [self.base_dir, self.configs_dir, self.models_dir]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        if not all(d.is_dir() for d in directories):
            raise IOError("Failed to create the required directory structure.")

    def _validate_repo_format(self, repo: str, name: str) -> None:
        if len(repo.strip().split("/")) != 2:
            raise ValueError(f"'{name}' must be in 'owner/repo' format.")

    def _validate_quantize(self, quantize: str) -> None:
        if quantize not in self.VALID_QUANTIZE_TYPES:
            raise ValueError(f"quantize must be one of {self.VALID_QUANTIZE_TYPES}.")

    def _validate_multimodal_ability(self, abilities: Optional[List[str]]) -> None:
        if not abilities:
            return

        if "None" in abilities and len(abilities) > 1:
            raise ValueError("'None' cannot exist with other abilities.")

        invalid_abilities = set(abilities) - self.VALID_MULTIMODAL_ABILITIES
        if invalid_abilities:
            raise ValueError(f"Invalid multimodal abilities: {invalid_abilities}")

    def _extract_repo_name(self, repo: str) -> str:
        return repo.strip().split('/')[-1]

    def _generate_display_name(self, model_config: Dict[str, Union[str, List[str]]]) -> str:
        if model_config.get("type") == ModelType.OPENAI_API.value:
            model_name = model_config["model_name"]
            nick_name = model_config.get("nick_name")
            return f"{nick_name or model_name}({ModelType.OPENAI_API.value})"

        model_name = model_config["model_name"]
        default_language = model_config["default_language"]
        quantize = model_config.get("quantize") or "None"
        multimodal_ability = model_config.get("multimodal_ability", [])

        if multimodal_ability and "None" not in multimodal_ability:
            abilities_str = "".join(multimodal_ability)
            return f"{model_name}({default_language},{quantize},{abilities_str})"

        return f"{model_name}({default_language},{quantize})"

    def get_config_path(self, model_config: Dict[str, Union[str, List[str]]]) -> Path:
        if model_config.get("type") == ModelType.OPENAI_API.value:
            model_name = model_config.get('model_name')
            if not model_name:
                raise RuntimeError("'model_name' not specified for OpenAI API Model.")
            return self.configs_dir / f"{model_name}{self.CONFIG_EXTENSION}"

        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            model_name = model_config.get('model_name', 'unknown')
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'.")

        repo_name = self._extract_repo_name(mlx_repo)
        return self.configs_dir / f"{repo_name}{self.CONFIG_EXTENSION}"

    def get_model_path(self, model_config: Dict[str, Union[str, List[str]]]) -> Path:
        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            model_name = model_config.get('model_name', 'unknown')
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'.")

        return self.models_dir / self._extract_repo_name(mlx_repo)

    def get_model_list(self) -> List[str]:
        self.model_configs = self.scan_models()
        return sorted(self.model_configs.keys())

    def create_config_json(self, model_config: Dict[str, Union[str, List[str]]]) -> None:
        config_path = self.get_config_path(model_config)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(model_config, f, ensure_ascii=False, indent=4)

    def add_config(self, original_repo: str, mlx_repo: str, model_name: Optional[str] = None, quantize: str = "None", default_language: str = "multi", system_prompt: Optional[str] = None,
            multimodal_ability: Optional[List[str]] = None, ) -> None:
        self._validate_repo_format(original_repo, "original_repo")
        self._validate_repo_format(mlx_repo, "mlx_repo")
        self._validate_quantize(quantize)

        if default_language not in self.VALID_LANGUAGES:
            raise ValueError(f"default_language must be one of {self.VALID_LANGUAGES}.")

        self._validate_multimodal_ability(multimodal_ability)

        system_prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        final_model_name = (model_name.strip() if model_name and model_name.strip() else self._extract_repo_name(mlx_repo))

        model_config = {"original_repo": original_repo.strip(), "mlx_repo": mlx_repo.strip(), "model_name": final_model_name, "quantize": None if quantize == "None" else quantize, "default_language": default_language,
            "system_prompt": system_prompt, "multimodal_ability": [] if multimodal_ability == ["None"] else (multimodal_ability or [])}

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name

        self.create_config_json(model_config)
        self.model_configs[display_name] = model_config

    def add_api_config(self, model_name: str, api_key: str, nick_name: Optional[str], base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        if not model_name or not api_key:
            raise ValueError("model_name and api_key are required.")

        model_config = {"model_name": model_name, "api_key": api_key, "base_url": base_url, "nick_name": nick_name, "system_prompt": system_prompt, "type": ModelType.OPENAI_API.value}

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name

        self.create_config_json(model_config)
        self.model_configs[display_name] = model_config

    def _process_config_file(self, config_file: Path) -> Optional[tuple]:
        try:
            with config_file.open("r", encoding="utf-8") as f:
                model_config = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {config_file}: {e}")
            return None

        if model_config.get("type") == ModelType.OPENAI_API.value:
            return self._process_api_config(model_config)
        else:
            return self._process_local_config(model_config)

    def _process_api_config(self, model_config: Dict) -> Optional[tuple]:
        api_key = model_config.get("api_key")
        if not api_key or not api_key.strip():
            return None

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name
        return display_name, model_config

    def _process_local_config(self, model_config: Dict) -> Optional[tuple]:
        model_name = model_config.get("model_name")
        default_language = model_config.get("default_language")

        if not all([model_name, default_language]):
            return None

        if "quantize" not in model_config or not model_config["quantize"]:
            model_config["quantize"] = "None"

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name
        return display_name, model_config

    def scan_models(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        model_configs = {}

        for config_file in self.configs_dir.glob(f"*{self.CONFIG_EXTENSION}"):
            if not config_file.is_file():
                continue

            result = self._process_config_file(config_file)
            if result:
                display_name, processed_config = result
                model_configs[display_name] = processed_config
            else:
                logging.info(f"Skipping incomplete config: {config_file}")

        return model_configs

    def load_model(self, model_name: str) -> None:
        if self.model:
            self.close_model()

        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise RuntimeError(f"Model '{model_name}' not found.")

        try:
            if model_config.get("type") == ModelType.OPENAI_API.value:
                self._load_openai_model(model_config)
            else:
                self._load_local_model(model_config)

            self.model_config = model_config
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def _load_openai_model(self, model_config: Dict) -> None:
        self.model = OpenAIModel(model_config.get("api_key"), model_config.get("model_name"), model_config.get("base_url"))

    def _load_local_model(self, model_config: Dict) -> None:
        local_model_path = self.get_model_path(model_config)

        if not local_model_path.exists():
            self._download_model(model_config, local_model_path)

        multimodal_ability = model_config.get("multimodal_ability", [])
        is_vision_model = multimodal_ability and "vision" in multimodal_ability

        ModelClass = VisionModel if is_vision_model else Model
        self.model = ModelClass(str(local_model_path))

    def _download_model(self, model_config: Dict, local_model_path: Path) -> None:
        mlx_repo = model_config.get("mlx_repo")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=mlx_repo, local_dir=str(local_model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to download model from '{mlx_repo}': {e}")

    def close_model(self) -> None:
        if self.model:
            self.model.close()
            self.model = None
            self.model_config = None
            gc.collect()
            mlx.core.clear_cache()

    def get_loaded_model(self) -> Optional[Model]:
        return weakref.ref(self.model)() if self.model else None

    def get_loaded_model_config(self) -> Optional[Dict[str, Union[str, List[str]]]]:
        return self.model_config

    def get_system_prompt(self, default: bool = False) -> Optional[str]:
        if not self.model_config:
            return None

        if not default and "custom_system_prompt" in self.model_config:
            return self.model_config.get("custom_system_prompt")

        return self.model_config.get("system_prompt")

    def set_custom_prompt(self, custom_system_prompt: str) -> None:
        if self.model_config:
            self.model_config["custom_system_prompt"] = custom_system_prompt
