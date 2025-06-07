import enum
import gc
import json
import logging
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List, Optional, Generator, Any

import mlx
import mlx_vlm
from mlx_lm import load, generate, stream_generate, sample_utils
from mlx_lm.generate import GenerationResponse
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


class BaseLocalModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.load()

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def format_chat_prompt(self, message: Dict, history: List[Dict], **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[str, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def generate_completion(self, prompt: str, stream: bool = False, **kwargs: Any) -> Union[str, Generator[str, None, None]]:
        return self.perform_generation(prompt, stream, **kwargs)

    def generate_response(self, message: str, history: List[Dict], stream: bool = False, **kwargs: Any) -> Union[str, Generator[str, None, None]]:
        user_message = Message(MessageRole.USER, message).to_dict()
        formatted_prompt = self.format_chat_prompt(
            message=user_message,
            history=history,
            **kwargs
        )
        return self.perform_generation(formatted_prompt, stream, **kwargs)


class TextModel(BaseLocalModel):
    def __init__(self, model_path: str):
        self.model = None
        self.tokenizer = None
        super().__init__(model_path)

    def load(self) -> None:
        try:
            self.model, self.tokenizer = load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load text model {self.model_path}: {e}")

    def format_chat_prompt(self, message: Dict, history: List[Dict], **kwargs) -> str:
        conversation = history + [message]
        return self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )

    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[GenerationResponse, None, None]]:
        gen_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 20),
            "top_p": kwargs.get("top_p", 0.9),
            "min_p": kwargs.get("min_p", 0.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        }

        sampler = sample_utils.make_sampler(
            temp=gen_params["temperature"],
            top_k=gen_params["top_k"],
            top_p=gen_params["top_p"],
            min_p=gen_params["min_p"]
        )
        logits_processors = sample_utils.make_logits_processors(
            repetition_penalty=gen_params["repetition_penalty"]
        )

        gen_args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "sampler": sampler,
            "logits_processors": logits_processors,
            "max_tokens": gen_params["max_tokens"],
        }

        if stream:
            return stream_generate(**gen_args)
        else:
            return generate(**gen_args)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        mlx.core.clear_cache()


class VisionModel(BaseLocalModel):
    def __init__(self, model_path: str):
        self.config = None
        self.model = None
        self.processor = None
        self.image_processor = None
        super().__init__(model_path)

    def load(self) -> None:
        try:
            self.config = mlx_vlm.utils.load_config(self.model_path)
            self.model, self.processor = mlx_vlm.load(self.model_path, processor_config={"trust_remote_code": True})
            self.image_processor = mlx_vlm.utils.load_image_processor(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load vision model {self.model_path}: {e}")

    def format_chat_prompt(self, message: Dict, history: List[Dict], **kwargs) -> str:
        conversation = history + [message]
        images = kwargs.get("images", [])
        return mlx_vlm.prompt_utils.apply_chat_template(
            processor=self.processor,
            config=self.config,
            prompt=conversation,
            add_generation_prompt=True,
            num_images=len(images)
        )

    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[str, None, None]]:
        gen_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 512),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "image": kwargs.get("images", []),
        }

        gen_args = {
            "model": self.model,
            "processor": self.processor,
            "image_processor": self.image_processor,
            "prompt": prompt,
            **gen_params,
        }

        if stream:
            return mlx_vlm.utils.stream_generate(**gen_args)
        else:
            return mlx_vlm.utils.generate(**gen_args)

    def close(self) -> None:
        del self.model
        del self.processor
        del self.image_processor
        del self.config
        gc.collect()
        mlx.core.clear_cache()


class OpenAIModel:
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            self.client.models.retrieve(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to or find model {self.model_name}: {e}")

    def generate_response(self, messages: List[Dict], stream: bool = False, **kwargs: Any) -> Any:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            **kwargs
        )

    def close(self):
        del self.client
        del self.api_key


class ModelType(enum.Enum):
    LOCAL = "local"
    OPENAI_API = "openai_api"


class MemoryUsageLevel(enum.Enum):
    HIGH = "high"
    STRICT = "strict"
    NONE = "none"


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

        self.model: Optional[BaseLocalModel, OpenAIModel] = None
        self.model_config: Optional[Dict[str, Union[str, List[str]]]] = None
        self.model_configs: Dict[str, Dict[str, Union[str, List[str]]]] = self.scan_models()

        self.memory_usage_level: Optional[MemoryUsageLevel] = MemoryUsageLevel.STRICT
        self.set_memory_usage_level(self.memory_usage_level)

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

    def add_config(self,
                   original_repo: str,
                   mlx_repo: str,
                   model_name: Optional[str] = None,
                   quantize: str = "None",
                   default_language: str = "multi",
                   system_prompt: Optional[str] = None,
                   multimodal_ability: Optional[List[str]] = None) -> None:
        self._validate_repo_format(original_repo, "original_repo")
        self._validate_repo_format(mlx_repo, "mlx_repo")
        self._validate_quantize(quantize)

        if default_language not in self.VALID_LANGUAGES:
            raise ValueError(f"default_language must be one of {self.VALID_LANGUAGES}.")

        self._validate_multimodal_ability(multimodal_ability)

        system_prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        final_model_name = (model_name.strip() if model_name and model_name.strip() else self._extract_repo_name(mlx_repo))

        model_config = {
            "original_repo": original_repo.strip(),
            "mlx_repo": mlx_repo.strip(),
            "model_name": final_model_name,
            "quantize": None if quantize == "None" else quantize,
            "default_language": default_language,
            "system_prompt": system_prompt,
            "multimodal_ability": [] if multimodal_ability == ["None"] else (multimodal_ability or [])
        }

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name

        self.create_config_json(model_config)
        self.model_configs[display_name] = model_config

    def add_api_config(self, model_name: str, api_key: str, nick_name: Optional[str], base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        if not model_name or not api_key:
            raise ValueError("model_name and api_key are required.")

        model_config = {
            "model_name": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "nick_name": nick_name,
            "system_prompt": system_prompt,
            "type": ModelType.OPENAI_API.value
        }

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

        ModelClass = VisionModel if is_vision_model else TextModel
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

    def get_loaded_model(self) -> Union[BaseLocalModel, OpenAIModel, None]:
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

    def get_active_memory(self) -> int:
        return mlx.core.get_active_memory()

    def get_cache_memory(self) -> int:
        return mlx.core.get_cache_memory()

    def get_system_memory_usage(self) -> int:
        logging.info(self.get_active_memory() + self.get_cache_memory())
        return self.get_active_memory() + self.get_cache_memory()

    def get_device_info(self) -> Dict[str, Union[str, int]]:
        return mlx.core.metal.device_info()

    def get_memory_usage_level(self) -> MemoryUsageLevel:
        return self.memory_usage_level

    def set_memory_usage_level(self, memory_usage_level: MemoryUsageLevel) -> None:
        self.memory_usage_level = memory_usage_level
        self.set_memory_usage_policy(self.memory_usage_level)

    def set_memory_usage_policy(self, memory_usage_level: MemoryUsageLevel) -> None:
        device_info = self.get_device_info()
        max_recommended_working_set_size = device_info.get("max_recommended_working_set_size")
        memory_size = device_info.get("memory_size")
        if memory_usage_level == MemoryUsageLevel.STRICT:
            mlx.core.set_wired_limit(max_recommended_working_set_size)
            mlx.core.set_memory_limit(memory_size - max_recommended_working_set_size)
            mlx.core.set_cache_limit(0)
        elif memory_usage_level == MemoryUsageLevel.HIGH:
            mlx.core.set_wired_limit(round(max_recommended_working_set_size / 2))
            mlx.core.set_memory_limit(memory_size - round(max_recommended_working_set_size / 2))
            mlx.core.set_cache_limit(memory_size - round(max_recommended_working_set_size / 2))
        else:
            mlx.core.set_wired_limit(0)
            mlx.core.set_memory_limit(memory_size)
            mlx.core.set_cache_limit(memory_size)
