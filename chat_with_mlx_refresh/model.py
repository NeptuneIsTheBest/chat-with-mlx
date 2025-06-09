import enum
import gc
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List, Optional, Generator, Any, Tuple

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
    VALID_QUANTIZE_TYPES = frozenset({"None", "4bit", "8bit", "bf16", "bf32"})
    VALID_LANGUAGES = frozenset({"multi"})
    VALID_MULTIMODAL_ABILITIES = frozenset({"None", "vision"})
    CONFIG_EXTENSION = ".json"

    STRICT_MEMORY_RATIO = 0.7
    STRICT_CACHE_RATIO = 0.3
    HIGH_WIRED_MULTIPLIER = 0.5
    HIGH_MEMORY_RATIO = 0.8
    HIGH_CACHE_RATIO = 0.6

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "models"
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"

        self._setup_directories()

        self.model: Optional[Union['BaseLocalModel', 'OpenAIModel']] = None
        self.model_config: Optional[Dict[str, Union[str, List[str]]]] = None
        self.model_configs: Dict[str, Dict[str, Union[str, List[str]]]] = self.scan_models()

        self.memory_usage_level = MemoryUsageLevel.STRICT
        self.set_memory_usage_level(self.memory_usage_level)

        self.active_generator = []

    def _setup_directories(self) -> None:
        directories = [self.base_dir, self.configs_dir, self.models_dir]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        missing_dirs = [d for d in directories if not d.is_dir()]
        if missing_dirs:
            raise IOError(f"Failed to create directories: {missing_dirs}")

    @staticmethod
    def _validate_repo_format(repo: str, name: str) -> None:
        repo = repo.strip()
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError(f"'{name}' must be in 'owner/repo' format, got: '{repo}'")

    def _validate_quantize(self, quantize: str) -> None:
        if quantize not in self.VALID_QUANTIZE_TYPES:
            raise ValueError(f"quantize must be one of {self.VALID_QUANTIZE_TYPES}, got: '{quantize}'")

    def _validate_multimodal_ability(self, abilities: Optional[List[str]]) -> None:
        if not abilities:
            return

        abilities_set = set(abilities)

        if "None" in abilities_set and len(abilities_set) > 1:
            raise ValueError("'None' cannot exist with other abilities")

        invalid_abilities = abilities_set - self.VALID_MULTIMODAL_ABILITIES
        if invalid_abilities:
            raise ValueError(f"Invalid multimodal abilities: {invalid_abilities}")

    @staticmethod
    def _extract_repo_name(repo: str) -> str:
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

        name_parts = [model_name, f"({default_language},{quantize}"]

        if multimodal_ability and "None" not in multimodal_ability:
            abilities_str = "".join(multimodal_ability)
            name_parts.insert(-1, f",{abilities_str}")

        return "".join(name_parts) + ")"

    def get_config_path(self, model_config: Dict[str, Union[str, List[str]]]) -> Path:
        if model_config.get("type") == ModelType.OPENAI_API.value:
            model_name = model_config.get('model_name')
            if not model_name:
                raise RuntimeError("'model_name' not specified for OpenAI API Model")
            return self.configs_dir / f"{model_name}{self.CONFIG_EXTENSION}"

        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            model_name = model_config.get('model_name', 'unknown')
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'")

        repo_name = self._extract_repo_name(mlx_repo)
        return self.configs_dir / f"{repo_name}{self.CONFIG_EXTENSION}"

    def get_model_path(self, model_config: Dict[str, Union[str, List[str]]]) -> Path:
        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            model_name = model_config.get('model_name', 'unknown')
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'")

        return self.models_dir / self._extract_repo_name(mlx_repo)

    def get_model_list(self) -> List[str]:
        self.model_configs = self.scan_models()
        return sorted(self.model_configs.keys())

    def _save_config_to_file(self, model_config: Dict[str, Union[str, List[str]]]) -> None:
        config_path = self.get_config_path(model_config)
        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=4)
        except OSError as e:
            raise RuntimeError(f"Failed to save config to {config_path}: {e}")

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
            raise ValueError(f"default_language must be one of {self.VALID_LANGUAGES}, got: '{default_language}'")

        self._validate_multimodal_ability(multimodal_ability)

        clean_system_prompt = system_prompt.strip() if system_prompt else None
        final_model_name = (model_name.strip() if model_name else
                            self._extract_repo_name(mlx_repo))

        model_config = {
            "original_repo": original_repo.strip(),
            "mlx_repo": mlx_repo.strip(),
            "model_name": final_model_name,
            "quantize": None if quantize == "None" else quantize,
            "default_language": default_language,
            "system_prompt": clean_system_prompt,
            "multimodal_ability": self._process_multimodal_abilities(multimodal_ability)
        }

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name

        self._save_config_to_file(model_config)
        self.model_configs[display_name] = model_config

    def _process_multimodal_abilities(self, abilities: Optional[List[str]]) -> List[str]:
        if not abilities or abilities == ["None"]:
            return []
        return abilities

    def add_api_config(self,
                       model_name: str,
                       api_key: str,
                       nick_name: Optional[str] = None,
                       base_url: Optional[str] = None,
                       system_prompt: Optional[str] = None) -> None:
        if not model_name or not api_key:
            raise ValueError("model_name and api_key are required")

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

        self._save_config_to_file(model_config)
        self.model_configs[display_name] = model_config

    def _load_config_file(self, config_file: Path) -> Optional[Dict]:
        try:
            with config_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Error loading config from {config_file}: {e}")
            return None

    def _process_config_file(self, config_file: Path) -> Optional[Tuple[str, Dict]]:
        model_config = self._load_config_file(config_file)
        if not model_config:
            return None

        if model_config.get("type") == ModelType.OPENAI_API.value:
            return self._process_api_config(model_config)
        else:
            return self._process_local_config(model_config)

    def _process_api_config(self, model_config: Dict) -> Optional[Tuple[str, Dict]]:
        api_key = model_config.get("api_key")
        if not api_key or not api_key.strip():
            return None

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name
        return display_name, model_config

    def _process_local_config(self, model_config: Dict) -> Optional[Tuple[str, Dict]]:
        required_fields = ["model_name", "default_language"]
        if not all(model_config.get(field) for field in required_fields):
            return None

        if "quantize" not in model_config or not model_config["quantize"]:
            model_config["quantize"] = "None"

        display_name = self._generate_display_name(model_config)
        model_config["display_name"] = display_name
        return display_name, model_config

    def scan_models(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        model_configs = {}

        config_files = list(self.configs_dir.glob(f"*{self.CONFIG_EXTENSION}"))

        for config_file in config_files:
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
            raise RuntimeError(f"Model '{model_name}' not found")

        try:
            if model_config.get("type") == ModelType.OPENAI_API.value:
                self._load_openai_model(model_config)
            else:
                self._load_local_model(model_config)

            self.model_config = model_config
            logging.info(f"Successfully loaded model: {model_name}")

        except Exception as e:
            logging.error(f"Error loading model '{model_name}': {e}")
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def _load_openai_model(self, model_config: Dict) -> None:
        self.model = OpenAIModel(
            api_key=model_config.get("api_key"),
            model_name=model_config.get("model_name"),
            base_url=model_config.get("base_url")
        )

    def _load_local_model(self, model_config: Dict) -> None:
        local_model_path = self.get_model_path(model_config)

        if not local_model_path.exists():
            self._download_model(model_config, local_model_path)

        multimodal_ability = model_config.get("multimodal_ability", [])
        is_vision_model = "vision" in multimodal_ability

        if is_vision_model:
            self.model = VisionModel(str(local_model_path))
        else:
            self.model = TextModel(str(local_model_path))

    def _download_model(self, model_config: Dict, local_model_path: Path) -> None:
        mlx_repo = model_config.get("mlx_repo")
        logging.info(f"Downloading model from {mlx_repo}...")

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=mlx_repo, local_dir=str(local_model_path))
            logging.info(f"Model downloaded successfully to {local_model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from '{mlx_repo}': {e}")

    def close_model(self) -> None:
        self.close_active_generator()

        if self.model:
            try:
                self.model.close()
            except Exception as e:
                logging.error(f"Error closing model: {e}")
            finally:
                self.model = None
                self.model_config = None

            gc.collect()
            mlx.core.clear_cache()

    def get_loaded_model(self) -> Optional[Union['BaseLocalModel', 'OpenAIModel']]:
        return self.model

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
        total_usage = self.get_active_memory() + self.get_cache_memory()
        logging.info(f"System memory usage: {total_usage}")
        return total_usage

    def get_device_info(self) -> Dict[str, Union[str, int]]:
        return mlx.core.metal.device_info()

    def get_memory_usage_level(self) -> MemoryUsageLevel:
        return self.memory_usage_level

    def set_memory_usage_level(self, memory_usage_level: MemoryUsageLevel) -> None:
        self.memory_usage_level = memory_usage_level
        self._apply_memory_policy(memory_usage_level)

    def _apply_memory_policy(self, memory_usage_level: MemoryUsageLevel) -> None:
        device_info = self.get_device_info()
        max_recommended_working_set_size = device_info.get("max_recommended_working_set_size")
        memory_size = device_info.get("memory_size")

        if memory_usage_level == MemoryUsageLevel.STRICT:
            self._set_strict_memory_policy(max_recommended_working_set_size, memory_size)
        elif memory_usage_level == MemoryUsageLevel.HIGH:
            self._set_high_memory_policy(max_recommended_working_set_size, memory_size)
        else:
            self._set_unlimited_memory_policy(memory_size)

    def _set_strict_memory_policy(self, max_recommended_size: int, memory_size: int) -> None:
        wired_limit = max_recommended_size
        available_memory = memory_size - wired_limit

        mlx.core.set_wired_limit(wired_limit)
        mlx.core.set_memory_limit(int(available_memory * self.STRICT_MEMORY_RATIO))
        mlx.core.set_cache_limit(int(available_memory * self.STRICT_CACHE_RATIO))

    def _set_high_memory_policy(self, max_recommended_size: int, memory_size: int) -> None:
        wired_limit = int(max_recommended_size * self.HIGH_WIRED_MULTIPLIER)
        available_memory = memory_size - wired_limit

        mlx.core.set_wired_limit(wired_limit)
        mlx.core.set_memory_limit(int(available_memory * self.HIGH_MEMORY_RATIO))
        mlx.core.set_cache_limit(int(available_memory * self.HIGH_CACHE_RATIO))

    def _set_unlimited_memory_policy(self, memory_size: int) -> None:
        mlx.core.set_wired_limit(0)
        mlx.core.set_memory_limit(memory_size)
        mlx.core.set_cache_limit(memory_size)

    def set_active_generator(self, gen) -> None:
        self.active_generator.append(gen)

    def close_active_generator(self) -> None:
        if self.active_generator:
            try:
                for gen in self.active_generator:
                    gen.close()
                    self.active_generator.remove(gen)

                    gc.collect()
                    mlx.core.clear_cache()
            except Exception as e:
                logging.info(f"Failed to close active generator: {e}")
