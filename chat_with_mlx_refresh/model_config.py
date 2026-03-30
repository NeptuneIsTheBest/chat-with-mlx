from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from huggingface_hub import hf_hub_download


@dataclass
class ModelConfig:
    mlx_repo: str
    model_name: str
    quantize: Optional[str] = None
    default_language: str = "multi"
    system_prompt: Optional[str] = None
    multimodal_ability: List[str] = field(default_factory=list)
    display_name: Optional[str] = None
    custom_system_prompt: Optional[str] = field(default=None, repr=False, compare=False)
    config_path: Optional[Path] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        return cls(
            mlx_repo=(payload.get("mlx_repo") or "").strip(),
            model_name=(payload.get("model_name") or "").strip(),
            quantize=payload.get("quantize"),
            default_language=payload.get("default_language") or "multi",
            system_prompt=payload.get("system_prompt"),
            multimodal_ability=list(payload.get("multimodal_ability") or []),
            display_name=payload.get("display_name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mlx_repo": self.mlx_repo,
            "model_name": self.model_name,
            "quantize": self.quantize,
            "default_language": self.default_language,
            "system_prompt": self.system_prompt,
            "multimodal_ability": list(self.multimodal_ability),
            "display_name": self.display_name,
        }

    @staticmethod
    def build_display_name(
        model_name: str,
        default_language: str,
        quantize: Optional[str],
        multimodal_ability: Optional[List[str]] = None,
    ) -> str:
        parts = [default_language, quantize or "None"]
        if multimodal_ability:
            parts.append("+".join(multimodal_ability))
        return f"{model_name}({','.join(parts)})"

    @property
    def resolved_display_name(self) -> str:
        if self.display_name:
            return self.display_name
        return self.build_display_name(
            model_name=self.model_name,
            default_language=self.default_language,
            quantize=self.quantize,
            multimodal_ability=self.multimodal_ability,
        )


class ModelConfigStore:
    VALID_QUANTIZE_TYPES = frozenset(
        {"None", "2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "bf16", "bf32"}
    )
    VALID_LANGUAGES = frozenset({"multi"})
    VALID_MULTIMODAL_ABILITIES = frozenset({"vision", "audio"})
    MULTIMODAL_ABILITY_ORDER = ("vision", "audio")
    VISION_CONFIG_KEYS = frozenset(
        {"vision_config", "image_token_id", "image_token_index", "vision_start_token_id", "vision_end_token_id"}
    )
    AUDIO_CONFIG_KEYS = frozenset(
        {"audio_config", "audio_token_id", "audio_token_index", "audio_start_token_id", "audio_end_token_id"}
    )
    REMOTE_CONFIG_FILENAME = "config.json"
    CONFIG_EXTENSION = ".json"
    SAFE_KEY_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "models"
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"
        self._setup_directories()

    def _setup_directories(self) -> None:
        directories = [self.base_dir, self.configs_dir, self.models_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        missing_dirs = [directory for directory in directories if not directory.is_dir()]
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

    def _validate_language(self, default_language: str) -> None:
        if default_language not in self.VALID_LANGUAGES:
            raise ValueError(
                f"default_language must be one of {self.VALID_LANGUAGES}, got: '{default_language}'"
            )

    def _validate_multimodal_ability(self, abilities: Optional[List[str]]) -> None:
        if not abilities:
            return

        abilities_set = set(abilities)
        invalid_abilities = abilities_set - self.VALID_MULTIMODAL_ABILITIES
        if invalid_abilities:
            raise ValueError(f"Invalid multimodal abilities: {invalid_abilities}")

    @staticmethod
    def _extract_repo_name(repo: str) -> str:
        return repo.strip().split("/")[-1]

    @classmethod
    def _slugify(cls, value: str) -> str:
        normalized = value.strip().replace("/", "__")
        normalized = cls.SAFE_KEY_PATTERN.sub("_", normalized)
        normalized = normalized.strip("._-")
        return normalized or "item"

    @staticmethod
    def _short_hash(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]

    def _build_model_directory_name(self, mlx_repo: str) -> str:
        return self._slugify(mlx_repo)

    def _build_config_filename(self, display_name: str) -> str:
        slug = self._slugify(display_name)
        return f"{slug}-{self._short_hash(display_name)}{self.CONFIG_EXTENSION}"

    def _process_multimodal_abilities(self, abilities: Optional[List[str]]) -> List[str]:
        if not abilities:
            return []

        normalized = []
        abilities_set = set(abilities)
        for ability in self.MULTIMODAL_ABILITY_ORDER:
            if ability in abilities_set:
                normalized.append(ability)
        return normalized

    @classmethod
    def _payload_contains_any_key(cls, payload: Any, keys: Iterable[str]) -> bool:
        if isinstance(payload, dict):
            if any(key in payload for key in keys):
                return True
            return any(cls._payload_contains_any_key(value, keys) for value in payload.values())

        if isinstance(payload, list):
            return any(cls._payload_contains_any_key(item, keys) for item in payload)

        return False

    @classmethod
    def detect_multimodal_abilities_from_payload(cls, payload: Dict[str, Any]) -> List[str]:
        detected = []
        if cls._payload_contains_any_key(payload, cls.VISION_CONFIG_KEYS):
            detected.append("vision")
        if cls._payload_contains_any_key(payload, cls.AUDIO_CONFIG_KEYS):
            detected.append("audio")
        return detected

    @staticmethod
    def format_multimodal_abilities(abilities: Optional[List[str]]) -> str:
        normalized = abilities or []
        if not normalized:
            return "Text only"
        return " + ".join(ability.capitalize() for ability in normalized)

    def _load_detection_payload(self, config_path: Path) -> Dict[str, Any]:
        try:
            with config_path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Config not found: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse config file: {config_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read config file: {config_path}") from exc

    def detect_multimodal_abilities_from_model_path(self, model_path: Path) -> List[str]:
        payload = self._load_detection_payload(model_path / self.REMOTE_CONFIG_FILENAME)
        return self.detect_multimodal_abilities_from_payload(payload)

    def detect_multimodal_abilities_from_repo(self, mlx_repo: str) -> List[str]:
        self._validate_repo_format(mlx_repo, "mlx_repo")
        try:
            config_path = Path(
                hf_hub_download(
                    repo_id=mlx_repo,
                    filename=self.REMOTE_CONFIG_FILENAME,
                    repo_type="model",
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch config for '{mlx_repo}': {exc}") from exc

        payload = self._load_detection_payload(config_path)
        return self.detect_multimodal_abilities_from_payload(payload)

    def detect_multimodal_abilities(self, model_config: ModelConfig) -> List[str]:
        model_path = self.get_model_path(model_config)
        if model_path.exists():
            return self.detect_multimodal_abilities_from_model_path(model_path)
        return self.detect_multimodal_abilities_from_repo(model_config.mlx_repo)

    def _generate_display_name(self, model_config: ModelConfig) -> str:
        return ModelConfig.build_display_name(
            model_name=model_config.model_name,
            default_language=model_config.default_language,
            quantize=model_config.quantize,
            multimodal_ability=model_config.multimodal_ability,
        )

    def get_config_path(self, model_config: ModelConfig) -> Path:
        if model_config.config_path is not None:
            return model_config.config_path
        return self.configs_dir / self._build_config_filename(model_config.resolved_display_name)

    def get_model_path(self, model_config: ModelConfig) -> Path:
        if not model_config.mlx_repo:
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_config.model_name or 'unknown'}'")
        return self.models_dir / self._build_model_directory_name(model_config.mlx_repo)

    def _save_config_to_file(self, model_config: ModelConfig) -> None:
        config_path = self.get_config_path(model_config)
        try:
            with config_path.open("w", encoding="utf-8") as file_obj:
                json.dump(model_config.to_dict(), file_obj, ensure_ascii=False, indent=4)
            model_config.config_path = config_path
        except OSError as exc:
            raise RuntimeError(f"Failed to save config to {config_path}: {exc}") from exc

    def add_config(
        self,
        mlx_repo: str,
        model_name: Optional[str] = None,
        quantize: str = "None",
        default_language: str = "multi",
        system_prompt: Optional[str] = None,
        multimodal_ability_override: Optional[List[str]] = None,
    ) -> ModelConfig:
        self._validate_repo_format(mlx_repo, "mlx_repo")
        self._validate_quantize(quantize)
        self._validate_language(default_language)

        normalized_override = self._process_multimodal_abilities(multimodal_ability_override)
        self._validate_multimodal_ability(normalized_override)

        detected_multimodal_ability = normalized_override
        if multimodal_ability_override is None:
            detected_multimodal_ability = self._process_multimodal_abilities(
                self.detect_multimodal_abilities_from_repo(mlx_repo)
            )

        config = ModelConfig(
            mlx_repo=mlx_repo.strip(),
            model_name=(model_name.strip() if model_name else self._extract_repo_name(mlx_repo)),
            quantize=None if quantize == "None" else quantize,
            default_language=default_language,
            system_prompt=system_prompt.strip() if system_prompt else None,
            multimodal_ability=detected_multimodal_ability,
        )
        config.display_name = self._generate_display_name(config)
        if config.resolved_display_name in self.load_configs():
            raise ValueError(f"Model '{config.resolved_display_name}' already exists")
        self._save_config_to_file(config)
        return config

    def delete_config(self, model_config: ModelConfig, delete_model_files: bool = False) -> None:
        config_path = self.get_config_path(model_config)
        try:
            if config_path.exists():
                config_path.unlink()
                logging.info("Deleted config file: %s", config_path)
        except OSError as exc:
            raise RuntimeError(f"Failed to delete config file {config_path}: {exc}") from exc

        if not delete_model_files:
            return

        model_path = self.get_model_path(model_config)
        try:
            if model_path.exists():
                shutil.rmtree(model_path)
                logging.info("Deleted model files: %s", model_path)
        except OSError as exc:
            logging.warning("Failed to delete model files %s: %s", model_path, exc)

    def _load_config_file(self, config_file: Path) -> Optional[ModelConfig]:
        try:
            with config_file.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        except (json.JSONDecodeError, OSError) as exc:
            logging.error("Error loading config from %s: %s", config_file, exc)
            return None

        model_config = ModelConfig.from_dict(payload)
        model_config.config_path = config_file
        return self._process_config(model_config)

    def _process_config(self, model_config: ModelConfig) -> Optional[ModelConfig]:
        if not model_config.model_name or not model_config.default_language:
            return None

        self._validate_multimodal_ability(model_config.multimodal_ability)
        model_config.multimodal_ability = self._process_multimodal_abilities(model_config.multimodal_ability)
        model_config.display_name = self._generate_display_name(model_config)
        return model_config

    def load_configs(self) -> Dict[str, ModelConfig]:
        model_configs: Dict[str, ModelConfig] = {}
        for config_file in self.configs_dir.glob(f"*{self.CONFIG_EXTENSION}"):
            if not config_file.is_file():
                continue

            model_config = self._load_config_file(config_file)
            if model_config is None:
                logging.info("Skipping incomplete config: %s", config_file)
                continue

            if model_config.resolved_display_name in model_configs:
                logging.error("Duplicate model display name detected: %s", model_config.resolved_display_name)
                continue
            model_configs[model_config.resolved_display_name] = model_config
        return model_configs
