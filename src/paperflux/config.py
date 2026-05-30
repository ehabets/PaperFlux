"""
Configuration management for PaperFlux.
Handles loading YAML configuration and converting to Pydantic models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Set

from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

load_dotenv()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str
    model: str


class UIConfig(BaseModel):
    """UI configuration."""
    detail_level: Literal["low", "medium", "high"] = "medium"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"
    verbosity: Literal["low", "medium", "high"] = "medium"
    max_output_tokens: int = 32768
    highlight_colors: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "contributions": [1.0, 1.0, 0.0],  # Yellow
            "limitations": [1.0, 0.6, 0.0],    # Orange
            "claims": [0.2, 0.4, 1.0],         # Blue
            "evidence": [0.0, 0.8, 0.3],       # Green
        }
    )


class ExtractionCategoriesConfig(BaseModel):
    """Configuration for categories to extract quotes for."""
    # Allows arbitrary category names (keys) and their descriptions (values)
    categories: Dict[str, str] = Field(
        default_factory=lambda: {
            "contributions": "Significant advancements, novel methods, or key findings presented in the paper.",
            "limitations": "Identified shortcomings, constraints, or areas where the research or methodology falls short.",
            "claims": "Specific assertions or hypotheses made by the authors that are central to the paper's arguments.",
            "evidence": "Data, experimental results, or logical arguments provided to support the claims made."
        }
    )


class MatchingConfig(BaseModel):
    """Quote matching configuration."""
    min_similarity: float = Field(default=0.88, ge=0.0, le=1.0)
    max_window_tokens: int = Field(default=80, ge=8)


class RagConfig(BaseModel):
    """RAG retrieval and summarization configuration."""

    category_prompt_file: str = "prompts/rag_category_prompt.j2"
    summary_prompt_file: str = "prompts/rag_summary_prompt.j2"
    category_system_prompt_file: str = "prompts/rag_category_system_prompt.txt"
    max_num_results: Optional[int] = Field(default=None, ge=1)
    max_quotes_per_category: int = Field(default=6, ge=1)
    include_search_results: bool = False
    vector_store_expires_after_days: int = Field(default=1, ge=1)


class Config(BaseModel):
    """Main configuration."""
    _config_dir: Optional[Path] = PrivateAttr(default=None)

    openai: OpenAIConfig
    ui: UIConfig
    extraction_categories: ExtractionCategoriesConfig = Field(default_factory=ExtractionCategoriesConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    rag: RagConfig = Field(default_factory=RagConfig)

    @classmethod
    def _missing_highlight_categories(
        cls, cfg_colors: Dict[str, List[float]], categories: Dict[str, str]
    ) -> Set[str]:
        """Return category names that are missing highlight color definitions."""
        defined = set(cfg_colors.keys()) if cfg_colors else set()
        required = set(categories.keys())
        return required - defined

    @model_validator(mode="after")
    def validate_highlight_colors(self) -> "Config":
        """Ensure every category has a highlight color defined."""
        ui_cfg: UIConfig = self.ui
        categories_cfg: ExtractionCategoriesConfig = self.extraction_categories
        missing = self._missing_highlight_categories(ui_cfg.highlight_colors, categories_cfg.categories)
        if missing:
            raise ValueError(
                "Highlight colors missing for categories: " + ", ".join(sorted(missing))
            )
        return self


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in string values."""
    if isinstance(value, str) and value.startswith("ENV:"):
        env_var = value[4:]
        if env_var not in os.environ:
            raise ValueError(
                f"Environment variable '{env_var}' referenced in the configuration is not set. "
                f"Set it (e.g., export {env_var}=...) or replace the config value."
            )
        return os.environ[env_var]
    return value


def _process_config_dict(config_dict: dict) -> dict:
    """Process configuration dictionary to expand environment variables."""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = _process_config_dict(value)
        elif isinstance(value, str):
            config_dict[key] = _expand_env_vars(value)
    return config_dict


def load(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config: Pydantic model of configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Process environment variables
    config_dict = _process_config_dict(config_dict)

    # Convert to Pydantic model
    cfg = Config(**config_dict)
    cfg._config_dir = config_path.parent
    return cfg
