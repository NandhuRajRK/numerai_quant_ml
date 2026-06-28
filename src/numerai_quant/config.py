"""Configuration loading for the Numerai baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Typed accessors around the YAML configuration."""

    raw: dict[str, Any]
    root_dir: Path

    @property
    def dataset_version(self) -> str:
        """Numerai dataset version, for example ``v5.2``."""
        return str(self.raw["numerai"]["dataset_version"])

    @property
    def random_seed(self) -> int:
        """Project-level random seed."""
        return int(self.raw["project"]["random_seed"])

    @property
    def model_name(self) -> str:
        """Configured model name used for artifact filenames."""
        return str(self.raw["model"]["name"])

    @property
    def target_col(self) -> str:
        """Target column in training and validation data."""
        return str(self.raw["model"]["target_col"])

    @property
    def id_col(self) -> str:
        """Numerai row identifier column."""
        return str(self.raw["model"]["id_col"])

    @property
    def era_col(self) -> str:
        """Numerai era column."""
        return str(self.raw["model"]["era_col"])

    @property
    def prediction_col(self) -> str:
        """Submission prediction column."""
        return str(self.raw["model"]["prediction_col"])

    @property
    def model_params(self) -> dict[str, Any]:
        """LightGBM model parameters."""
        return dict(self.raw["model"]["params"])

    def path(self, key: str) -> Path:
        """Return a configured path resolved relative to the project root."""
        return self.root_dir / str(self.raw["paths"][key])

    def numerai_file(self, key: str) -> str:
        """Return the configured Numerai dataset filename for a key."""
        return str(self.raw["numerai"][key])

    def section(self, key: str) -> dict[str, Any]:
        """Return a top-level config section as a mutable mapping copy."""
        value = self.raw[key]
        if not isinstance(value, dict):
            raise ValueError(f"Expected config section {key!r} to be a mapping.")
        return dict(value)


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upward until ``pyproject.toml`` exists."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def load_config(config_path: Path | str = "configs/baseline.yaml") -> PipelineConfig:
    """Load the YAML configuration file."""
    root_dir = find_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root_dir / path

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    return PipelineConfig(raw=raw, root_dir=root_dir)


def load_config_mapping(raw: dict[str, Any], *, root_dir: Path | None = None) -> PipelineConfig:
    """Build a config object from an already-loaded mapping."""
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping.")
    resolved_root = root_dir or find_project_root()
    return PipelineConfig(raw=raw, root_dir=resolved_root)
