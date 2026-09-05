"""Configuration loading and validation utilities."""

from datetime import date
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
REQUIRED_FACTORS = {
    "company_size",
    "industry_fit",
    "lead_source",
    "interaction_recency",
}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the project configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The validated configuration dictionary.

    Raises:
        ValueError: If the file is missing, malformed, or incomplete.
    """
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    required_sections = {"evaluation_date", "batch_size", "qualification", "thresholds"}
    missing_sections = required_sections - config.keys()
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Missing configuration sections: {missing}")

    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")

    try:
        date.fromisoformat(config["evaluation_date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_date must use YYYY-MM-DD format") from exc

    qualification = config["qualification"]
    if not isinstance(qualification, dict):
        raise ValueError("qualification must be a YAML mapping")

    missing_factors = REQUIRED_FACTORS - qualification.keys()
    if missing_factors:
        missing = ", ".join(sorted(missing_factors))
        raise ValueError(f"Missing qualification factors: {missing}")

    if not isinstance(config["thresholds"], list) or not config["thresholds"]:
        raise ValueError("thresholds must be a non-empty list")

    return config
