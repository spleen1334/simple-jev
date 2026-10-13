"""A small Ollama classification experiment, without trained calibration."""

from .classifier import classify, download_model

__all__ = ["classify", "download_model"]
