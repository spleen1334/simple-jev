"""Small shared helpers for model loading and terminal presentation."""

from .console import show_result
from .loaders import load_example_model

__all__ = ["load_example_model", "show_result"]
