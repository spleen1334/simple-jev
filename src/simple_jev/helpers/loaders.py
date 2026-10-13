"""Cache-first model loading and quieter startup for the learning examples."""

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from llama_cpp import Llama, llama_cpp
from llama_cpp._logger import llama_log_callback as _default_model_log

from simple_jev import download_model, load_model
from simple_jev.classifier import DEFAULT_FILENAME, DEFAULT_REPO


@llama_cpp.llama_log_callback
def _model_log(_level, message, _user_data):
    text = message.decode(errors="replace")
    # A deliberately small context is fine for these short examples. Delegate
    # other messages to the pinned library's normal verbosity/error handling.
    if "n_ctx_seq" in text and "full capacity of the model" in text:
        return
    _default_model_log(_level, message, _user_data)


def load_example_model() -> Llama:
    """Prefer the downloaded file, without a network metadata check on every run."""
    # Keep the callback at module scope: C holds a pointer to this Python object.
    llama_cpp.llama_log_set(_model_log, None)
    try:
        path = hf_hub_download(
            repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, local_files_only=True
        )
    except LocalEntryNotFoundError:
        # A fresh setup still shows real download progress and download errors.
        path = download_model()
    return load_model(model_path=path)
