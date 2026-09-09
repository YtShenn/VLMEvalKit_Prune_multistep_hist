"""GUIPruner-reproduction (unofficial reproduction) for Qwen3-VL."""

__all__ = ["Qwen3VLGUIPrunerChat"]


def __getattr__(name):
    # Keep TAR/SSP unit tests independent of model weights and transformers.
    if name == "Qwen3VLGUIPrunerChat":
        from .model import Qwen3VLGUIPrunerChat
        return Qwen3VLGUIPrunerChat
    raise AttributeError(name)
