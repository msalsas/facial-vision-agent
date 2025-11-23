from .agent import HairVisionAgent

__all__ = ["HairVisionAgent"]

try:
    from importlib.metadata import version
    __version__ = version("hair-vision-agent")
except ImportError:
    __version__ = "0.1.0-dev"