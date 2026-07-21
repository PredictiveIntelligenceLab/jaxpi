from importlib import import_module


__all__ = ["archs", "models", "samplers", "utils"]
_SUBMODULES = frozenset(__all__)


def __getattr__(name):
    """Lazily import public submodules while preserving the package API."""
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)

__version__ = "0.0.1"
__author__ = "Sifan Wang"
