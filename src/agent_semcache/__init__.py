"""agent-semcache: a local, lexical prompt cache for LLM agents (standard library only)."""
from .cache import Hit, SemCache, cached
from .similarity import MEASURES, normalize, similarity

__all__ = ["SemCache", "Hit", "cached", "similarity", "normalize", "MEASURES"]
__version__ = "0.1.0"
