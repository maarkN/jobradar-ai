"""FastAPI dependency providers. Overridable in tests via app.dependency_overrides."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from jobradar.api.runs import RunRegistry
from jobradar.cache import ExtractionCache
from jobradar.config import Settings, load_settings
from jobradar.embeddings import Embedder, OpenAIEmbedder
from jobradar.extract import Extractor
from jobradar.graph import SupportsExtract
from jobradar.index import QdrantJobIndex
from jobradar.llm import get_provider
from jobradar.sources import FetchFn, Source, make_fetch_fn


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_index() -> QdrantJobIndex:
    settings = get_settings()
    return QdrantJobIndex(settings, settings.embedding_dim)


@lru_cache
def get_registry() -> RunRegistry:
    return RunRegistry()


def get_embedder() -> Embedder:
    return OpenAIEmbedder(get_settings())


def get_extractor() -> SupportsExtract:
    settings = get_settings()
    return Extractor(get_provider(settings), ExtractionCache(settings.cache_dir), settings)


def get_fetch_factory() -> Callable[[list[Source]], FetchFn]:
    settings = get_settings()
    return lambda sources: make_fetch_fn(sources, settings)
