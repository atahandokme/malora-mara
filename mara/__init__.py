"""MaRA: context-level retrieval adapter over segment-pooled backbone states."""

from .mara import (  # noqa: F401
    EvidenceRouter,
    LastTokenRouter,
    MLPRouter,
    TransformerRouter,
    TokenMambaRouter,
    PoolOnlyRouter,
    RNNRouter,
    derive_p_chunk_spans,
)
