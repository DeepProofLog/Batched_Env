"""
KGE-Initialized Embeddings Module.

Initializes policy embeddings from pre-trained KGE model.
Provides better starting point for RL training, faster convergence,
and improved generalization.

Purpose: Transfer learned entity/relation representations from KGE to policy.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


def initialize_embeddings_from_kge(
    embedder: nn.Module,
    kge_inference: Any,
    idx2const: Dict[int, str],
    idx2pred: Dict[int, str],
    const2idx: Dict[str, int],
    pred2idx: Dict[str, int],
    verbose: bool = True,
) -> int:
    """Initialize embedder embeddings from KGE model.

    Copies entity and relation embeddings from KGE model to the
    policy's EmbedderLearnable module.

    Args:
        embedder: EmbedderLearnable module to initialize.
        kge_inference: KGE inference engine with loaded model.
        idx2const: Index to constant string mapping.
        idx2pred: Index to predicate string mapping.
        const2idx: Constant string to index mapping (for KGE).
        pred2idx: Predicate string to index mapping (for KGE).
        verbose: Print initialization statistics.

    Returns:
        Number of embeddings successfully initialized.
    """
    # Check if embedder has learnable embeddings
    if not hasattr(embedder, 'constant_embedder') or not hasattr(embedder, 'predicate_embedder'):
        if verbose:
            print("[KGEInit] Embedder does not have learnable embeddings, skipping")
        return 0

    # Get KGE model embeddings
    kge_model = kge_inference.model
    if kge_model is None:
        if verbose:
            print("[KGEInit] KGE model not loaded, skipping")
        return 0

    # Get embedding tensors from KGE model
    kge_entity_emb = _get_kge_entity_embeddings(kge_model)
    kge_relation_emb = _get_kge_relation_embeddings(kge_model)

    if kge_entity_emb is None or kge_relation_emb is None:
        if verbose:
            print("[KGEInit] Could not extract KGE embeddings, skipping")
        return 0

    # Get KGE vocabulary mappings
    kge_entity2id = getattr(kge_inference, 'entity2id', {})
    kge_relation2id = getattr(kge_inference, 'relation2id', {})

    # Initialize constant embeddings
    const_initialized = _copy_embeddings(
        source_emb=kge_entity_emb,
        target_module=embedder.constant_embedder.embedder,
        source_vocab=kge_entity2id,
        target_idx2str=idx2const,
        name="constant",
        verbose=verbose,
    )

    # Initialize predicate embeddings
    pred_initialized = _copy_embeddings(
        source_emb=kge_relation_emb,
        target_module=embedder.predicate_embedder.embedder,
        source_vocab=kge_relation2id,
        target_idx2str=idx2pred,
        name="predicate",
        verbose=verbose,
    )

    total = const_initialized + pred_initialized
    if verbose:
        print(f"[KGEInit] Initialized {total} embeddings ({const_initialized} constants, {pred_initialized} predicates)")

    return total


def _get_kge_entity_embeddings(kge_model: nn.Module) -> Optional[Tensor]:
    """Extract entity embeddings from KGE model.

    Supports multiple KGE architectures (RotatE, ComplEx, TransE, etc.).

    Args:
        kge_model: KGE model module.

    Returns:
        Entity embedding tensor or None if not found.
    """
    # Try common attribute names
    for attr in ['entity_embedding', 'ent_embedding', 'entity_emb', 'ent_emb']:
        if hasattr(kge_model, attr):
            emb = getattr(kge_model, attr)
            if isinstance(emb, nn.Embedding):
                return emb.weight.data
            elif isinstance(emb, nn.Parameter):
                return emb.data
            elif isinstance(emb, Tensor):
                return emb

    # Try nested attributes
    if hasattr(kge_model, 'embeddings'):
        embs = kge_model.embeddings
        if hasattr(embs, 'entity_embedding'):
            return embs.entity_embedding.weight.data

    return None


def _get_kge_relation_embeddings(kge_model: nn.Module) -> Optional[Tensor]:
    """Extract relation embeddings from KGE model.

    Args:
        kge_model: KGE model module.

    Returns:
        Relation embedding tensor or None if not found.
    """
    # Try common attribute names
    for attr in ['relation_embedding', 'rel_embedding', 'relation_emb', 'rel_emb']:
        if hasattr(kge_model, attr):
            emb = getattr(kge_model, attr)
            if isinstance(emb, nn.Embedding):
                return emb.weight.data
            elif isinstance(emb, nn.Parameter):
                return emb.data
            elif isinstance(emb, Tensor):
                return emb

    # Try nested attributes
    if hasattr(kge_model, 'embeddings'):
        embs = kge_model.embeddings
        if hasattr(embs, 'relation_embedding'):
            return embs.relation_embedding.weight.data

    return None


def _copy_embeddings(
    source_emb: Tensor,
    target_module: nn.Embedding,
    source_vocab: Dict[str, int],
    target_idx2str: Dict[int, str],
    name: str,
    verbose: bool = True,
) -> int:
    """Copy embeddings from source to target for matching vocabulary items.

    Args:
        source_emb: [V_src, D_src] source embedding tensor.
        target_module: Target nn.Embedding module.
        source_vocab: Source vocabulary (str -> idx).
        target_idx2str: Target index to string mapping.
        name: Name for logging.
        verbose: Print statistics.

    Returns:
        Number of embeddings copied.
    """
    copied = 0
    target_dim = target_module.embedding_dim
    source_dim = source_emb.shape[1]

    # Handle dimension mismatch
    if source_dim != target_dim:
        if verbose:
            print(f"[KGEInit] {name} dimension mismatch: KGE={source_dim}, policy={target_dim}")
            if source_dim > target_dim:
                print(f"[KGEInit] Truncating KGE embeddings to {target_dim}")
            else:
                print(f"[KGEInit] Zero-padding KGE embeddings to {target_dim}")

    # Reverse source vocab for lookup
    source_id2str = {v: k for k, v in source_vocab.items()}

    with torch.no_grad():
        for target_idx, target_str in target_idx2str.items():
            # Skip padding index
            if target_idx == 0:
                continue

            # Normalize string for lookup
            lookup_str = target_str.lower().strip()

            # Try to find in source vocab
            source_idx = source_vocab.get(lookup_str)
            if source_idx is None:
                # Try without normalization
                source_idx = source_vocab.get(target_str)

            if source_idx is not None and source_idx < source_emb.shape[0]:
                # Get source embedding
                src_vec = source_emb[source_idx]

                # Handle dimension mismatch
                if source_dim > target_dim:
                    src_vec = src_vec[:target_dim]
                elif source_dim < target_dim:
                    padding = torch.zeros(target_dim - source_dim, device=src_vec.device)
                    src_vec = torch.cat([src_vec, padding])

                # Copy to target
                if target_idx < target_module.weight.shape[0]:
                    target_module.weight[target_idx] = src_vec
                    copied += 1

    if verbose:
        total = len(target_idx2str) - 1  # Exclude padding
        coverage = 100 * copied / max(total, 1)
        print(f"[KGEInit] {name}: {copied}/{total} ({coverage:.1f}%) embeddings initialized")

    return copied


def apply_kge_init(
    config: Any,
    embedder: nn.Module,
    kge_inference: Any,
    data_handler: Any,
) -> bool:
    """Apply KGE initialization if enabled in config.

    Args:
        config: TrainConfig with KGE init settings.
        embedder: EmbedderLearnable module.
        kge_inference: KGE inference engine.
        data_handler: DataHandler with vocabulary mappings.

    Returns:
        True if initialization was applied, False otherwise.
    """
    if not getattr(config, 'kge_init_embeddings', False):
        return False

    if kge_inference is None:
        print("[KGEInit] KGE inference not available, skipping initialization")
        return False

    verbose = getattr(config, 'verbose', True)

    # Get vocabulary mappings from data handler
    idx2const = getattr(data_handler, 'idx2const', {})
    idx2pred = getattr(data_handler, 'idx2pred', {})
    const2idx = getattr(data_handler, 'const2idx', {})
    pred2idx = getattr(data_handler, 'pred2idx', {})

    # Initialize embeddings
    n_init = initialize_embeddings_from_kge(
        embedder=embedder,
        kge_inference=kge_inference,
        idx2const=idx2const,
        idx2pred=idx2pred,
        const2idx=const2idx,
        pred2idx=pred2idx,
        verbose=verbose,
    )

    return n_init > 0
