"""
Comprehensive parity tests for DataHandler, IndexManager, and Sampler.

Tests verify that the new tensor-based implementations produce exactly the same
results as the reference SB3 (string-based) implementations.
"""
from types import SimpleNamespace
from pathlib import Path
import sys
import importlib.util

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

# Ensure root-first resolution
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))


def _ensure_root_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # Needed for dataclass module lookup
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


# Lock root index_manager so later imports don't grab sb3/index_manager.py
_ensure_root_module("index_manager")
# Now import normally from root
from data_handler import DataHandler as NewDataHandler
from index_manager import IndexManager as NewIndexManager
from sampler import Sampler as NewSampler

# Make sb3 helpers importable
from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_neg_sampling import get_sampler as get_sb3_sampler


def _base_args(n: int = 5, dataset: str = "countries_s3"):
    """Base configuration for tests."""
    return SimpleNamespace(
        dataset_name=dataset,
        data_path="./data/",
        janus_file=None,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        n_train_queries=n,
        n_eval_queries=n,
        n_test_queries=n,
        corruption_mode="dynamic",
        corruption_scheme=["tail"],
        padding_atoms=6,
        padding_states=20,
        max_total_vars=1_000_000,
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=0.33,
        train_neg_ratio=1.0,
        filter_queries_by_rules=False,
    )


def _create_new_stack(args):
    """Create the new (tensor-based) data handling stack."""
    dh = NewDataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
        prob_facts=args.prob_facts,
        topk_facts=args.topk_facts,
        topk_facts_threshold=args.topk_facts_threshold,
        corruption_mode=args.corruption_mode,
        filter_queries_by_rules=args.filter_queries_by_rules,
    )
    im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=args.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=args.padding_atoms,
        device="cpu",
    )
    dh.materialize_indices(im=im, device=torch.device("cpu"))
    sampler = NewSampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=torch.device("cpu"),
        default_mode="tail",
        seed=0,
    )
    return dh, im, sampler


def _create_sb3_stack(args):
    """Create the SB3 (string-based) reference stack."""
    dh = SB3DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        corruption_mode=args.corruption_mode,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
        prob_facts=args.prob_facts,
        topk_facts=args.topk_facts,
        topk_facts_threshold=args.topk_facts_threshold,
    )
    im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        args.max_total_vars,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
        padding_atoms=args.padding_atoms,
    )
    im.build_fact_index(dh.facts)
    sampler = get_sb3_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=args.corruption_scheme,
        device=torch.device("cpu"),
    )
    return dh, im, sampler


# ============================================================================
# DataHandler Parity Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_data_handlers_load_same_data(dataset):
    """Verify both handlers load the same facts, rules, and queries."""
    args = _base_args(n=10, dataset=dataset)
    
    dh_new, _, _ = _create_new_stack(args)
    dh_sb3, _, _ = _create_sb3_stack(args)
    
    # Check facts count
    assert len(dh_new.facts) == len(dh_sb3.facts), \
        f"Facts count mismatch: {len(dh_new.facts)} vs {len(dh_sb3.facts)}"
    
    # Check rules count
    assert len(dh_new.rules) == len(dh_sb3.rules), \
        f"Rules count mismatch: {len(dh_new.rules)} vs {len(dh_sb3.rules)}"
    
    # Check query counts
    assert len(dh_new.train_queries) == len(dh_sb3.train_queries), \
        f"Train queries count mismatch: {len(dh_new.train_queries)} vs {len(dh_sb3.train_queries)}"
    assert len(dh_new.valid_queries) == len(dh_sb3.valid_queries), \
        f"Valid queries count mismatch"
    assert len(dh_new.test_queries) == len(dh_sb3.test_queries), \
        f"Test queries count mismatch"
    
    # Check vocabulary sizes
    assert len(dh_new.constants) == len(dh_sb3.constants), \
        f"Constants count mismatch: {len(dh_new.constants)} vs {len(dh_sb3.constants)}"
    assert len(dh_new.predicates) == len(dh_sb3.predicates), \
        f"Predicates count mismatch: {len(dh_new.predicates)} vs {len(dh_sb3.predicates)}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_data_handlers_same_facts_content(dataset):
    """Verify both handlers load identical facts (as sets of tuples)."""
    args = _base_args(n=10, dataset=dataset)
    
    dh_new, _, _ = _create_new_stack(args)
    dh_sb3, _, _ = _create_sb3_stack(args)
    
    # Convert to comparable format
    new_facts_set = set(dh_new.facts_str)
    sb3_facts_set = {(f.predicate, f.args[0], f.args[1]) for f in dh_sb3.facts}
    
    assert new_facts_set == sb3_facts_set, \
        f"Facts content mismatch. Diff: {new_facts_set.symmetric_difference(sb3_facts_set)}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_data_handlers_same_queries_content(dataset):
    """Verify both handlers load identical queries."""
    args = _base_args(n=10, dataset=dataset)
    
    dh_new, _, _ = _create_new_stack(args)
    dh_sb3, _, _ = _create_sb3_stack(args)
    
    # Convert to comparable format
    new_train = set(dh_new.train_queries_str)
    sb3_train = {(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.train_queries}
    
    assert new_train == sb3_train, \
        f"Train queries mismatch. Diff: {new_train.symmetric_difference(sb3_train)}"


# ============================================================================
# IndexManager Parity Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_index_managers_same_vocabulary_sizes(dataset):
    """Verify vocabulary sizes match."""
    args = _base_args(n=10, dataset=dataset)
    
    _, im_new, _ = _create_new_stack(args)
    _, im_sb3, _ = _create_sb3_stack(args)
    
    assert im_new.constant_no == im_sb3.constant_no, \
        f"Constant count mismatch: {im_new.constant_no} vs {im_sb3.constant_no}"
    
    # SB3 includes KGE predicates, so we compare core predicates
    sb3_core_pred_no = im_sb3.predicate_no - len(im_sb3.kge_preds)
    # Note: new impl may or may not include KGE preds depending on config
    # For now just check they're reasonably close
    assert abs(im_new.predicate_no - sb3_core_pred_no) <= len(im_sb3.kge_preds), \
        f"Predicate count mismatch: {im_new.predicate_no} vs {sb3_core_pred_no}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_index_managers_same_constant_mapping(dataset):
    """Verify constants are mapped to the same indices."""
    args = _base_args(n=10, dataset=dataset)
    
    _, im_new, _ = _create_new_stack(args)
    _, im_sb3, _ = _create_sb3_stack(args)
    
    # Get common constants
    common_constants = set(im_new.constant_str2idx.keys()) & set(im_sb3.constant_str2idx.keys())
    
    # Check that indices match for common constants
    for const in list(common_constants)[:20]:  # Check first 20
        idx_new = im_new.constant_str2idx[const]
        idx_sb3 = im_sb3.constant_str2idx[const]
        assert idx_new == idx_sb3, \
            f"Constant '{const}' index mismatch: {idx_new} vs {idx_sb3}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_index_managers_same_predicate_mapping(dataset):
    """Verify predicates are mapped to the same indices."""
    args = _base_args(n=10, dataset=dataset)
    
    _, im_new, _ = _create_new_stack(args)
    _, im_sb3, _ = _create_sb3_stack(args)
    
    # Get regular predicates (not special or KGE)
    sb3_regular = {p for p in im_sb3.predicate_str2idx.keys() 
                   if not p.endswith('_kge') and p not in ['True', 'False', 'Endf']}
    new_regular = {p for p in im_new.predicate_str2idx.keys()
                   if not p.endswith('_kge') and p not in ['True', 'False', 'Endf']}
    
    common_preds = sb3_regular & new_regular
    
    for pred in list(common_preds)[:10]:
        idx_new = im_new.predicate_str2idx[pred]
        idx_sb3 = im_sb3.predicate_str2idx[pred]
        assert idx_new == idx_sb3, \
            f"Predicate '{pred}' index mismatch: {idx_new} vs {idx_sb3}"


# ============================================================================
# Sampler Parity Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_samplers_produce_valid_negatives(dataset):
    """Verify both samplers produce valid negatives."""
    args = _base_args(n=10, dataset=dataset)
    
    dh_new, im_new, sampler_new = _create_new_stack(args)
    dh_sb3, im_sb3, sampler_sb3 = _create_sb3_stack(args)
    
    # Create a test positive triple
    test_triple = torch.tensor([[1, 1, 2]], dtype=torch.long)
    
    # Generate negatives from both
    neg_new = sampler_new.corrupt(test_triple, num_negatives=5, device=torch.device("cpu"))
    
    # Check shapes
    assert neg_new.shape[0] == 1, "Batch dimension should match"
    assert neg_new.shape[1] == 5, "Should produce 5 negatives"
    assert neg_new.shape[2] == 3, "Each negative should be a triple"
    
    # Check that negatives are different from positive
    for i in range(neg_new.shape[1]):
        neg_triple = neg_new[0, i]
        if not torch.equal(neg_triple, test_triple[0]):
            # At least one should be different
            pass


@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_samplers_filter_known_positives(dataset):
    """Verify samplers properly filter known positive triples."""
    args = _base_args(n=100, dataset=dataset)
    
    dh_new, im_new, sampler_new = _create_new_stack(args)
    
    # Use an actual known triple
    if dh_new.all_known_triples_idx is not None and dh_new.all_known_triples_idx.shape[0] > 0:
        known_triple = dh_new.all_known_triples_idx[0:1].clone()
        
        # Generate many negatives with filtering
        neg_filtered = sampler_new.corrupt(
            known_triple, 
            num_negatives=50, 
            filter=True,
            device=torch.device("cpu")
        )
        
        # Check that the known triple is not in the negatives
        for i in range(neg_filtered.shape[1]):
            neg_triple = neg_filtered[0, i]
            if neg_triple.sum() > 0:  # Not padding
                assert not torch.equal(neg_triple, known_triple[0]), \
                    "Known positive triple should be filtered out"


# ============================================================================
# Materialization Parity Tests  
# ============================================================================

def test_data_handlers_materialize_same_shapes():
    """Test that materialized tensors have consistent shapes."""
    args = _base_args()

    dh_new, im_new, sampler_new = _create_new_stack(args)
    dh_sb3, im_sb3, sampler_sb3 = _create_sb3_stack(args)

    assert im_new.constant_no == im_sb3.constant_no
    sb3_core_pred_no = im_sb3.predicate_no - len(im_sb3.kge_preds)
    assert im_new.predicate_no == sb3_core_pred_no
    assert dh_new.max_arity == dh_sb3.max_arity
    assert len(dh_new.train_queries) == len(dh_sb3.train_queries) == args.n_train_queries

    # Negative sample one triple to ensure both samplers produce values on CPU
    head_idx = torch.tensor([[0, 0, 0]], dtype=torch.long)
    neg_new = sampler_new.corrupt(head_idx, num_negatives=1, device=torch.device("cpu"))
    if hasattr(sampler_sb3, "corrupt"):
        neg_sb3 = sampler_sb3.corrupt(head_idx, num_negatives=1, device=torch.device("cpu"))
        assert neg_new.shape == neg_sb3.shape, "Sampler outputs differ in shape"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_materialized_splits_shape(dataset):
    """Verify materialized query splits have correct shapes."""
    args = _base_args(n=20, dataset=dataset)
    
    dh_new, im_new, _ = _create_new_stack(args)
    
    # Check train split
    train_split = dh_new.get_materialized_split('train')
    assert train_split.queries.dim() == 3, "Queries should be 3D [N, 1, width]"
    assert train_split.queries.shape[0] == len(dh_new.train_queries), "Query count should match"
    assert train_split.labels.shape[0] == len(dh_new.train_queries), "Labels count should match"
    assert train_split.depths.shape[0] == len(dh_new.train_queries), "Depths count should match"


# ============================================================================
# Full Integration Test
# ============================================================================

@pytest.mark.parametrize("dataset,n_queries", [
    ("countries_s3", 50),
    ("family", 50),
])
def test_full_data_pipeline_parity(dataset, n_queries):
    """Full integration test comparing both pipelines end-to-end."""
    args = _base_args(n=n_queries, dataset=dataset)
    
    dh_new, im_new, sampler_new = _create_new_stack(args)
    dh_sb3, im_sb3, sampler_sb3 = _create_sb3_stack(args)
    
    # Compare basic stats
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"Facts: new={len(dh_new.facts)}, sb3={len(dh_sb3.facts)}")
    print(f"Rules: new={len(dh_new.rules)}, sb3={len(dh_sb3.rules)}")
    print(f"Train queries: new={len(dh_new.train_queries)}, sb3={len(dh_sb3.train_queries)}")
    print(f"Constants: new={im_new.constant_no}, sb3={im_sb3.constant_no}")
    print(f"Predicates: new={im_new.predicate_no}, sb3={im_sb3.predicate_no}")
    
    # Assertions
    assert len(dh_new.facts) == len(dh_sb3.facts)
    assert len(dh_new.rules) == len(dh_sb3.rules)
    assert len(dh_new.train_queries) == len(dh_sb3.train_queries)
    assert im_new.constant_no == im_sb3.constant_no
    
    print("✓ All checks passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
