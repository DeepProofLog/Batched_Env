"""
DataHandler Parity Tests.

Tests verifying that the tensor-based DataHandler produces EXACTLY the same
results as the reference SB3 (string-based) DataHandler implementation.

This module tests:
- Facts loading and ordering
- Rules parsing and content
- Query loading (train/valid/test)
- Vocabulary (constants and predicates)
- Materialization indices

Usage:
    pytest tests/parity/test_data_handler_parity.py -v
    pytest tests/parity/test_data_handler_parity.py -v -k "countries"
    pytest tests/parity/test_data_handler_parity.py -v -k "family"
    
    # Run with full datasets (no query limits):
    FULL_DATASET=1 pytest tests/parity/test_data_handler_parity.py -v
    
    # Run from command line:
    python tests/parity/test_data_handler_parity.py --dataset countries_s3
"""
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


# ============================================================================
# Path Setup
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))


def _ensure_root_module(name: str):
    """Ensure a module is loaded from the root path, not sb3."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Lock root modules before importing sb3 versions
_ensure_root_module("index_manager")

from data_handler import DataHandler as NewDataHandler
from index_manager import IndexManager as NewIndexManager
from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager


# ============================================================================
# Default Configuration
# ============================================================================

# Environment variable to enable full dataset testing
FULL_DATASET = os.environ.get("FULL_DATASET", "0") == "1"


def create_default_config(dataset: str = "countries_s3") -> SimpleNamespace:
    """
    Create default configuration for DataHandler parity tests.
    
    Args:
        dataset: Dataset name (e.g., "countries_s3", "family")
        
    Returns:
        Configuration namespace with all test parameters
    """
    n_queries = None if FULL_DATASET else 50
    
    return SimpleNamespace(
        # Dataset settings
        dataset_name=dataset,
        data_path="./data/",
        
        # File names
        janus_file=None,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        
        # Query limits (None = use all)
        n_train_queries=n_queries,
        n_eval_queries=n_queries,
        n_test_queries=n_queries,
        
        # Corruption settings
        corruption_mode="dynamic",
        
        # Index manager settings
        padding_atoms=6,
        max_total_vars=1_000_000,
        
        # Depth settings (None = no filtering)
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        
        # Probabilistic facts settings
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=0.33,
        
        # Filtering settings
        filter_queries_by_rules=False,
        
        # Device
        device="cpu",
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a configuration namespace."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Stack Creation Helpers
# ============================================================================

def create_new_stack(config: SimpleNamespace):
    """
    Create the tensor-based data handling stack.
    
    Args:
        config: Test configuration
        
    Returns:
        Tuple of (DataHandler, IndexManager)
    """
    dh = NewDataHandler(
        dataset_name=config.dataset_name,
        base_path=config.data_path,
        janus_file=config.janus_file,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        n_train_queries=config.n_train_queries,
        n_eval_queries=config.n_eval_queries,
        n_test_queries=config.n_test_queries,
        train_depth=config.train_depth,
        valid_depth=config.valid_depth,
        test_depth=config.test_depth,
        prob_facts=config.prob_facts,
        topk_facts=config.topk_facts,
        topk_facts_threshold=config.topk_facts_threshold,
        corruption_mode=config.corruption_mode,
        filter_queries_by_rules=config.filter_queries_by_rules,
        deterministic=True,
    )
    
    im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=config.device,
    )
    
    dh.materialize_indices(im=im, device=torch.device(config.device))
    return dh, im


def create_sb3_stack(config: SimpleNamespace):
    """
    Create the SB3 (string-based) reference stack.
    
    Args:
        config: Test configuration
        
    Returns:
        Tuple of (DataHandler, IndexManager)
    """
    dh = SB3DataHandler(
        dataset_name=config.dataset_name,
        base_path=config.data_path,
        janus_file=config.janus_file,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        n_train_queries=config.n_train_queries,
        n_eval_queries=config.n_eval_queries,
        n_test_queries=config.n_test_queries,
        corruption_mode=config.corruption_mode,
        train_depth=config.train_depth,
        valid_depth=config.valid_depth,
        test_depth=config.test_depth,
        prob_facts=config.prob_facts,
        topk_facts=config.topk_facts,
        topk_facts_threshold=config.topk_facts_threshold,
        deterministic=True,
    )
    
    im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        config.max_total_vars,
        dh.rules,
        max_arity=dh.max_arity,
        device=config.device,
        padding_atoms=config.padding_atoms,
    )
    
    im.build_fact_index(dh.facts, deterministic=True)
    return dh, im


# ============================================================================
# Stack Caching
# ============================================================================

_STACK_CACHE = {}


def get_stacks(dataset: str):
    """
    Get or create stacks for a dataset (cached for efficiency).
    
    Args:
        dataset: Dataset name
        
    Returns:
        Tuple of (new_dh, new_im, sb3_dh, sb3_im)
    """
    if dataset not in _STACK_CACHE:
        config = create_default_config(dataset=dataset)
        dh_new, im_new = create_new_stack(config)
        dh_sb3, im_sb3 = create_sb3_stack(config)
        _STACK_CACHE[dataset] = (dh_new, im_new, dh_sb3, im_sb3)
    return _STACK_CACHE[dataset]


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def base_config():
    """Base configuration for all tests in this module."""
    return create_default_config()


# ============================================================================
# DataHandler Parity Tests
# ============================================================================

class TestDataHandlerParity:
    """
    Test DataHandler parity: facts, rules, and queries must match exactly.
    
    These tests verify that the tensor-based DataHandler produces identical
    output to the SB3 string-based implementation in terms of:
    - Facts content and ordering
    - Rules structure (head and body)
    - Query content for all splits
    - Vocabulary (constants and predicates)
    """

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_facts_parity(self, dataset: str):
        """
        Verify facts match between implementations.
        
        Checks:
        - Same number of facts
        - Same content (predicate, head, tail) for each fact
        - Same ordering
        """
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_facts = list(dh_new.facts_str)
        sb3_facts = [(f.predicate, f.args[0], f.args[1]) for f in dh_sb3.facts]
        
        assert len(new_facts) == len(sb3_facts), \
            f"Facts count mismatch: new={len(new_facts)} vs sb3={len(sb3_facts)}"
        
        for i, (new_f, sb3_f) in enumerate(zip(new_facts, sb3_facts)):
            assert new_f == sb3_f, \
                f"Fact mismatch at position {i}: new={new_f} vs sb3={sb3_f}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_rules_parity(self, dataset: str):
        """
        Verify rules match between implementations.
        
        Checks:
        - Same number of rules
        - Same head structure for each rule
        - Same body structure for each rule
        """
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        assert len(dh_new.rules) == len(dh_sb3.rules), \
            f"Rules count mismatch: new={len(dh_new.rules)} vs sb3={len(dh_sb3.rules)}"
        
        for i, (new_r, sb3_r) in enumerate(zip(dh_new.rules, dh_sb3.rules)):
            # Compare rule heads
            new_head = (new_r.head.predicate, *new_r.head.args)
            sb3_head = (sb3_r.head.predicate, *sb3_r.head.args)
            assert new_head == sb3_head, \
                f"Rule head mismatch at {i}: new={new_head} vs sb3={sb3_head}"
            
            # Compare rule bodies
            new_body = [(a.predicate, *a.args) for a in new_r.body]
            sb3_body = [(a.predicate, *a.args) for a in sb3_r.body]
            assert new_body == sb3_body, \
                f"Rule body mismatch at {i}: new={new_body} vs sb3={sb3_body}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_train_queries_parity(self, dataset: str):
        """Verify train queries match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_queries = list(dh_new.train_queries_str)
        sb3_queries = [(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.train_queries]
        
        assert len(new_queries) == len(sb3_queries), \
            f"Train queries count mismatch: new={len(new_queries)} vs sb3={len(sb3_queries)}"
        
        for i, (new_q, sb3_q) in enumerate(zip(new_queries, sb3_queries)):
            assert new_q == sb3_q, \
                f"Train query mismatch at {i}: new={new_q} vs sb3={sb3_q}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_valid_queries_parity(self, dataset: str):
        """Verify valid queries match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_queries = list(dh_new.valid_queries_str)
        sb3_queries = [(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.valid_queries]
        
        assert len(new_queries) == len(sb3_queries), \
            f"Valid queries count mismatch: new={len(new_queries)} vs sb3={len(sb3_queries)}"
        
        for i, (new_q, sb3_q) in enumerate(zip(new_queries, sb3_queries)):
            assert new_q == sb3_q, \
                f"Valid query mismatch at {i}: new={new_q} vs sb3={sb3_q}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_test_queries_parity(self, dataset: str):
        """Verify test queries match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_queries = list(dh_new.test_queries_str)
        sb3_queries = [(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.test_queries]
        
        assert len(new_queries) == len(sb3_queries), \
            f"Test queries count mismatch: new={len(new_queries)} vs sb3={len(sb3_queries)}"
        
        for i, (new_q, sb3_q) in enumerate(zip(new_queries, sb3_queries)):
            assert new_q == sb3_q, \
                f"Test query mismatch at {i}: new={new_q} vs sb3={sb3_q}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_vocabulary_parity(self, dataset: str):
        """
        Verify vocabulary (constants and predicates) matches.
        
        Checks:
        - Same number of constants
        - Same constant values and order
        - Same number of predicates
        - Same predicate values and order
        """
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        # Compare constants
        new_constants = list(dh_new.constants)
        sb3_constants = list(dh_sb3.constants)
        assert len(new_constants) == len(sb3_constants), \
            f"Constants count mismatch: new={len(new_constants)} vs sb3={len(sb3_constants)}"
        for i, (nc, sc) in enumerate(zip(new_constants, sb3_constants)):
            assert nc == sc, f"Constant mismatch at {i}: new={nc} vs sb3={sc}"
        
        # Compare predicates
        new_predicates = list(dh_new.predicates)
        sb3_predicates = list(dh_sb3.predicates)
        assert len(new_predicates) == len(sb3_predicates), \
            f"Predicates count mismatch: new={len(new_predicates)} vs sb3={len(sb3_predicates)}"
        for i, (np_pred, sp) in enumerate(zip(new_predicates, sb3_predicates)):
            assert np_pred == sp, f"Predicate mismatch at {i}: new={np_pred} vs sb3={sp}"


# ============================================================================
# IndexManager Parity Tests
# ============================================================================

class TestIndexManagerParity:
    """
    Test IndexManager parity: indices, mappings, and ranges must match.
    
    These tests verify that the tensor-based IndexManager produces identical
    mappings to the SB3 implementation.
    """

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_constant_mapping_parity(self, dataset: str):
        """Verify constant-to-index mappings match exactly."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert len(im_new.constant_str2idx) == len(im_sb3.constant_str2idx), \
            f"Constant mapping length mismatch: new={len(im_new.constant_str2idx)} vs sb3={len(im_sb3.constant_str2idx)}"
        
        for const, new_idx in im_new.constant_str2idx.items():
            sb3_idx = im_sb3.constant_str2idx.get(const)
            assert sb3_idx is not None, f"Constant '{const}' not found in SB3"
            assert new_idx == sb3_idx, \
                f"Constant '{const}' index mismatch: new={new_idx} vs sb3={sb3_idx}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_predicate_mapping_parity(self, dataset: str):
        """Verify predicate-to-index mappings match exactly."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert len(im_new.predicate_str2idx) == len(im_sb3.predicate_str2idx), \
            f"Predicate mapping length mismatch: new={len(im_new.predicate_str2idx)} vs sb3={len(im_sb3.predicate_str2idx)}"
        
        for pred, new_idx in im_new.predicate_str2idx.items():
            sb3_idx = im_sb3.predicate_str2idx.get(pred)
            assert sb3_idx is not None, f"Predicate '{pred}' not found in SB3"
            assert new_idx == sb3_idx, \
                f"Predicate '{pred}' index mismatch: new={new_idx} vs sb3={sb3_idx}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_variable_index_range_parity(self, dataset: str):
        """Verify variable index ranges match."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert im_new.constant_no == im_sb3.constant_no, \
            f"Constant count mismatch: new={im_new.constant_no} vs sb3={im_sb3.constant_no}"
        assert im_new.predicate_no == im_sb3.predicate_no, \
            f"Predicate count mismatch: new={im_new.predicate_no} vs sb3={im_sb3.predicate_no}"
        assert im_new.runtime_var_start_index == im_sb3.variable_start_index, \
            f"Variable start index mismatch: new={im_new.runtime_var_start_index} vs sb3={im_sb3.variable_start_index}"
        assert im_new.runtime_variable_no == im_sb3.max_total_vars, \
            f"Max total vars mismatch: new={im_new.runtime_variable_no} vs sb3={im_sb3.max_total_vars}"


# ============================================================================
# Materialization Parity Tests
# ============================================================================

class TestMaterializationParity:
    """
    Test materialization parity: tensors must match in shape and content.
    
    These tests verify that materialized tensors (facts, queries) are identical
    between implementations.
    """

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_materialized_facts_parity(self, dataset: str):
        """Verify materialized facts tensors match in content and order."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        new_facts_list = [tuple(t.tolist()) for t in im_new.facts_idx]
        sb3_facts_list = [tuple(t.tolist()) for t in im_sb3.facts_idx]
        
        assert len(new_facts_list) == len(sb3_facts_list), \
            f"Facts count mismatch: new={len(new_facts_list)} vs sb3={len(sb3_facts_list)}"
        
        for i, (nf, sf) in enumerate(zip(new_facts_list, sb3_facts_list)):
            assert nf == sf, f"Fact mismatch at {i}: new={nf} vs sb3={sf}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_materialized_queries_parity(self, dataset: str):
        """Verify materialized query tensors match."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        for split_name, sb3_queries in [
            ('train', dh_sb3.train_queries),
            ('valid', dh_sb3.valid_queries),
            ('test', dh_sb3.test_queries),
        ]:
            split = dh_new.get_materialized_split(split_name)
            new_queries = split.queries.squeeze(1)
            
            sb3_queries_indexed = []
            for q in sb3_queries:
                pred_idx = im_sb3.predicate_str2idx[q.predicate]
                head_idx = im_sb3.constant_str2idx[q.args[0]]
                tail_idx = im_sb3.constant_str2idx[q.args[1]]
                sb3_queries_indexed.append([pred_idx, head_idx, tail_idx])
            sb3_tensor = torch.tensor(sb3_queries_indexed, dtype=torch.long)
            
            assert new_queries.shape == sb3_tensor.shape, \
                f"{split_name} queries shape mismatch: new={new_queries.shape} vs sb3={sb3_tensor.shape}"
            
            for i in range(new_queries.shape[0]):
                assert torch.equal(new_queries[i], sb3_tensor[i]), \
                    f"{split_name} query mismatch at {i}: new={new_queries[i].tolist()} vs sb3={sb3_tensor[i].tolist()}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_all_known_triples_parity(self, dataset: str):
        """Verify all_known_triples_idx matches SB3's all_known_triples."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Convert SB3 all_known_triples to indices
        sb3_triples_list = []
        for t in dh_sb3.all_known_triples:
            pred_idx = im_sb3.predicate_str2idx[t.predicate]
            head_idx = im_sb3.constant_str2idx[t.args[0]]
            tail_idx = im_sb3.constant_str2idx[t.args[1]]
            sb3_triples_list.append((pred_idx, head_idx, tail_idx))
        
        new_triples_list = [tuple(t.tolist()) for t in dh_new.all_known_triples_idx]
        
        assert len(new_triples_list) == len(sb3_triples_list), \
            f"All known triples count mismatch: new={len(new_triples_list)} vs sb3={len(sb3_triples_list)}"
        
        for i, (nt, st) in enumerate(zip(new_triples_list, sb3_triples_list)):
            assert nt == st, f"Triple mismatch at {i}: new={nt} vs sb3={st}"


# ============================================================================
# CLI Runner
# ============================================================================

def run_parity_tests(
    dataset: str = "countries_s3",
    verbose: bool = False,
) -> bool:
    """
    Run DataHandler parity tests programmatically.
    
    Args:
        dataset: Dataset name to test
        verbose: Enable verbose output
        
    Returns:
        True if all tests pass, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"DATAHANDLER PARITY TEST")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"Full dataset mode: {FULL_DATASET}")
    print(f"{'='*70}\n")
    
    config = create_default_config(dataset=dataset)
    
    print("Creating tensor-based stack...")
    dh_new, im_new = create_new_stack(config)
    print(f"  ✓ Loaded {len(list(dh_new.facts_str))} facts, {len(dh_new.rules)} rules")
    
    print("Creating SB3 stack...")
    dh_sb3, im_sb3 = create_sb3_stack(config)
    print(f"  ✓ Loaded {len(dh_sb3.facts)} facts, {len(dh_sb3.rules)} rules")
    
    # Run comparisons
    all_passed = True
    
    # Facts comparison
    print("\nComparing facts...")
    new_facts = list(dh_new.facts_str)
    sb3_facts = [(f.predicate, f.args[0], f.args[1]) for f in dh_sb3.facts]
    if new_facts == sb3_facts:
        print(f"  ✓ Facts match ({len(new_facts)} facts)")
    else:
        print(f"  ✗ Facts mismatch")
        all_passed = False
    
    # Queries comparison
    for split_name, new_queries, sb3_queries in [
        ('train', list(dh_new.train_queries_str), dh_sb3.train_queries),
        ('valid', list(dh_new.valid_queries_str), dh_sb3.valid_queries),
        ('test', list(dh_new.test_queries_str), dh_sb3.test_queries),
    ]:
        sb3_q_list = [(q.predicate, q.args[0], q.args[1]) for q in sb3_queries]
        if new_queries == sb3_q_list:
            print(f"  ✓ {split_name.capitalize()} queries match ({len(new_queries)})")
        else:
            print(f"  ✗ {split_name.capitalize()} queries mismatch")
            all_passed = False
    
    # Summary
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED - DataHandlers are equivalent")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"{'='*70}\n")
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DataHandler Parity Tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/parity/test_data_handler_parity.py --dataset countries_s3
    python tests/parity/test_data_handler_parity.py --dataset family --verbose
    FULL_DATASET=1 python tests/parity/test_data_handler_parity.py
        """
    )
    parser.add_argument('--dataset', type=str, default='countries_s3',
                        choices=['countries_s3', 'family', 'fb15k237', 'wn18rr'],
                        help='Dataset name (default: countries_s3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    passed = run_parity_tests(
        dataset=args.dataset,
        verbose=args.verbose,
    )
    
    sys.exit(0 if passed else 1)
