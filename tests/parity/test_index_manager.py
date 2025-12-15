"""
IndexManager Parity Tests.

Tests verifying that the tensor-based IndexManager produces EXACTLY the same
mappings and behavior as the SB3 IndexManager implementation.

This module tests:
- Constant-to-index mappings
- Predicate-to-index mappings
- Vocabulary sizes
- Variable index ranges
- Special predicates (True, False, Endf)
- Atom-to-tensor conversions
- Facts storage and indexing

Usage:
    pytest tests/parity/test_index_manager_parity.py -v
    pytest tests/parity/test_index_manager_parity.py -v -k "countries"
    pytest tests/parity/test_index_manager_parity.py -v -k "family"
    
    # Run with full datasets:
    FULL_DATASET=1 pytest tests/parity/test_index_manager_parity.py -v
    
    # Run from command line:
    python tests/parity/test_index_manager_parity.py --dataset countries_s3
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

from index_manager import IndexManager as NewIndexManager
from data_handler import DataHandler as NewDataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_dataset import DataHandler as SB3DataHandler


# ============================================================================
# Default Configuration
# ============================================================================

FULL_DATASET = os.environ.get("FULL_DATASET", "0") == "1"


def create_default_config(dataset: str = "countries_s3") -> SimpleNamespace:
    """
    Create default configuration for IndexManager parity tests.
    
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
        
        # Query limits
        n_train_queries=n_queries,
        n_eval_queries=n_queries,
        n_test_queries=n_queries,
        
        # Index manager settings
        padding_atoms=10,
        max_total_vars=1000,
        
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
    """Create the tensor-based IndexManager stack."""
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
        filter_queries_by_rules=False,
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
    """Create the SB3 IndexManager stack."""
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
    """Get or create stacks for a dataset (cached for efficiency)."""
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
# IndexManager Parity Tests
# ============================================================================

class TestIndexManagerParity:
    """
    Test IndexManager parity: mappings, sizes, and conversions must match.
    
    Verifies that tensor-based IndexManager produces identical mappings
    and index ranges as the SB3 implementation.
    """

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_constant_mapping_parity(self, dataset: str):
        """Verify constant-to-index mappings match in count and content."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert len(im_new.constant_str2idx) == len(im_sb3.constant_str2idx), \
            f"Constant count mismatch: {len(im_new.constant_str2idx)} vs {len(im_sb3.constant_str2idx)}"
        
        for const, new_idx in im_new.constant_str2idx.items():
            sb3_idx = im_sb3.constant_str2idx.get(const)
            assert sb3_idx is not None, f"Constant '{const}' not found in SB3"
            assert new_idx == sb3_idx, \
                f"Constant '{const}' index mismatch: new={new_idx} vs sb3={sb3_idx}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_predicate_mapping_parity(self, dataset: str):
        """Verify predicate-to-index mappings match in count and content."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert len(im_new.predicate_str2idx) == len(im_sb3.predicate_str2idx), \
            f"Predicate count mismatch: {len(im_new.predicate_str2idx)} vs {len(im_sb3.predicate_str2idx)}"
        
        for pred, new_idx in im_new.predicate_str2idx.items():
            sb3_idx = im_sb3.predicate_str2idx.get(pred)
            assert sb3_idx is not None, f"Predicate '{pred}' not found in SB3"
            assert new_idx == sb3_idx, \
                f"Predicate '{pred}' index mismatch: new={new_idx} vs sb3={sb3_idx}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_vocabulary_sizes_parity(self, dataset: str):
        """Verify vocabulary sizes match exactly."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert im_new.constant_no == im_sb3.constant_no, \
            f"Constant count mismatch: {im_new.constant_no} vs {im_sb3.constant_no}"
        assert im_new.predicate_no == im_sb3.predicate_no, \
            f"Predicate count mismatch: {im_new.predicate_no} vs {im_sb3.predicate_no}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_variable_index_range_parity(self, dataset: str):
        """Verify variable index ranges match."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert im_new.runtime_var_start_index == im_sb3.variable_start_index, \
            f"Variable start mismatch: {im_new.runtime_var_start_index} vs {im_sb3.variable_start_index}"
        assert im_new.runtime_variable_no == im_sb3.max_total_vars, \
            f"Max vars mismatch: {im_new.runtime_variable_no} vs {im_sb3.max_total_vars}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_special_predicates_parity(self, dataset: str):
        """Verify special predicates (True, False, Endf) have same indices."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        for pred in ["True", "False", "Endf"]:
            assert pred in im_new.predicate_str2idx, f"Missing {pred} in new"
            assert pred in im_sb3.predicate_str2idx, f"Missing {pred} in sb3"
            assert im_new.predicate_str2idx[pred] == im_sb3.predicate_str2idx[pred], \
                f"Special pred {pred} index mismatch"


# ============================================================================
# State Conversion Parity Tests
# ============================================================================

class TestStateConversionParity:
    """
    Test state/atom conversion produces identical results.
    
    Verifies that atom_to_tensor and state_to_tensor produce the same
    indices as SB3's get_atom_sub_index.
    """

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_atom_to_tensor_parity(self, dataset: str):
        """Verify atom_to_tensor produces same indices as SB3."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        for i, fact in enumerate(dh_sb3.facts):
            pred_str = fact.predicate
            arg1_str = fact.args[0]
            arg2_str = fact.args[1]
            
            # New implementation
            new_tensor = im_new.atom_to_tensor(pred_str, arg1_str, arg2_str)
            
            # SB3 implementation
            sb3_tensor = im_sb3.get_atom_sub_index([fact])
            
            assert new_tensor[0] == sb3_tensor[0, 0], \
                f"Fact {i} pred mismatch: new={new_tensor[0]} vs sb3={sb3_tensor[0, 0]}"
            assert new_tensor[1] == sb3_tensor[0, 1], \
                f"Fact {i} arg1 mismatch: new={new_tensor[1]} vs sb3={sb3_tensor[0, 1]}"
            assert new_tensor[2] == sb3_tensor[0, 2], \
                f"Fact {i} arg2 mismatch: new={new_tensor[2]} vs sb3={sb3_tensor[0, 2]}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_state_to_tensor_parity(self, dataset: str):
        """Verify state_to_tensor produces same indices as SB3."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Convert facts
        facts_strs = [(f.predicate, f.args[0], f.args[1]) for f in dh_sb3.facts]
        new_tensor = im_new.state_to_tensor(facts_strs)
        
        # Check each fact
        for i, fact in enumerate(dh_sb3.facts):
            sb3_tensor = im_sb3.get_atom_sub_index([fact])
            
            assert new_tensor[i, 0] == sb3_tensor[0, 0], \
                f"Fact {i} pred mismatch in batch"
            assert new_tensor[i, 1] == sb3_tensor[0, 1], \
                f"Fact {i} arg1 mismatch in batch"
            assert new_tensor[i, 2] == sb3_tensor[0, 2], \
                f"Fact {i} arg2 mismatch in batch"


# ============================================================================
# Facts Storage Parity Tests
# ============================================================================

class TestFactsStorageParity:
    """
    Test facts storage and indexing produces identical results.
    
    Verifies that stored facts tensors match between implementations.
    """

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_facts_tensor_parity(self, dataset: str):
        """Verify materialized facts tensor matches SB3 in content and order."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        new_facts_list = [tuple(t.tolist()) for t in im_new.facts_idx]
        sb3_facts_list = [tuple(t.tolist()) for t in im_sb3.facts_idx]
        
        assert len(new_facts_list) == len(sb3_facts_list), \
            f"Facts count mismatch: {len(new_facts_list)} vs {len(sb3_facts_list)}"
        
        for i, (nf, sf) in enumerate(zip(new_facts_list, sb3_facts_list)):
            assert nf == sf, f"Fact mismatch at {i}: new={nf} vs sb3={sf}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_facts_content_parity(self, dataset: str):
        """Verify facts content matches exactly in order."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        # Build lists from both and compare
        new_facts_list = []
        for i in range(im_new.facts_idx.shape[0]):
            t = im_new.facts_idx[i]
            pred = im_new.idx2predicate[int(t[0])]
            arg1 = im_new.idx2constant[int(t[1])]
            arg2 = im_new.idx2constant[int(t[2])]
            new_facts_list.append((pred, arg1, arg2))
        
        # Use sorted facts from SB3
        sb3_facts_list = [(f.predicate, f.args[0], f.args[1]) for f in im_sb3.facts_sorted]
        
        assert len(new_facts_list) == len(sb3_facts_list), \
            f"Facts count mismatch: {len(new_facts_list)} vs {len(sb3_facts_list)}"
        
        for i, (nf, sf) in enumerate(zip(new_facts_list, sb3_facts_list)):
            assert nf == sf, f"Fact mismatch at {i}: new={nf} vs sb3={sf}"


# ============================================================================
# Full Integration Test
# ============================================================================

class TestFullParity:
    """Full integration test for IndexManager parity."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_full_index_manager_parity(self, dataset: str):
        """
        Complete parity test covering all aspects.
        
        This is a comprehensive test that verifies:
        - Vocabulary sizes
        - Variable ranges
        - All constant mappings
        - All predicate mappings
        - Facts tensor content
        - Atom conversion consistency
        """
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # 1. Vocabulary sizes
        assert im_new.constant_no == im_sb3.constant_no, "Constant count mismatch"
        assert im_new.predicate_no == im_sb3.predicate_no, "Predicate count mismatch"
        
        # 2. Variable ranges
        assert im_new.runtime_var_start_index == im_sb3.variable_start_index, "Variable start mismatch"
        
        # 3. All constant mappings
        for const, new_idx in im_new.constant_str2idx.items():
            sb3_idx = im_sb3.constant_str2idx.get(const)
            assert new_idx == sb3_idx, f"Constant '{const}' index mismatch"
        
        # 4. All predicate mappings
        for pred, new_idx in im_new.predicate_str2idx.items():
            sb3_idx = im_sb3.predicate_str2idx.get(pred)
            assert new_idx == sb3_idx, f"Predicate '{pred}' index mismatch"
        
        # 5. Facts tensor content
        new_facts_list = [tuple(t.tolist()) for t in im_new.facts_idx]
        sb3_facts_list = [tuple(t.tolist()) for t in im_sb3.facts_idx]
        assert new_facts_list == sb3_facts_list, "Facts content mismatch"
        
        # 6. Atom conversion consistency (first 100 facts)
        for fact in dh_sb3.facts[:100]:
            new_t = im_new.atom_to_tensor(fact.predicate, fact.args[0], fact.args[1])
            sb3_t = im_sb3.get_atom_sub_index([fact])
            assert new_t[0] == sb3_t[0, 0] and new_t[1] == sb3_t[0, 1] and new_t[2] == sb3_t[0, 2], \
                f"Atom conversion mismatch for {fact}"


# ============================================================================
# CLI Runner
# ============================================================================

def run_parity_tests(
    dataset: str = "countries_s3",
    verbose: bool = False,
) -> bool:
    """
    Run IndexManager parity tests programmatically.
    
    Args:
        dataset: Dataset name to test
        verbose: Enable verbose output
        
    Returns:
        True if all tests pass, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"INDEXMANAGER PARITY TEST")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"Full dataset mode: {FULL_DATASET}")
    print(f"{'='*70}\n")
    
    config = create_default_config(dataset=dataset)
    
    print("Creating tensor-based stack...")
    dh_new, im_new = create_new_stack(config)
    print(f"  ✓ Constants: {im_new.constant_no}, Predicates: {im_new.predicate_no}")
    
    print("Creating SB3 stack...")
    dh_sb3, im_sb3 = create_sb3_stack(config)
    print(f"  ✓ Constants: {im_sb3.constant_no}, Predicates: {im_sb3.predicate_no}")
    
    # Run comparisons
    all_passed = True
    
    # Check vocabulary sizes
    print("\nComparing vocabulary sizes...")
    if im_new.constant_no == im_sb3.constant_no and im_new.predicate_no == im_sb3.predicate_no:
        print(f"  ✓ Vocabulary sizes match")
    else:
        print(f"  ✗ Vocabulary sizes mismatch")
        all_passed = False
    
    # Check mappings
    print("Comparing constant mappings...")
    const_match = all(
        im_new.constant_str2idx.get(c) == im_sb3.constant_str2idx.get(c)
        for c in im_new.constant_str2idx
    )
    if const_match:
        print(f"  ✓ Constant mappings match ({len(im_new.constant_str2idx)} constants)")
    else:
        print(f"  ✗ Constant mappings mismatch")
        all_passed = False
    
    print("Comparing predicate mappings...")
    pred_match = all(
        im_new.predicate_str2idx.get(p) == im_sb3.predicate_str2idx.get(p)
        for p in im_new.predicate_str2idx
    )
    if pred_match:
        print(f"  ✓ Predicate mappings match ({len(im_new.predicate_str2idx)} predicates)")
    else:
        print(f"  ✗ Predicate mappings mismatch")
        all_passed = False
    
    # Summary
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED - IndexManagers are equivalent")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"{'='*70}\n")
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='IndexManager Parity Tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/parity/test_index_manager_parity.py --dataset countries_s3
    python tests/parity/test_index_manager_parity.py --dataset family --verbose
    FULL_DATASET=1 python tests/parity/test_index_manager_parity.py
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
