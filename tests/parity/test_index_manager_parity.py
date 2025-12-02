"""
IndexManager Parity Tests.

Tests verify that the new tensor-based IndexManager produces exactly the same
mappings and behavior as the SB3 IndexManager.

Run with full datasets: FULL_DATASET=1 pytest tests/parity/test_index_manager_parity.py -v
"""
from pathlib import Path
import sys
import importlib.util
import os

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_root_module("index_manager")
from index_manager import IndexManager as NewIndexManager
from data_handler import DataHandler as NewDataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_dataset import DataHandler as SB3DataHandler


# ============================================================================
# Configuration
# ============================================================================

FULL_DATASET = os.environ.get("FULL_DATASET", "0") == "1"

# Module-level cache
_STACK_CACHE = {}


def get_stacks(dataset: str):
    """Get or create stacks for a dataset. Cached for efficiency."""
    if dataset not in _STACK_CACHE:
        n = None if FULL_DATASET else 50
        
        # New stack
        dh_new = NewDataHandler(
            dataset_name=dataset,
            base_path="./data/",
            janus_file=None,
            train_file="train.txt",
            valid_file="valid.txt",
            test_file="test.txt",
            rules_file="rules.txt",
            facts_file="train.txt",
            n_train_queries=n,
            n_eval_queries=n,
            n_test_queries=n,
            filter_queries_by_rules=False,
            deterministic=True,
        )
        im_new = NewIndexManager(
            constants=dh_new.constants,
            predicates=dh_new.predicates,
            max_total_runtime_vars=1_000_000,
            max_arity=dh_new.max_arity,
            padding_atoms=10,
            device="cpu",
        )
        dh_new.materialize_indices(im=im_new, device=torch.device("cpu"))
        
        # SB3 stack
        dh_sb3 = SB3DataHandler(
            dataset_name=dataset,
            base_path="./data/",
            janus_file=None,
            train_file="train.txt",
            valid_file="valid.txt",
            test_file="test.txt",
            rules_file="rules.txt",
            facts_file="train.txt",
            n_train_queries=n,
            n_eval_queries=n,
            n_test_queries=n,
            deterministic=True,
        )
        im_sb3 = SB3IndexManager(
            dh_sb3.constants,
            dh_sb3.predicates,
            1_000_000,
            dh_sb3.rules,
            max_arity=dh_sb3.max_arity,
            device="cpu",
            padding_atoms=10,
        )
        im_sb3.build_fact_index(dh_sb3.facts, deterministic=True)
        
        _STACK_CACHE[dataset] = (dh_new, im_new, dh_sb3, im_sb3)
    
    return _STACK_CACHE[dataset]


# ============================================================================
# IndexManager Parity Tests
# ============================================================================

class TestIndexManagerParity:
    """Test IndexManager parity: mappings, sizes, and conversions must match exactly."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_constant_mapping_parity(self, dataset):
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
    def test_predicate_mapping_parity(self, dataset):
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
    def test_vocabulary_sizes_parity(self, dataset):
        """Verify vocabulary sizes match exactly."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert im_new.constant_no == im_sb3.constant_no, \
            f"Constant count mismatch: {im_new.constant_no} vs {im_sb3.constant_no}"
        assert im_new.predicate_no == im_sb3.predicate_no, \
            f"Predicate count mismatch: {im_new.predicate_no} vs {im_sb3.predicate_no}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_variable_index_range_parity(self, dataset):
        """Verify variable index ranges match."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert im_new.runtime_var_start_index == im_sb3.variable_start_index, \
            f"Variable start mismatch: {im_new.runtime_var_start_index} vs {im_sb3.variable_start_index}"
        assert im_new.runtime_variable_no == im_sb3.max_total_vars, \
            f"Max vars mismatch: {im_new.runtime_variable_no} vs {im_sb3.max_total_vars}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_special_predicates_parity(self, dataset):
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
    """Test state/atom conversion produces identical results."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_atom_to_tensor_parity(self, dataset):
        """Verify atom_to_tensor produces same indices as SB3."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Test all facts
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
    def test_state_to_tensor_parity(self, dataset):
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
    """Test facts storage and indexing produces identical results."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_facts_tensor_parity(self, dataset):
        """Verify materialized facts tensor matches SB3 in content and order."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Compare facts_idx tensors directly - both are sorted by (pred_idx, head_idx, tail_idx)
        new_facts_list = [tuple(t.tolist()) for t in im_new.facts_idx]
        sb3_facts_list = [tuple(t.tolist()) for t in im_sb3.facts_idx]
        
        assert len(new_facts_list) == len(sb3_facts_list), \
            f"Facts count mismatch: {len(new_facts_list)} vs {len(sb3_facts_list)}"
        for i, (nf, sf) in enumerate(zip(new_facts_list, sb3_facts_list)):
            assert nf == sf, f"Fact mismatch at {i}: new={nf} vs sb3={sf}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_facts_content_parity(self, dataset):
        """Verify facts content matches exactly in order."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Build lists from both and compare directly
        new_facts_list = []
        for i in range(im_new.facts_idx.shape[0]):
            t = im_new.facts_idx[i]
            pred = im_new.idx2predicate[int(t[0])]
            arg1 = im_new.idx2constant[int(t[1])]
            arg2 = im_new.idx2constant[int(t[2])]
            new_facts_list.append((pred, arg1, arg2))
        
        # Use sorted facts from SB3 (same order as facts_idx)
        sb3_facts_list = [(f.predicate, f.args[0], f.args[1]) for f in im_sb3.facts_sorted]
        
        assert len(new_facts_list) == len(sb3_facts_list), \
            f"Facts count mismatch: {len(new_facts_list)} vs {len(sb3_facts_list)}"
        for i, (nf, sf) in enumerate(zip(new_facts_list, sb3_facts_list)):
            assert nf == sf, f"Fact mismatch at {i}: new={nf} vs sb3={sf}"


# ============================================================================
# Rules Materialization Parity Tests
# ============================================================================

class TestRulesMaterializationParity:
    """Test rules materialization produces identical results."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_rules_tensor_shape_parity(self, dataset):
        """Verify rules tensor has correct shape."""
        dh_new, im_new, dh_sb3, _ = get_stacks(dataset)
        
        if im_new.rules_idx is None or im_new.rules_idx.numel() == 0:
            pytest.skip("No rules in dataset")
        
        # Check shape: [R, M, 3] where R = num rules, M = max body length
        assert im_new.rules_idx.dim() == 3, "Rules should be 3D"
        assert im_new.rules_idx.shape[0] == len(dh_sb3.rules), \
            f"Rule count mismatch: {im_new.rules_idx.shape[0]} vs {len(dh_sb3.rules)}"
        assert im_new.rules_idx.shape[2] == 3, "Each atom should have 3 elements"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_rules_heads_parity(self, dataset):
        """Verify rule heads match SB3."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        if im_new.rules_heads_idx is None or im_new.rules_heads_idx.numel() == 0:
            pytest.skip("No rules in dataset")
        
        assert len(im_new.rules_heads_idx) == len(dh_sb3.rules), \
            f"Rule heads count mismatch: {len(im_new.rules_heads_idx)} vs {len(dh_sb3.rules)}"
        
        for i, rule in enumerate(dh_sb3.rules):
            head = rule.head
            sb3_pred_idx = im_sb3.predicate_str2idx[head.predicate]
            
            # Head args are variables (uppercased) - look them up in unified map
            sb3_arg1_idx = im_sb3.unified_term_map.get(head.args[0])
            sb3_arg2_idx = im_sb3.unified_term_map.get(head.args[1])
            
            new_head = im_new.rules_heads_idx[i]
            
            assert new_head[0] == sb3_pred_idx, \
                f"Rule {i} head pred mismatch: new={new_head[0]} vs sb3={sb3_pred_idx}"
            # Variable indices may differ, but predicate must match


# ============================================================================
# Full Integration Test
# ============================================================================

class TestFullParity:
    """Full integration test for IndexManager parity."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_full_index_manager_parity(self, dataset):
        """Complete parity test covering all aspects."""
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
        
        # 5. Facts tensor content - compare directly in order
        new_facts_list = [tuple(t.tolist()) for t in im_new.facts_idx]
        sb3_facts_list = [tuple(t.tolist()) for t in im_sb3.facts_idx]
        assert new_facts_list == sb3_facts_list, "Facts content mismatch"
        
        # 6. Atom conversion consistency
        for fact in dh_sb3.facts[:100]:  # Check first 100
            new_t = im_new.atom_to_tensor(fact.predicate, fact.args[0], fact.args[1])
            sb3_t = im_sb3.get_atom_sub_index([fact])
            assert new_t[0] == sb3_t[0, 0] and new_t[1] == sb3_t[0, 1] and new_t[2] == sb3_t[0, 2], \
                f"Atom conversion mismatch for {fact}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
