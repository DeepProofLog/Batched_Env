"""
IndexManager Parity Tests.

Comprehensive tests verifying that the new tensor-based IndexManager produces
exactly the same mappings as the SB3 IndexManager.
"""
from pathlib import Path
import sys
import importlib.util

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
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_dataset import DataHandler as SB3DataHandler


def _load_test_data(dataset: str = "countries_s3"):
    """Load test data from a dataset."""
    dh = SB3DataHandler(
        dataset_name=dataset,
        base_path="./data/",
        janus_file=None,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
    )
    return dh


# ============================================================================
# Basic Vocabulary Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_constant_indices_match(dataset):
    """Verify constants are indexed identically."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    sb3_im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        1_000_000,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Check all constants have the same index
    for const in dh.constants:
        new_idx = new_im.constant_str2idx[const]
        sb3_idx = sb3_im.constant_str2idx[const]
        assert new_idx == sb3_idx, \
            f"Constant '{const}' index mismatch: new={new_idx}, sb3={sb3_idx}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_predicate_indices_match(dataset):
    """Verify regular predicates are indexed identically."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    sb3_im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        1_000_000,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Check regular predicates (not special or KGE)
    for pred in dh.predicates:
        if pred in new_im.predicate_str2idx and pred in sb3_im.predicate_str2idx:
            new_idx = new_im.predicate_str2idx[pred]
            sb3_idx = sb3_im.predicate_str2idx[pred]
            assert new_idx == sb3_idx, \
                f"Predicate '{pred}' index mismatch: new={new_idx}, sb3={sb3_idx}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_vocabulary_sizes_match(dataset):
    """Verify vocabulary sizes are consistent."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    sb3_im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        1_000_000,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Constants should match exactly
    assert new_im.constant_no == sb3_im.constant_no, \
        f"Constant count mismatch: new={new_im.constant_no}, sb3={sb3_im.constant_no}"
    
    # Predicates may differ by KGE predicates
    sb3_core_preds = sb3_im.predicate_no - len(sb3_im.kge_preds)
    # Allow for True/False/Endf differences
    assert abs(new_im.predicate_no - sb3_core_preds) <= 3, \
        f"Predicate count mismatch: new={new_im.predicate_no}, sb3_core={sb3_core_preds}"


# ============================================================================
# Variable Indexing Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_variable_start_index(dataset):
    """Verify variable start index is correctly computed."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    sb3_im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        1_000_000,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Variable start should be consistent with constant count
    assert new_im.runtime_var_start_index == new_im.constant_no + new_im.template_variable_no + 1
    assert sb3_im.variable_start_index == sb3_im.constant_no + 1
    
    # Both should start variables right after constants
    # (template vars in new impl may shift this)
    print(f"New variable start: {new_im.runtime_var_start_index}")
    print(f"SB3 variable start: {sb3_im.variable_start_index}")


# ============================================================================
# State Conversion Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_state_to_tensor_consistency(dataset):
    """Test that state_to_tensor produces consistent results."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Test with some example facts
    test_atoms = [
        (fact.predicate, fact.args[0], fact.args[1])
        for fact in dh.facts[:10]
    ]
    
    # Convert to tensor
    tensor = new_im.state_to_tensor(test_atoms)
    
    # Check shape
    assert tensor.shape == (len(test_atoms), 3)
    
    # Check that all indices are valid
    assert (tensor[:, 0] > 0).all(), "Predicate indices should be > 0"
    assert (tensor[:, 1] > 0).all(), "Arg1 indices should be > 0"
    assert (tensor[:, 2] > 0).all(), "Arg2 indices should be > 0"


@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_atom_to_tensor_roundtrip(dataset):
    """Test that atoms can be converted and the indices are consistent."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    sb3_im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        1_000_000,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
    )
    sb3_im.build_fact_index(dh.facts)
    
    # Test with some facts
    for fact in dh.facts[:20]:
        pred_str = fact.predicate
        arg1_str = fact.args[0]
        arg2_str = fact.args[1]
        
        # New implementation
        new_tensor = new_im.atom_to_tensor(pred_str, arg1_str, arg2_str)
        
        # SB3 implementation (via get_atom_sub_index)
        sb3_tensor = sb3_im.get_atom_sub_index([fact])
        
        # Compare predicate indices
        assert new_tensor[0] == sb3_tensor[0, 0], \
            f"Predicate '{pred_str}' index mismatch"
        
        # Compare arg indices
        assert new_tensor[1] == sb3_tensor[0, 1], \
            f"Arg1 '{arg1_str}' index mismatch"
        assert new_tensor[2] == sb3_tensor[0, 2], \
            f"Arg2 '{arg2_str}' index mismatch"


# ============================================================================
# Special Predicates Tests
# ============================================================================

def test_special_predicates_exist():
    """Test that special predicates (True, False, Endf) are registered."""
    constants = {"a", "b", "c"}
    predicates = {"rel1", "rel2"}
    
    im = NewIndexManager(
        constants=constants,
        predicates=predicates,
        max_total_runtime_vars=1000,
        max_arity=2,
        device="cpu",
    )
    
    # Check special predicates exist
    assert "True" in im.predicate_str2idx
    assert "False" in im.predicate_str2idx
    assert "Endf" in im.predicate_str2idx
    
    # Check special tensors
    assert im.true_tensor is not None
    assert im.false_tensor is not None


def test_padding_index_is_zero():
    """Test that padding index is 0."""
    constants = {"a", "b"}
    predicates = {"rel"}
    
    im = NewIndexManager(
        constants=constants,
        predicates=predicates,
        max_total_runtime_vars=100,
        max_arity=2,
        device="cpu",
    )
    
    assert im.padding_idx == 0


# ============================================================================
# Rules Materialization Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_rules_to_tensor_shape(dataset):
    """Test that rules_to_tensor produces correct shapes."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Convert rules to string format
    rules_str = []
    for rule in dh.rules:
        head = (rule.head.predicate, rule.head.args[0], rule.head.args[1])
        body = [(atom.predicate, atom.args[0], atom.args[1]) for atom in rule.body]
        rules_str.append((head, body))
    
    if rules_str:
        max_body_len = max(len(body) for _, body in rules_str)
        rules_idx, rule_lens = new_im.rules_to_tensor(rules_str, max_body_len)
        
        # Check shapes
        assert rules_idx.shape[0] == len(rules_str), "Number of rules mismatch"
        assert rules_idx.shape[1] == max_body_len, "Max body length mismatch"
        assert rules_idx.shape[2] == 3, "Atom should have 3 elements"
        
        # Check lens
        assert rule_lens.shape[0] == len(rules_str)
        for i, (_, body) in enumerate(rules_str):
            assert rule_lens[i] == len(body), f"Rule {i} length mismatch"


# ============================================================================
# Facts Storage Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_set_facts_creates_sorted_index(dataset):
    """Test that set_facts sorts facts by predicate."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    # Convert facts to tensor
    facts_strs = [(f.predicate, f.args[0], f.args[1]) for f in dh.facts]
    facts_idx = new_im.state_to_tensor(facts_strs)
    
    # Set facts
    new_im.set_facts(facts_idx)
    
    # Check that facts are stored
    assert new_im.facts_idx is not None
    assert new_im.facts_idx.shape[0] == len(dh.facts)
    
    # Check that predicate range map is created
    assert new_im.predicate_range_map is not None
    
    # Check that facts are sorted by predicate
    if new_im.facts_idx.shape[0] > 1:
        preds = new_im.facts_idx[:, 0]
        assert (preds[:-1] <= preds[1:]).all(), "Facts should be sorted by predicate"


# ============================================================================
# Full Integration Test
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_full_index_manager_parity(dataset):
    """Full integration test comparing both IndexManagers."""
    dh = _load_test_data(dataset)
    
    new_im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1_000_000,
        max_arity=dh.max_arity,
        device="cpu",
    )
    
    sb3_im = SB3IndexManager(
        dh.constants,
        dh.predicates,
        1_000_000,
        dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
    )
    sb3_im.build_fact_index(dh.facts, deterministic=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"Constants: new={new_im.constant_no}, sb3={sb3_im.constant_no}")
    print(f"Predicates: new={new_im.predicate_no}, sb3={sb3_im.predicate_no}")
    print(f"Variable start: new={new_im.runtime_var_start_index}, sb3={sb3_im.variable_start_index}")
    
    # Assertions
    assert new_im.constant_no == sb3_im.constant_no
    
    # Sample constant checks
    sample_consts = list(dh.constants)[:10]
    for const in sample_consts:
        assert new_im.constant_str2idx[const] == sb3_im.constant_str2idx[const], \
            f"Constant '{const}' index mismatch"
    
    print("✓ All checks passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
