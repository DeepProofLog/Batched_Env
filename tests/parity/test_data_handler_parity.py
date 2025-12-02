"""
Comprehensive parity tests for DataHandler and IndexManager.

Tests verify that the new tensor-based implementations produce exactly the same
results as the reference SB3 (string-based) implementations.

Run with full datasets: FULL_DATASET=1 pytest tests/parity/test_data_handler_parity.py -v
"""
from types import SimpleNamespace
from pathlib import Path
import sys
import importlib.util
import os

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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


# Lock root index_manager so later imports don't grab sb3/index_manager.py
_ensure_root_module("index_manager")
from data_handler import DataHandler as NewDataHandler
from index_manager import IndexManager as NewIndexManager
from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager


# ============================================================================
# Configuration
# ============================================================================

# Set FULL_DATASET=1 to run with full datasets (no query limits)
FULL_DATASET = os.environ.get("FULL_DATASET", "0") == "1"


def _base_args(dataset: str = "countries_s3"):
    """Base configuration for tests."""
    n = None if FULL_DATASET else 50
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
        padding_atoms=6,
        max_total_vars=1_000_000,
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=0.33,
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
        deterministic=True,
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
    return dh, im


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
        deterministic=True,
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
    im.build_fact_index(dh.facts, deterministic=True)
    return dh, im


# Module-level cache for stacks
_STACK_CACHE = {}


def get_stacks(dataset: str):
    """Get or create stacks for a dataset. Cached for efficiency."""
    if dataset not in _STACK_CACHE:
        args = _base_args(dataset=dataset)
        dh_new, im_new = _create_new_stack(args)
        dh_sb3, im_sb3 = _create_sb3_stack(args)
        _STACK_CACHE[dataset] = (dh_new, im_new, dh_sb3, im_sb3)
    return _STACK_CACHE[dataset]


# ============================================================================
# DataHandler Parity Tests
# ============================================================================

class TestDataHandlerParity:
    """Test DataHandler parity: facts, rules, and queries must match in count, content, and order."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_facts_parity(self, dataset):
        """Verify facts match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_facts = list(dh_new.facts_str)
        sb3_facts = [(f.predicate, f.args[0], f.args[1]) for f in dh_sb3.facts]
        
        assert len(new_facts) == len(sb3_facts), \
            f"Facts count mismatch: {len(new_facts)} vs {len(sb3_facts)}"
        
        for i, (new_f, sb3_f) in enumerate(zip(new_facts, sb3_facts)):
            assert new_f == sb3_f, \
                f"Fact mismatch at position {i}: new={new_f} vs sb3={sb3_f}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_rules_parity(self, dataset):
        """Verify rules match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        assert len(dh_new.rules) == len(dh_sb3.rules), \
            f"Rules count mismatch: {len(dh_new.rules)} vs {len(dh_sb3.rules)}"
        
        for i, (new_r, sb3_r) in enumerate(zip(dh_new.rules, dh_sb3.rules)):
            new_head = (new_r.head.predicate, *new_r.head.args)
            sb3_head = (sb3_r.head.predicate, *sb3_r.head.args)
            assert new_head == sb3_head, \
                f"Rule head mismatch at {i}: new={new_head} vs sb3={sb3_head}"
            
            new_body = [(a.predicate, *a.args) for a in new_r.body]
            sb3_body = [(a.predicate, *a.args) for a in sb3_r.body]
            assert new_body == sb3_body, \
                f"Rule body mismatch at {i}: new={new_body} vs sb3={sb3_body}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_train_queries_parity(self, dataset):
        """Verify train queries match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_queries = list(dh_new.train_queries_str)
        sb3_queries = [(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.train_queries]
        
        assert len(new_queries) == len(sb3_queries), \
            f"Train queries count mismatch: {len(new_queries)} vs {len(sb3_queries)}"
        
        for i, (new_q, sb3_q) in enumerate(zip(new_queries, sb3_queries)):
            assert new_q == sb3_q, \
                f"Train query mismatch at {i}: new={new_q} vs sb3={sb3_q}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_valid_queries_parity(self, dataset):
        """Verify valid queries match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_queries = list(dh_new.valid_queries_str)
        sb3_queries = [(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.valid_queries]
        
        assert len(new_queries) == len(sb3_queries), \
            f"Valid queries count mismatch: {len(new_queries)} vs {len(sb3_queries)}"
        
        for i, (new_q, sb3_q) in enumerate(zip(new_queries, sb3_queries)):
            assert new_q == sb3_q, \
                f"Valid query mismatch at {i}: new={new_q} vs sb3={sb3_q}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_test_queries_parity(self, dataset):
        """Verify test queries match in count, content, and order."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        new_queries = list(dh_new.test_queries_str)
        sb3_queries = [(q.predicate, q.args[0], q.args[1]) for q in dh_sb3.test_queries]
        
        assert len(new_queries) == len(sb3_queries), \
            f"Test queries count mismatch: {len(new_queries)} vs {len(sb3_queries)}"
        
        for i, (new_q, sb3_q) in enumerate(zip(new_queries, sb3_queries)):
            assert new_q == sb3_q, \
                f"Test query mismatch at {i}: new={new_q} vs sb3={sb3_q}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_vocabulary_parity(self, dataset):
        """Verify constants and predicates match in count and content."""
        dh_new, _, dh_sb3, _ = get_stacks(dataset)
        
        # Compare as lists directly
        new_constants = list(dh_new.constants)
        sb3_constants = list(dh_sb3.constants)
        assert len(new_constants) == len(sb3_constants), \
            f"Constants count mismatch: {len(new_constants)} vs {len(sb3_constants)}"
        for i, (nc, sc) in enumerate(zip(new_constants, sb3_constants)):
            assert nc == sc, f"Constant mismatch at {i}: new={nc} vs sb3={sc}"
        
        new_predicates = list(dh_new.predicates)
        sb3_predicates = list(dh_sb3.predicates)
        assert len(new_predicates) == len(sb3_predicates), \
            f"Predicates count mismatch: {len(new_predicates)} vs {len(sb3_predicates)}"
        for i, (np, sp) in enumerate(zip(new_predicates, sb3_predicates)):
            assert np == sp, f"Predicate mismatch at {i}: new={np} vs sb3={sp}"


# ============================================================================
# IndexManager Parity Tests
# ============================================================================

class TestIndexManagerParity:
    """Test IndexManager parity: indices, mappings, and ranges must match exactly."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_constant_mapping_parity(self, dataset):
        """Verify constant-to-index mappings match in count, content, and order."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert len(im_new.constant_str2idx) == len(im_sb3.constant_str2idx), \
            f"Constant mapping length mismatch: {len(im_new.constant_str2idx)} vs {len(im_sb3.constant_str2idx)}"
        
        for const, new_idx in im_new.constant_str2idx.items():
            sb3_idx = im_sb3.constant_str2idx.get(const)
            assert sb3_idx is not None, f"Constant '{const}' not found in SB3"
            assert new_idx == sb3_idx, \
                f"Constant '{const}' index mismatch: new={new_idx} vs sb3={sb3_idx}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_predicate_mapping_parity(self, dataset):
        """Verify predicate-to-index mappings match in count, content, and order."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert len(im_new.predicate_str2idx) == len(im_sb3.predicate_str2idx), \
            f"Predicate mapping length mismatch: {len(im_new.predicate_str2idx)} vs {len(im_sb3.predicate_str2idx)}"
        
        for pred, new_idx in im_new.predicate_str2idx.items():
            sb3_idx = im_sb3.predicate_str2idx.get(pred)
            assert sb3_idx is not None, f"Predicate '{pred}' not found in SB3"
            assert new_idx == sb3_idx, \
                f"Predicate '{pred}' index mismatch: new={new_idx} vs sb3={sb3_idx}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_variable_index_range_parity(self, dataset):
        """Verify variable index ranges match."""
        _, im_new, _, im_sb3 = get_stacks(dataset)
        
        assert im_new.constant_no == im_sb3.constant_no, \
            f"Constant count mismatch: {im_new.constant_no} vs {im_sb3.constant_no}"
        assert im_new.predicate_no == im_sb3.predicate_no, \
            f"Predicate count mismatch: {im_new.predicate_no} vs {im_sb3.predicate_no}"
        assert im_new.runtime_var_start_index == im_sb3.variable_start_index, \
            f"Variable start index mismatch: {im_new.runtime_var_start_index} vs {im_sb3.variable_start_index}"
        assert im_new.runtime_variable_no == im_sb3.max_total_vars, \
            f"Max total vars mismatch: {im_new.runtime_variable_no} vs {im_sb3.max_total_vars}"


# ============================================================================
# Materialization Parity Tests
# ============================================================================

class TestMaterializationParity:
    """Test materialization parity: tensors must match in shape, content, and order."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_materialized_facts_parity(self, dataset):
        """Verify materialized facts tensors match in content and order."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Compare facts_idx tensors directly - both are sorted by (pred_idx, head_idx, tail_idx)
        new_facts_list = [tuple(t.tolist()) for t in im_new.facts_idx]
        sb3_facts_list = [tuple(t.tolist()) for t in im_sb3.facts_idx]
        
        assert len(new_facts_list) == len(sb3_facts_list), \
            f"Facts count mismatch: {len(new_facts_list)} vs {len(sb3_facts_list)}"
        for i, (nf, sf) in enumerate(zip(new_facts_list, sb3_facts_list)):
            assert nf == sf, f"Fact mismatch at {i}: new={nf} vs sb3={sf}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_materialized_queries_parity(self, dataset):
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
                f"{split_name} queries shape mismatch: {new_queries.shape} vs {sb3_tensor.shape}"
            
            for i in range(new_queries.shape[0]):
                assert torch.equal(new_queries[i], sb3_tensor[i]), \
                    f"{split_name} query mismatch at {i}: new={new_queries[i].tolist()} vs sb3={sb3_tensor[i].tolist()}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_all_known_triples_parity(self, dataset):
        """Verify all_known_triples_idx matches SB3's all_known_triples."""
        dh_new, im_new, dh_sb3, im_sb3 = get_stacks(dataset)
        
        # Convert SB3 all_known_triples to indices as list
        sb3_triples_list = []
        for t in dh_sb3.all_known_triples:
            pred_idx = im_sb3.predicate_str2idx[t.predicate]
            head_idx = im_sb3.constant_str2idx[t.args[0]]
            tail_idx = im_sb3.constant_str2idx[t.args[1]]
            sb3_triples_list.append((pred_idx, head_idx, tail_idx))
        
        new_triples_list = [tuple(t.tolist()) for t in dh_new.all_known_triples_idx]
        
        assert len(new_triples_list) == len(sb3_triples_list), \
            f"All known triples count mismatch: {len(new_triples_list)} vs {len(sb3_triples_list)}"
        for i, (nt, st) in enumerate(zip(new_triples_list, sb3_triples_list)):
            assert nt == st, f"Triple mismatch at {i}: new={nt} vs sb3={st}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
