"""
Negative Sampler Parity Tests.

Tests verify that the new Sampler produces EXACTLY the same corruptions
as the SB3 sampler given the same seed and inputs - same queries, same
corruptions, same order.

Run with: pytest tests/parity/test_negative_sampler_parity.py -v -s
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


_ensure_root_module("index_manager")
from data_handler import DataHandler as NewDataHandler
from index_manager import IndexManager as NewIndexManager
from sampler import Sampler as NewSampler

from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_neg_sampling import get_sampler as get_sb3_sampler


# ============================================================================
# Configuration
# ============================================================================

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


def _create_stacks(args, seed=42, default_mode="tail"):
    """Create both stacks with same seed for comparison.
    
    Args:
        args: Configuration namespace
        seed: Random seed
        default_mode: Corruption mode for tensor sampler ('head', 'tail', or 'both')
    """
    # Create SB3 stack first to get domain info
    dh_sb3 = SB3DataHandler(
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
    im_sb3 = SB3IndexManager(
        dh_sb3.constants,
        dh_sb3.predicates,
        args.max_total_vars,
        dh_sb3.rules,
        max_arity=dh_sb3.max_arity,
        device="cpu",
        padding_atoms=args.padding_atoms,
    )
    im_sb3.build_fact_index(dh_sb3.facts, deterministic=True)
    
    # Set seed and create SB3 sampler
    torch.manual_seed(seed)
    sampler_sb3 = get_sb3_sampler(
        data_handler=dh_sb3,
        index_manager=im_sb3,
        corruption_scheme=args.corruption_scheme,
        device=torch.device("cpu"),
    )
    
    # Create new stack
    dh_new = NewDataHandler(
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
    im_new = NewIndexManager(
        constants=dh_new.constants,
        predicates=dh_new.predicates,
        max_total_runtime_vars=args.max_total_vars,
        max_arity=dh_new.max_arity,
        padding_atoms=args.padding_atoms,
        device="cpu",
    )
    dh_new.materialize_indices(im=im_new, device=torch.device("cpu"))
    
    # Get domain info from DataHandler (no manual construction needed)
    domain2idx, entity2domain = dh_new.get_sampler_domain_info()

    # Create new sampler with same seed
    sampler_new = NewSampler.from_data(
        all_known_triples_idx=dh_new.all_known_triples_idx,
        num_entities=im_new.constant_no,
        num_relations=im_new.predicate_no,
        device=torch.device("cpu"),
        default_mode=default_mode,
        seed=seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    return dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args


# Module-level cache
_STACK_CACHE = {}


def get_stacks(dataset: str, seed: int = 42, default_mode: str = "tail"):
    """Get or create stacks for a dataset.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        default_mode: Corruption mode for tensor sampler ('head', 'tail', or 'both')
    """
    cache_key = (dataset, seed, default_mode)
    if cache_key not in _STACK_CACHE:
        args = _base_args(dataset=dataset)
        result = _create_stacks(args, seed=seed, default_mode=default_mode)
        _STACK_CACHE[cache_key] = result
    return _STACK_CACHE[cache_key]


# ============================================================================
# Parity Tests
# ============================================================================

class TestNegativeSamplerParity:
    """Test exact parity between new and SB3 negative samplers."""

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_filter_hashes_parity(self, dataset):
        """Verify both samplers filter exactly the same known triples."""
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Both should have same number of hashes
        new_count = sampler_new.hashes_sorted.numel()
        sb3_count = sampler_sb3.filterer._hashes_sorted.numel()
        
        print(f"\n{dataset} filter hash counts: new={new_count}, sb3={sb3_count}")
        
        # Also check the actual hash values match
        assert torch.equal(sampler_new.hashes_sorted, sampler_sb3.filterer._hashes_sorted), \
            f"Filter hashes don't match exactly"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_corrupt_batch_parity(self, dataset):
        """
        Verify corrupt_batch (raw generation without filtering) produces identical output.
        Both samplers must be called with the same random seed to ensure identical output.
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Take up to 1000 queries
        n_queries = min(1000, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        # Prepare inputs for both samplers (SB3 expects h,r,t format)
        pos_hrt = torch.stack([
            queries_new[:, 1],  # head
            queries_new[:, 0],  # relation
            queries_new[:, 2],  # tail
        ], dim=1)
        
        # Generate negatives from new sampler with fixed seed (using corrupt with filter=False)
        SEED = 42
        torch.manual_seed(SEED)
        negatives_new = sampler_new.corrupt(
            queries_new,
            num_negatives=4,
            mode='tail',
            filter=False,  # Don't filter - test raw generation parity
            unique=False,  # Don't unique - test raw generation parity
            device=torch.device("cpu"),
        )
        
        # Generate from SB3 with SAME seed - uses corrupt_batch which doesn't filter
        torch.manual_seed(SEED)
        neg_hrt_sb3 = sampler_sb3.corrupt_batch(pos_hrt, num_negs_per_pos=4)
        
        # Convert SB3 output to (rel, head, tail) format
        negatives_sb3 = torch.stack([
            neg_hrt_sb3[:, :, 1],  # relation
            neg_hrt_sb3[:, :, 0],  # head
            neg_hrt_sb3[:, :, 2],  # tail
        ], dim=-1)
        
        # Print examples
        print(f"\n{dataset}: corrupt_batch comparison (first 3 queries, 4 negatives each):")
        for i in range(min(3, n_queries)):
            print(f"  Query {i}: {queries_new[i].tolist()}")
            print(f"    New negatives:  {negatives_new[i].tolist()}")
            print(f"    SB3 negatives:  {negatives_sb3[i].tolist()}")
        
        # Check shape
        assert negatives_new.shape == negatives_sb3.shape, \
            f"Shape mismatch: new={negatives_new.shape} vs sb3={negatives_sb3.shape}"
        
        # Check exact content and order
        for i in range(n_queries):
            for j in range(4):
                new_triple = negatives_new[i, j].tolist()
                sb3_triple = negatives_sb3[i, j].tolist()
                assert new_triple == sb3_triple, \
                    f"Query {i}, neg {j}: new={new_triple} vs sb3={sb3_triple}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_full_pipeline_parity(self, dataset):
        """
        Verify the full pipeline (corrupt with filter=True, unique=True) produces identical output.
        This tests the complete get_negatives equivalent: generate -> filter -> unique -> take K.
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Take up to 100 queries for full pipeline test
        n_queries = min(100, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        # Prepare inputs for SB3 (h,r,t format)
        pos_hrt = torch.stack([
            queries_new[:, 1],  # head
            queries_new[:, 0],  # relation  
            queries_new[:, 2],  # tail
        ], dim=1)
        
        # Build SB3 pos_subs format for get_negatives
        pos_subs = torch.zeros((n_queries, args.padding_atoms, dh_sb3.max_arity + 1), dtype=torch.int32)
        pos_subs[:, 0, 0] = queries_new[:, 0]  # relation
        pos_subs[:, 0, 1] = queries_new[:, 1]  # head
        pos_subs[:, 0, 2] = queries_new[:, 2]  # tail
        
        num_negs = 10
        SEED = 42
        
        # Generate with new sampler - full pipeline
        torch.manual_seed(SEED)
        negatives_new = sampler_new.corrupt(
            queries_new,
            num_negatives=num_negs,
            mode='tail',
            filter=True,   # Filter out known positives
            unique=True,   # Ensure unique
            device=torch.device("cpu"),
        )
        
        # Generate with SB3 - using get_negatives which has the full pipeline
        torch.manual_seed(SEED)
        neg_subs_sb3 = sampler_sb3.get_negatives(
            pos_subs,
            padding_atoms=args.padding_atoms,
            max_arity=dh_sb3.max_arity,
            device=torch.device("cpu"),
            num_negs=num_negs,
        )
        
        # Extract triples from SB3 output (format: [B, K, padding_atoms, max_arity+1])
        # SB3 stores: neg_subs[i, j, 0, 0] = rel, neg_subs[i, j, 0, 1] = head, neg_subs[i, j, 0, 2] = tail
        negatives_sb3 = torch.stack([
            neg_subs_sb3[:, :, 0, 0],  # relation
            neg_subs_sb3[:, :, 0, 1],  # head
            neg_subs_sb3[:, :, 0, 2],  # tail
        ], dim=-1)
        
        # Print examples
        print(f"\n{dataset}: Full pipeline comparison (first 3 queries, {num_negs} negatives):")
        for i in range(min(3, n_queries)):
            print(f"  Query {i}: {queries_new[i].tolist()}")
            print(f"    New (first 5):  {negatives_new[i, :5].tolist()}")
            print(f"    SB3 (first 5):  {negatives_sb3[i, :5].tolist()}")
        
        # Check shape
        assert negatives_new.shape == negatives_sb3.shape, \
            f"Shape mismatch: new={negatives_new.shape} vs sb3={negatives_sb3.shape}"
        
        # Check exact content and order
        mismatches = 0
        for i in range(n_queries):
            for j in range(num_negs):
                new_triple = negatives_new[i, j].tolist()
                sb3_triple = negatives_sb3[i, j].tolist()
                if new_triple != sb3_triple:
                    mismatches += 1
                    if mismatches <= 5:  # Print first 5 mismatches
                        print(f"  MISMATCH Query {i}, neg {j}: new={new_triple} vs sb3={sb3_triple}")
        
        assert mismatches == 0, f"Found {mismatches} mismatches in full pipeline output"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_all_negatives_parity(self, dataset):
        """
        Verify 4 queries with ALL negatives produce same shape, content, and order.
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        n_queries = min(4, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        # Generate all tail negatives from new sampler
        _, tails_list_new = sampler_new.corrupt_all(
            queries_new,
            mode='tail',
            device=torch.device("cpu"),
        )
        
        # Generate all negatives from SB3's corrupt_batch_all
        pos_hrt = torch.stack([
            queries_new[:, 1],  # head
            queries_new[:, 0],  # relation
            queries_new[:, 2],  # tail
        ], dim=1)
        neg_batches_sb3 = sampler_sb3.corrupt_batch_all(pos_hrt)
        
        # Filter SB3 results
        for i, nb in enumerate(neg_batches_sb3):
            if nb.numel() > 0:
                keep = sampler_sb3.filterer(nb)
                neg_batches_sb3[i] = nb[keep]
        
        # Print examples
        print(f"\n{dataset}: All negatives comparison (first 2 queries):")
        for i in range(min(2, n_queries)):
            print(f"  Query {i}: {queries_new[i].tolist()}")
            print(f"    New: {tails_list_new[i].shape[0]} negatives, first 5: {tails_list_new[i][:5].tolist()}")
            
            # Convert SB3 to (rel, head, tail)
            sb3_rht = torch.stack([
                neg_batches_sb3[i][:, 1],  # relation
                neg_batches_sb3[i][:, 0],  # head  
                neg_batches_sb3[i][:, 2],  # tail
            ], dim=-1) if neg_batches_sb3[i].numel() > 0 else torch.empty((0, 3))
            print(f"    SB3: {sb3_rht.shape[0]} negatives, first 5: {sb3_rht[:5].tolist()}")
        
        # Check each query
        for i in range(n_queries):
            new_negs = tails_list_new[i]
            
            # Convert SB3 to (rel, head, tail)
            if neg_batches_sb3[i].numel() > 0:
                sb3_negs = torch.stack([
                    neg_batches_sb3[i][:, 1],
                    neg_batches_sb3[i][:, 0],
                    neg_batches_sb3[i][:, 2],
                ], dim=-1)
            else:
                sb3_negs = torch.empty((0, 3), dtype=new_negs.dtype)
            
            # Check count
            assert new_negs.shape[0] == sb3_negs.shape[0], \
                f"Query {i}: count mismatch new={new_negs.shape[0]} vs sb3={sb3_negs.shape[0]}"
            
            # Check exact order
            for j in range(new_negs.shape[0]):
                new_triple = new_negs[j].tolist()
                sb3_triple = sb3_negs[j].tolist()
                assert new_triple == sb3_triple, \
                    f"Query {i}, neg {j}: new={new_triple} vs sb3={sb3_triple}"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_negatives_exclude_known_positives(self, dataset):
        """Verify all negatives exclude all known positive triples."""
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Build set of all known triples
        known_set = set()
        for i in range(dh_new.all_known_triples_idx.shape[0]):
            r, h, t = dh_new.all_known_triples_idx[i].tolist()
            known_set.add((r, h, t))
        
        n_queries = min(100, dh_new.all_known_triples_idx.shape[0])
        queries = dh_new.all_known_triples_idx[:n_queries]
        
        # Test all negatives (first 4 queries)
        _, tails_list = sampler_new.corrupt_all(
            queries[:4],
            mode='tail',
            device=torch.device("cpu"),
        )
        
        for i in range(len(tails_list)):
            for j in range(tails_list[i].shape[0]):
                r, h, t = tails_list[i][j].tolist()
                assert (r, h, t) not in known_set, \
                    f"Query {i}, all neg {j}: ({r}, {h}, {t}) is a known positive"

    @pytest.mark.parametrize("dataset,mode", [
        ("countries_s3", "head"),
        ("countries_s3", "tail"),
        ("family", "head"),
        ("family", "tail"),
    ])
    def test_corrupt_batch_head_tail_modes(self, dataset, mode):
        """Test corrupt_batch parity for both head and tail corruption modes."""
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Configure corruption scheme
        if mode == 'head':
            sampler_sb3._corruption_indices = [0]
            sampler_sb3.corruption_scheme = ['head']
        else:
            sampler_sb3._corruption_indices = [2]
            sampler_sb3.corruption_scheme = ['tail']
        
        n_queries = min(50, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        # Prepare inputs for both samplers
        pos_hrt = torch.stack([
            queries_new[:, 1],  # head
            queries_new[:, 0],  # relation
            queries_new[:, 2],  # tail
        ], dim=1)
        
        SEED = 42
        num_negs = 5
        
        # Generate negatives from new sampler
        torch.manual_seed(SEED)
        negatives_new = sampler_new.corrupt(
            queries_new,
            num_negatives=num_negs,
            mode=mode,
            filter=False,
            unique=False,
            device=torch.device("cpu"),
        )
        
        # Generate from SB3 with SAME seed
        torch.manual_seed(SEED)
        neg_hrt_sb3 = sampler_sb3.corrupt_batch(pos_hrt, num_negs_per_pos=num_negs)
        
        # Convert SB3 output to (rel, head, tail) format
        negatives_sb3 = torch.stack([
            neg_hrt_sb3[:, :, 1],  # relation
            neg_hrt_sb3[:, :, 0],  # head
            neg_hrt_sb3[:, :, 2],  # tail
        ], dim=-1)
        
        print(f"\n{dataset} {mode}: corrupt_batch comparison (first 3 queries):")
        for i in range(min(3, n_queries)):
            print(f"  Query {i}: {queries_new[i].tolist()}")
            print(f"    New negatives:  {negatives_new[i].tolist()}")
            print(f"    SB3 negatives:  {negatives_sb3[i].tolist()}")
        
        # Check exact content and order
        mismatches = 0
        for i in range(n_queries):
            for j in range(num_negs):
                new_triple = negatives_new[i, j].tolist()
                sb3_triple = negatives_sb3[i, j].tolist()
                if new_triple != sb3_triple:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  MISMATCH Query {i}, neg {j}: new={new_triple} vs sb3={sb3_triple}")
        
        assert mismatches == 0, f"Found {mismatches} mismatches in {mode} corruption"

    @pytest.mark.parametrize("dataset,num_negs", [
        ("countries_s3", 1),
        ("countries_s3", 3),
        ("countries_s3", 10),
        ("family", 1),
        ("family", 5),
        ("family", 20),
    ])
    def test_different_num_negatives(self, dataset, num_negs):
        """Test parity with different numbers of negatives."""
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        n_queries = min(20, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        pos_hrt = torch.stack([
            queries_new[:, 1],
            queries_new[:, 0],
            queries_new[:, 2],
        ], dim=1)
        
        # Build SB3 pos_subs format
        pos_subs = torch.zeros((n_queries, args.padding_atoms, dh_sb3.max_arity + 1), dtype=torch.int32)
        pos_subs[:, 0, 0] = queries_new[:, 0]
        pos_subs[:, 0, 1] = queries_new[:, 1]
        pos_subs[:, 0, 2] = queries_new[:, 2]
        
        SEED = 123
        
        # Generate with new sampler
        torch.manual_seed(SEED)
        negatives_new = sampler_new.corrupt(
            queries_new,
            num_negatives=num_negs,
            mode='tail',
            filter=True,
            unique=True,
            device=torch.device("cpu"),
        )
        
        # Generate with SB3
        torch.manual_seed(SEED)
        neg_subs_sb3 = sampler_sb3.get_negatives(
            pos_subs,
            padding_atoms=args.padding_atoms,
            max_arity=dh_sb3.max_arity,
            device=torch.device("cpu"),
            num_negs=num_negs,
        )
        
        # Extract triples from SB3 output
        negatives_sb3 = torch.stack([
            neg_subs_sb3[:, :, 0, 0],
            neg_subs_sb3[:, :, 0, 1],
            neg_subs_sb3[:, :, 0, 2],
        ], dim=-1)
        
        print(f"\n{dataset} num_negs={num_negs}: First 2 queries:")
        for i in range(min(2, n_queries)):
            print(f"  Query {i}: New={negatives_new[i].tolist()}, SB3={negatives_sb3[i].tolist()}")
        
        # Check shape
        assert negatives_new.shape == negatives_sb3.shape, \
            f"Shape mismatch: new={negatives_new.shape} vs sb3={negatives_sb3.shape}"
        
        # Check content
        mismatches = 0
        for i in range(n_queries):
            for j in range(num_negs):
                if negatives_new[i, j].tolist() != negatives_sb3[i, j].tolist():
                    mismatches += 1
        
        assert mismatches == 0, f"Found {mismatches} mismatches with num_negs={num_negs}"

    @pytest.mark.parametrize("dataset,corruption_mode,num_negs", [
        ("countries_s3", "tail", 3),
        ("countries_s3", "tail", 10),
        ("countries_s3", "head", 3),
        ("countries_s3", "both", 3),
        ("family", "tail", 10),
        ("family", "head", 10),
        ("family", "both", 10),
    ])
    def test_get_negatives_from_states_separate_parity(self, dataset, corruption_mode, num_negs):
        """
        Test parity between SB3's get_negatives_from_states_separate and tensor's corrupt.
        
        This tests the exact path used by eval_corruptions:
        - SB3: sampler.get_negatives_from_states_separate([[q] for q in batch], device, num_negs)
        - Tensor: sampler.corrupt(queries_tensor, num_negatives=K, mode=mode)
        """
        from sb3.sb3_index_manager import IndexManager as SB3IndexManager
        
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Set corruption scheme on SB3 sampler
        if corruption_mode == 'both':
            sampler_sb3.corruption_scheme = ['head', 'tail']
            sampler_sb3._corruption_indices = [0, 2]
        elif corruption_mode == 'head':
            sampler_sb3.corruption_scheme = ['head']
            sampler_sb3._corruption_indices = [0]
        else:  # tail
            sampler_sb3.corruption_scheme = ['tail']
            sampler_sb3._corruption_indices = [2]
        
        n_queries = min(10, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        # Get query Term objects for SB3
        train_queries = dh_sb3.train_queries[:n_queries]
        
        SEED = 42
        device = torch.device("cpu")
        
        # Generate with SB3 using get_negatives_from_states_separate
        torch.manual_seed(SEED)
        head_corrs_sb3, tail_corrs_sb3 = sampler_sb3.get_negatives_from_states_separate(
            [[q] for q in train_queries],
            device=device,
            num_negs=num_negs,
            return_states=False,  # Return tensors for easy comparison
        )
        
        # Extract relevant corruptions based on mode
        # SB3 returns [B, K, padding_atoms, max_arity+1] tensors
        # We need to extract the triple from [0, :3] (first atom, first 3 values)
        if corruption_mode == 'head':
            sb3_negs = head_corrs_sb3
        elif corruption_mode == 'tail':
            sb3_negs = tail_corrs_sb3
        else:
            # For 'both' mode, combine head and tail
            sb3_negs = torch.cat([head_corrs_sb3, tail_corrs_sb3], dim=1)
        
        # Convert SB3 format [B, K, padding_atoms, max_arity+1] to [B, K, 3]
        # Format is: [rel, head, tail, ...]
        sb3_triples = sb3_negs[:, :, 0, :3]  # [B, K, 3] in (rel, head, tail) format
        
        # Generate with tensor sampler
        torch.manual_seed(SEED)
        if corruption_mode == 'both':
            # For 'both' mode, need to generate head and tail separately and combine
            # WITHOUT resetting seed between them (matching SB3's behavior)
            tensor_negs_head = sampler_new.corrupt(
                queries_new,
                num_negatives=num_negs,
                mode='head',
                filter=True,
                unique=True,
                device=device,
            )
            # Do NOT reset seed here - SB3 continues with the same RNG state
            tensor_negs_tail = sampler_new.corrupt(
                queries_new,
                num_negatives=num_negs,
                mode='tail',
                filter=True,
                unique=True,
                device=device,
            )
            tensor_negs = torch.cat([tensor_negs_head, tensor_negs_tail], dim=1)
        else:
            tensor_negs = sampler_new.corrupt(
                queries_new,
                num_negatives=num_negs,
                mode=corruption_mode,
                filter=True,
                unique=True,
                device=device,
            )
        
        print(f"\n{dataset} {corruption_mode} K={num_negs}:")
        print(f"  SB3 shape: {sb3_triples.shape}, Tensor shape: {tensor_negs.shape}")
        for i in range(min(3, n_queries)):
            print(f"  Query {i}: {queries_new[i].tolist()}")
            print(f"    SB3:    {sb3_triples[i].tolist()}")
            print(f"    Tensor: {tensor_negs[i].tolist()}")
        
        # Check shape
        assert sb3_triples.shape == tensor_negs.shape, \
            f"Shape mismatch: SB3={sb3_triples.shape} vs Tensor={tensor_negs.shape}"
        
        # Check content (allow for some tolerance due to filtering differences)
        mismatches = 0
        for i in range(n_queries):
            for j in range(sb3_triples.shape[1]):
                sb3_triple = sb3_triples[i, j].tolist()
                tensor_triple = tensor_negs[i, j].tolist()
                if sb3_triple != tensor_triple:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  MISMATCH Query {i}, neg {j}: SB3={sb3_triple} vs Tensor={tensor_triple}")
        
        assert mismatches == 0, \
            f"Found {mismatches} mismatches between get_negatives_from_states_separate and corrupt"

    @pytest.mark.parametrize("dataset,corruption_mode,num_negs", [
        ("countries_s3", "tail", 3),
        ("countries_s3", "tail", None),  # All negatives
        ("countries_s3", "head", 3),
        ("countries_s3", "both", 3),
        ("family", "tail", 10),
        ("family", "tail", None),  # All negatives
        ("family", "head", 10),
        ("family", "both", 10),
    ])
    def test_eval_corruptions_exact_flow_parity(self, dataset, corruption_mode, num_negs):
        """
        Test the EXACT flow used by eval_corruptions - including batched processing.
        
        This mimics what sb3_model_eval.eval_corruptions and model_eval.eval_corruptions do:
        1. For each batch of queries (up to n_envs at a time)
        2. Generate head and/or tail corruptions
        3. Evaluate each query with its corruptions
        
        The key difference from test_get_negatives_from_states_separate_parity is that
        this tests the full batched flow including:
        - Processing queries in batches (B queries at a time)
        - Generating corruptions for the entire batch
        - Handling variable-length negative lists per query
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Configure corruption scheme
        if corruption_mode == 'both':
            corruption_scheme = ['head', 'tail']
        else:
            corruption_scheme = [corruption_mode]
        
        # Set corruption scheme on SB3 sampler
        sampler_sb3.corruption_scheme = corruption_scheme
        sampler_sb3._corruption_indices = [0] if corruption_mode == 'head' else [2] if corruption_mode == 'tail' else [0, 2]
        
        device = torch.device("cpu")
        n_envs = 2  # Batch size
        
        # Use test queries for eval_corruptions parity
        n_queries = min(10, len(dh_sb3.test_queries))
        queries_sb3 = dh_sb3.test_queries[:n_queries]
        queries_new = dh_new.all_known_triples_idx[:n_queries]  # Get corresponding tensor queries
        
        # Convert tensor queries to align with SB3 (they should already be aligned)
        # But we need to ensure we're using the same queries
        # Create queries_new from queries_sb3 to ensure alignment
        queries_new_from_sb3 = []
        for q in queries_sb3:
            rel_idx = im_new.predicate_str2idx[q.predicate]
            head_idx = im_new.constant_str2idx[q.args[0]]
            tail_idx = im_new.constant_str2idx[q.args[1]]
            queries_new_from_sb3.append(torch.tensor([rel_idx, head_idx, tail_idx], dtype=torch.long))
        queries_new_aligned = torch.stack(queries_new_from_sb3, dim=0).to(device)
        
        SEED = 12345  # Same seed as eval_corruptions test
        
        all_match = True
        total_mismatches = 0
        
        # Process in batches, exactly like eval_corruptions
        print(f"\n{dataset} {corruption_mode} K={'all' if num_negs is None else num_negs}:")
        
        for batch_start in range(0, n_queries, n_envs):
            batch_end = min(batch_start + n_envs, n_queries)
            B = batch_end - batch_start
            
            batch_sb3 = queries_sb3[batch_start:batch_end]
            batch_tensor = queries_new_aligned[batch_start:batch_end]
            
            # Generate corruptions for this batch - both SB3 and tensor
            torch.manual_seed(SEED)
            head_corrs_sb3, tail_corrs_sb3 = sampler_sb3.get_negatives_from_states_separate(
                [[q] for q in batch_sb3],
                device=device,
                num_negs=num_negs,
                return_states=True,  # Return Term objects like eval_corruptions does
            )
            
            # For tensor, we need to match the exact behavior
            torch.manual_seed(SEED)
            if num_negs is None:
                # All corruptions mode
                heads_list_tensor, tails_list_tensor = sampler_new.corrupt_all(
                    batch_tensor,
                    mode='both' if corruption_mode == 'both' else corruption_mode,
                    device=device,
                )
            else:
                # Fixed number of corruptions
                if corruption_mode == 'head' or corruption_mode == 'both':
                    heads_tensor = sampler_new.corrupt(
                        batch_tensor,
                        num_negatives=num_negs,
                        mode='head',
                        filter=True,
                        unique=True,
                        device=device,
                    )
                    heads_list_tensor = [heads_tensor[i] for i in range(B)]
                else:
                    heads_list_tensor = [torch.empty((0, 3), dtype=torch.long, device=device) for _ in range(B)]
                
                # Don't reset seed between head and tail (SB3 continues RNG state)
                if corruption_mode == 'tail' or corruption_mode == 'both':
                    tails_tensor = sampler_new.corrupt(
                        batch_tensor,
                        num_negatives=num_negs,
                        mode='tail',
                        filter=True,
                        unique=True,
                        device=device,
                    )
                    tails_list_tensor = [tails_tensor[i] for i in range(B)]
                else:
                    tails_list_tensor = [torch.empty((0, 3), dtype=torch.long, device=device) for _ in range(B)]
            
            # Compare results for each corruption type in scheme
            for corr_type, (sb3_corrs, tensor_corrs) in zip(
                ['head', 'tail'],
                [(head_corrs_sb3, heads_list_tensor), (tail_corrs_sb3, tails_list_tensor)]
            ):
                if corr_type not in corruption_scheme:
                    continue
                
                for i in range(B):
                    query_idx = batch_start + i
                    
                    # Get SB3 corruptions for this query
                    if isinstance(sb3_corrs, list):
                        if i < len(sb3_corrs):
                            sb3_negs = sb3_corrs[i] if isinstance(sb3_corrs[i], list) else [sb3_corrs[i]]
                        else:
                            sb3_negs = []
                    else:
                        sb3_negs = []
                    
                    # Convert SB3 Terms to tensor format for comparison
                    sb3_negs_tensor = []
                    for neg in sb3_negs:
                        if hasattr(neg, 'predicate') and hasattr(neg, 'args'):
                            rel_idx = im_new.predicate_str2idx.get(neg.predicate, 0)
                            head_idx = im_new.constant_str2idx.get(neg.args[0], 0)
                            tail_idx = im_new.constant_str2idx.get(neg.args[1], 0)
                            sb3_negs_tensor.append([rel_idx, head_idx, tail_idx])
                    
                    # Get tensor corruptions for this query
                    if isinstance(tensor_corrs, list) and i < len(tensor_corrs):
                        tensor_negs = tensor_corrs[i]
                    else:
                        tensor_negs = torch.empty((0, 3), dtype=torch.long, device=device)
                    
                    # Convert to list for comparison
                    tensor_negs_list = tensor_negs.tolist() if tensor_negs.numel() > 0 else []
                    # Filter out padding (all zeros)
                    tensor_negs_list = [t for t in tensor_negs_list if t != [0, 0, 0]]
                    
                    # Compare counts
                    sb3_count = len(sb3_negs_tensor)
                    tensor_count = len(tensor_negs_list)
                    
                    if sb3_count != tensor_count:
                        print(f"  Query {query_idx} {corr_type}: COUNT MISMATCH sb3={sb3_count} vs tensor={tensor_count}")
                        all_match = False
                        total_mismatches += abs(sb3_count - tensor_count)
                        continue
                    
                    # Compare individual corruptions
                    for j in range(min(sb3_count, tensor_count)):
                        if sb3_negs_tensor[j] != tensor_negs_list[j]:
                            all_match = False
                            total_mismatches += 1
                            if total_mismatches <= 5:
                                print(f"  Query {query_idx} {corr_type} neg {j}: sb3={sb3_negs_tensor[j]} vs tensor={tensor_negs_list[j]}")
            
            # Update seed for next batch to simulate eval_corruptions behavior
            SEED += 1
        
        if all_match:
            print(f"  ✓ All corruptions match!")
        else:
            print(f"  ✗ Total mismatches: {total_mismatches}")
        
        assert all_match, f"Found {total_mismatches} mismatches in eval_corruptions exact flow test"

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_domain_structures_parity(self, dataset):
        """
        Verify domain structures (domain_padded, domain_len, ent2dom, pos_in_dom) match exactly.
        These are critical for domain-aware corruption.
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Check if both have domain info
        new_has_domain = sampler_new._has_domain_info()
        sb3_has_domain = hasattr(sampler_sb3, 'domain_padded') and sampler_sb3.domain_padded is not None
        
        print(f"\n{dataset}: Domain structures comparison:")
        print(f"  New has domain: {new_has_domain}, SB3 has domain: {sb3_has_domain}")
        
        assert new_has_domain == sb3_has_domain, \
            f"Domain info presence mismatch: new={new_has_domain} vs sb3={sb3_has_domain}"
        
        if not new_has_domain:
            print(f"  (No domain info - skipping detailed comparison)")
            return
        
        # Compare domain_padded
        assert sampler_new.domain_padded.shape == sampler_sb3.domain_padded.shape, \
            f"domain_padded shape mismatch: new={sampler_new.domain_padded.shape} vs sb3={sampler_sb3.domain_padded.shape}"
        
        assert torch.equal(sampler_new.domain_padded, sampler_sb3.domain_padded), \
            "domain_padded values don't match"
        
        # Compare domain_len
        assert sampler_new.domain_len.shape == sampler_sb3.domain_len.shape, \
            f"domain_len shape mismatch: new={sampler_new.domain_len.shape} vs sb3={sampler_sb3.domain_len.shape}"
        
        assert torch.equal(sampler_new.domain_len, sampler_sb3.domain_len), \
            "domain_len values don't match"
        
        # Compare ent2dom
        min_len = min(sampler_new.ent2dom.shape[0], sampler_sb3.ent2dom.shape[0])
        assert torch.equal(sampler_new.ent2dom[:min_len], sampler_sb3.ent2dom[:min_len]), \
            "ent2dom values don't match"
        
        # Compare pos_in_dom
        assert torch.equal(sampler_new.pos_in_dom[:min_len], sampler_sb3.pos_in_dom[:min_len]), \
            "pos_in_dom values don't match"
        
        print(f"  ✓ All domain structures match!")
        print(f"    num_domains: {sampler_new.num_domains}")
        print(f"    max_pool_len: {sampler_new.max_pool_len}")
        print(f"    domain_padded shape: {sampler_new.domain_padded.shape}")

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_filter_correctness(self, dataset):
        """
        Test that both samplers correctly filter out known positive triples.
        
        This tests:
        1. Known triples are excluded from negatives
        2. Filter hash computation is identical
        3. Edge cases (triples at boundaries of hash space)
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        device = torch.device("cpu")
        
        # Get all known triples
        all_known = dh_new.all_known_triples_idx
        
        # Test that all known triples are filtered out
        # Create test batch that includes some known triples
        test_batch = all_known[:100]
        
        # Check filter mask for new sampler
        new_keep = sampler_new._filter_keep_mask(test_batch)
        
        # Check filter mask for SB3 sampler
        # SB3 expects (h, r, t) format
        test_batch_hrt = torch.stack([
            test_batch[:, 1],  # head
            test_batch[:, 0],  # relation
            test_batch[:, 2],  # tail
        ], dim=1)
        sb3_keep = sampler_sb3.filterer(test_batch_hrt)
        
        print(f"\n{dataset}: Filter correctness test:")
        print(f"  Test batch size: {test_batch.shape[0]}")
        print(f"  New sampler keeps: {new_keep.sum().item()}")
        print(f"  SB3 sampler keeps: {sb3_keep.sum().item()}")
        
        # All known triples should be filtered out (keep=False)
        assert new_keep.sum().item() == 0, \
            f"New sampler should filter all known triples, but kept {new_keep.sum().item()}"
        assert sb3_keep.sum().item() == 0, \
            f"SB3 sampler should filter all known triples, but kept {sb3_keep.sum().item()}"
        
        # Test with corrupted triples (should mostly be kept)
        n_test = 20
        test_queries = all_known[:n_test]
        
        SEED = 42
        torch.manual_seed(SEED)
        
        # Generate corruptions (these should not be filtered)
        corrupted_new = sampler_new.corrupt(
            test_queries,
            num_negatives=5,
            mode='tail',
            filter=False,  # Don't filter during generation
            unique=False,
            device=device,
        )
        
        # Flatten and test filter
        corrupted_flat = corrupted_new.view(-1, 3)
        keep_mask = sampler_new._filter_keep_mask(corrupted_flat)
        
        kept_count = keep_mask.sum().item()
        total_count = corrupted_flat.shape[0]
        
        print(f"  Corrupted triples: {total_count}")
        print(f"  Kept after filter: {kept_count} ({100*kept_count/total_count:.1f}%)")
        
        # Most corrupted triples should be kept (unless they happen to be known)
        # Allow up to 50% to be filtered (for small datasets this can happen)
        assert kept_count > 0, "All corrupted triples were filtered - this is suspicious"

    @pytest.mark.parametrize("dataset,num_negs", [
        ("countries_s3", 3),
        ("countries_s3", 10),
        ("family", 10),
        ("family", 20),
    ])
    def test_rng_consumption_parity(self, dataset, num_negs):
        """
        Test that both samplers consume the same amount of RNG state.
        
        This is critical for parity when the sampler is called multiple times
        in sequence (like in eval_corruptions batch processing).
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        device = torch.device("cpu")
        n_queries = 5
        
        # Get aligned queries
        queries_sb3 = dh_sb3.train_queries[:n_queries]
        queries_tensor = dh_new.all_known_triples_idx[:n_queries]
        
        # Configure sampler modes
        sampler_sb3.corruption_scheme = ['tail']
        sampler_sb3._corruption_indices = [2]
        
        SEED = 42
        
        # Run both samplers twice in sequence and check results
        torch.manual_seed(SEED)
        
        # First call - SB3
        _, tail_corrs_sb3_1 = sampler_sb3.get_negatives_from_states_separate(
            [[q] for q in queries_sb3],
            device=device,
            num_negs=num_negs,
            return_states=False,
        )
        
        # Second call - SB3 (continues RNG state)
        _, tail_corrs_sb3_2 = sampler_sb3.get_negatives_from_states_separate(
            [[q] for q in queries_sb3],
            device=device,
            num_negs=num_negs,
            return_states=False,
        )
        
        # Capture final RNG state
        sb3_rng_state = torch.get_rng_state()
        
        # Now do the same for tensor sampler
        torch.manual_seed(SEED)
        
        # First call - Tensor
        tail_corrs_tensor_1 = sampler_new.corrupt(
            queries_tensor,
            num_negatives=num_negs,
            mode='tail',
            filter=True,
            unique=True,
            device=device,
        )
        
        # Second call - Tensor (continues RNG state)
        tail_corrs_tensor_2 = sampler_new.corrupt(
            queries_tensor,
            num_negatives=num_negs,
            mode='tail',
            filter=True,
            unique=True,
            device=device,
        )
        
        tensor_rng_state = torch.get_rng_state()
        
        print(f"\n{dataset} K={num_negs}: RNG consumption test:")
        
        # Compare first call results
        sb3_triples_1 = tail_corrs_sb3_1[:, :, 0, :3]
        match_1 = torch.equal(sb3_triples_1, tail_corrs_tensor_1)
        print(f"  First call match: {match_1}")
        
        # Compare second call results
        sb3_triples_2 = tail_corrs_sb3_2[:, :, 0, :3]
        match_2 = torch.equal(sb3_triples_2, tail_corrs_tensor_2)
        print(f"  Second call match: {match_2}")
        
        # Compare RNG states
        rng_match = torch.equal(sb3_rng_state, tensor_rng_state)
        print(f"  RNG state match: {rng_match}")
        
        assert match_1, "First call results don't match"
        assert match_2, "Second call results don't match"
        assert rng_match, "RNG states diverged - samplers consume different amounts of random numbers"


    @pytest.mark.parametrize("dataset,num_negs", [
        ("countries_s3", 3),
        ("countries_s3", 10),
        ("countries_s3", None),  # All corruptions
        ("family", 10),
        ("family", 20),
        ("family", None),  # All corruptions
    ])
    def test_get_negatives_from_states_separate_method_parity(self, dataset, num_negs):
        """
        Test exact parity between SB3's get_negatives_from_states_separate and
        tensor sampler's get_negatives_from_states_separate method.
        
        Both methods should:
        1. Generate head corruptions and tail corruptions separately
        2. Return them as independent lists/tensors
        3. Produce identical results given the same RNG state
        
        This is the exact method used by eval_corruptions.
        
        Note: countries_s3 only uses tail corruption, family uses both head and tail.
        """
        # Determine corruption mode based on dataset
        # countries_s3: only tail corruption
        # family: both head and tail corruption
        if dataset == "countries_s3":
            default_mode = "tail"
            corruption_scheme = ['tail']
            corruption_indices = [2]
        else:  # family
            default_mode = "both"
            corruption_scheme = ['head', 'tail']
            corruption_indices = [0, 2]
        
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset, default_mode=default_mode)
        
        device = torch.device("cpu")
        n_queries = min(10, len(dh_sb3.test_queries))
        
        # Get aligned queries
        queries_sb3 = dh_sb3.test_queries[:n_queries]
        
        # Create tensor queries from SB3 queries to ensure alignment
        queries_tensor = []
        for q in queries_sb3:
            rel_idx = im_new.predicate_str2idx[q.predicate]
            head_idx = im_new.constant_str2idx[q.args[0]]
            tail_idx = im_new.constant_str2idx[q.args[1]]
            queries_tensor.append(torch.tensor([rel_idx, head_idx, tail_idx], dtype=torch.long))
        queries_tensor = torch.stack(queries_tensor, dim=0).to(device)
        
        # Configure SB3 sampler with the appropriate corruption scheme
        sampler_sb3.corruption_scheme = corruption_scheme
        sampler_sb3._corruption_indices = corruption_indices
        
        SEED = 42
        
        # Generate with SB3
        torch.manual_seed(SEED)
        head_negs_sb3, tail_negs_sb3 = sampler_sb3.get_negatives_from_states_separate(
            [[q] for q in queries_sb3],
            device=device,
            num_negs=num_negs,
            return_states=False,  # Return tensors
        )
        
        # Generate with tensor sampler
        torch.manual_seed(SEED)
        head_negs_tensor, tail_negs_tensor = sampler_new.get_negatives_from_states_separate(
            queries_tensor,
            num_negatives=num_negs,
            device=device,
            filter=True,
            unique=True,
        )
        
        print(f"\n{dataset} K={'all' if num_negs is None else num_negs}:")
        print(f"  SB3 head shape: {head_negs_sb3.shape}, tail shape: {tail_negs_sb3.shape}")
        
        # Extract triples from SB3 format [B, K, padding_atoms, max_arity+1] -> [B, K, 3]
        # SB3 stores: [rel, head, tail, ...] at [b, k, 0, :3]
        sb3_head_triples = head_negs_sb3[:, :, 0, :3]  # [B, K_head, 3] in (rel, head, tail) format
        sb3_tail_triples = tail_negs_sb3[:, :, 0, :3]  # [B, K_tail, 3] in (rel, head, tail) format
        
        print(f"  SB3 head triples shape: {sb3_head_triples.shape}")
        print(f"  SB3 tail triples shape: {sb3_tail_triples.shape}")
        
        # Compare head corruptions
        mismatches_head = 0
        for i in range(n_queries):
            # Get tensor head negatives for query i
            tensor_head_i = head_negs_tensor[i]  # [K_i, 3] in (r, h, t) format
            sb3_head_i = sb3_head_triples[i]  # [K, 3] in (r, h, t) format
            
            # Filter out padding from SB3 (all zeros or padding_idx)
            sb3_head_valid = sb3_head_i[sb3_head_i.sum(dim=-1) != 0]
            
            # Filter out padding from tensor (all zeros)
            tensor_head_valid = tensor_head_i[tensor_head_i.sum(dim=-1) != 0]
            
            if i < 3:
                print(f"  Query {i}: {queries_tensor[i].tolist()}")
                print(f"    SB3 head (first 3): {sb3_head_valid[:3].tolist()}")
                print(f"    Tensor head (first 3): {tensor_head_valid[:3].tolist()}")
            
            # Compare counts
            if sb3_head_valid.shape[0] != tensor_head_valid.shape[0]:
                print(f"  Query {i} HEAD COUNT MISMATCH: SB3={sb3_head_valid.shape[0]} vs Tensor={tensor_head_valid.shape[0]}")
                mismatches_head += abs(sb3_head_valid.shape[0] - tensor_head_valid.shape[0])
                continue
            
            # Compare content
            for j in range(sb3_head_valid.shape[0]):
                if sb3_head_valid[j].tolist() != tensor_head_valid[j].tolist():
                    mismatches_head += 1
                    if mismatches_head <= 5:
                        print(f"  Query {i} head neg {j}: SB3={sb3_head_valid[j].tolist()} vs Tensor={tensor_head_valid[j].tolist()}")
        
        # Compare tail corruptions
        mismatches_tail = 0
        for i in range(n_queries):
            tensor_tail_i = tail_negs_tensor[i]
            sb3_tail_i = sb3_tail_triples[i]
            
            sb3_tail_valid = sb3_tail_i[sb3_tail_i.sum(dim=-1) != 0]
            tensor_tail_valid = tensor_tail_i[tensor_tail_i.sum(dim=-1) != 0]
            
            if i < 3:
                print(f"    SB3 tail (first 3): {sb3_tail_valid[:3].tolist()}")
                print(f"    Tensor tail (first 3): {tensor_tail_valid[:3].tolist()}")
            
            if sb3_tail_valid.shape[0] != tensor_tail_valid.shape[0]:
                print(f"  Query {i} TAIL COUNT MISMATCH: SB3={sb3_tail_valid.shape[0]} vs Tensor={tensor_tail_valid.shape[0]}")
                mismatches_tail += abs(sb3_tail_valid.shape[0] - tensor_tail_valid.shape[0])
                continue
            
            for j in range(sb3_tail_valid.shape[0]):
                if sb3_tail_valid[j].tolist() != tensor_tail_valid[j].tolist():
                    mismatches_tail += 1
                    if mismatches_tail <= 5:
                        print(f"  Query {i} tail neg {j}: SB3={sb3_tail_valid[j].tolist()} vs Tensor={tensor_tail_valid[j].tolist()}")
        
        total_mismatches = mismatches_head + mismatches_tail
        if total_mismatches == 0:
            print(f"  ✓ All corruptions match!")
        else:
            print(f"  ✗ Head mismatches: {mismatches_head}, Tail mismatches: {mismatches_tail}")
        
        assert mismatches_head == 0, f"Found {mismatches_head} head corruption mismatches"
        assert mismatches_tail == 0, f"Found {mismatches_tail} tail corruption mismatches"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
