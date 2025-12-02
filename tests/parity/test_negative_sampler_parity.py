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
from sampler_new import Sampler as NewSampler

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


def _create_stacks(args, seed=42):
    """Create both stacks with same seed for comparison."""
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
    
    # Build domain info from SB3's data handler (same as get_sampler does)
    domain2idx = None
    entity2domain = None
    if 'countries' in args.dataset_name or 'ablation' in args.dataset_name:
        if dh_sb3.domain2entity is not None:
            domain2idx = {}
            entity2domain = {}
            for domain, entities in dh_sb3.domain2entity.items():
                indexed_entities = []
                for e in entities:
                    if e in im_sb3.constant_str2idx:
                        idx = im_sb3.constant_str2idx[e]
                        indexed_entities.append(idx)
                        entity2domain[idx] = domain
                if indexed_entities:
                    domain2idx[domain] = indexed_entities
    
    # Create new sampler with same seed
    sampler_new = NewSampler.from_data(
        all_known_triples_idx=dh_new.all_known_triples_idx,
        num_entities=im_new.constant_no,
        num_relations=im_new.predicate_no,
        device=torch.device("cpu"),
        default_mode="tail",
        seed=seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    return dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args


# Module-level cache
_STACK_CACHE = {}


def get_stacks(dataset: str, seed: int = 42):
    """Get or create stacks for a dataset."""
    cache_key = (dataset, seed)
    if cache_key not in _STACK_CACHE:
        args = _base_args(dataset=dataset)
        result = _create_stacks(args, seed=seed)
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
    def test_sampled_negatives_parity(self, dataset):
        """
        Verify 1000 queries with 4 negatives produce same shape, content, and order.
        Both samplers must be called with the same random seed to ensure identical output.
        """
        dh_new, im_new, sampler_new, dh_sb3, im_sb3, sampler_sb3, args = get_stacks(dataset)
        
        # Take up to 1000 queries
        n_queries = min(1000, dh_new.all_known_triples_idx.shape[0])
        queries_new = dh_new.all_known_triples_idx[:n_queries]
        
        # Prepare inputs for both samplers
        pos_hrt = torch.stack([
            queries_new[:, 1],  # head
            queries_new[:, 0],  # relation
            queries_new[:, 2],  # tail
        ], dim=1)
        
        # Generate negatives from new sampler with fixed seed
        SEED = 42
        torch.manual_seed(SEED)
        negatives_new = sampler_new.corrupt(
            queries_new,
            num_negatives=4,
            mode='tail',
            filter=False,  # Don't filter - test raw generation parity
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
        print(f"\n{dataset}: Sampled negatives comparison (first 3 queries, 4 negatives each):")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
