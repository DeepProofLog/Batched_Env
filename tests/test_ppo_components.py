import sys
import os
import torch
from types import SimpleNamespace

# Ensure repository root is on sys.path so local imports resolve when running from tests/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import DataHandler
from index_manager import IndexManager
from neg_sampling import get_sampler


def test_negative_sampler_with_wn18rr():
    """Build a minimal DataHandler/IndexManager for wn18rr and exercise the sampler."""
    args = SimpleNamespace(
        dataset_name="wn18rr",
        data_path="data",
        janus_file="wn18rr.pl",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="wn18rr.pl",
        n_train_queries=10,
        n_eval_queries=2,
        n_test_queries=2,
        corruption_mode=None,
    )

    # Load a small slice of the dataset (DataHandler supports limiting via constructor args)
    dh = DataHandler(
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
    )

    # Create index manager (small runtime vars and padding to keep tensors tiny)
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=20,
        rules=dh.rules,
        padding_atoms=6,
        max_arity=dh.max_arity,
        device=torch.device("cpu"),
    )
    im.build_fact_index(dh.facts)

    sampler = get_sampler(data_handler=dh, index_manager=im, corruption_scheme=["tail"], device=torch.device("cpu"))

    # Build a tiny batch of positive sub-indices from first few training queries
    batch_queries = dh.train_queries[:2]
    subs = torch.stack([im.get_atom_sub_index([q]) for q in batch_queries], dim=0).to(torch.int32)

    # Generate sampled negatives (2 per positive)
    negs = sampler.get_negatives(subs, padding_atoms=im.padding_atoms, max_arity=im.max_arity, device=torch.device("cpu"), num_negs=2)

    print("subs.shape:", subs.shape)
    print("negs.shape:", negs.shape)

    assert negs.dim() == 4
    assert negs.size(0) == subs.size(0)


if __name__ == '__main__':
    test_negative_sampler_with_wn18rr()
