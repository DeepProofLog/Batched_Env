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


def _base_args(n: int = 5):
    return SimpleNamespace(
        dataset_name="countries_s3",
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


def test_data_handlers_materialize_same_shapes():
    args = _base_args()

    # New stack
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
    sampler_new = NewSampler.from_data(
        all_known_triples_idx=dh_new.all_known_triples_idx,
        num_entities=im_new.constant_no,
        num_relations=im_new.predicate_no,
        device=torch.device("cpu"),
        default_mode="tail",
        seed=0,
    )

    # SB3 stack
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
    im_sb3.build_fact_index(dh_sb3.facts)
    sampler_sb3 = get_sb3_sampler(
        data_handler=dh_sb3,
        index_manager=im_sb3,
        corruption_scheme=args.corruption_scheme,
        device=torch.device("cpu"),
    )

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
