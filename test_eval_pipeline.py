"""Integration test for the TorchRL evaluation pipeline using real components."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

from data_handler import DataHandler
from embeddings import get_embedder
from env import BatchedEnv
from index_manager import IndexManager
from model_eval import _evaluate_ranking_metrics
from ppo.ppo_model import create_torchrl_modules
from sampler import Sampler
from unification_engine import UnificationEngine


def _build_eval_components(device: torch.device):
    args = SimpleNamespace(
        dataset_name="wn18rr",
        data_path="data",
        janus_file=None,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        n_train_queries=32,
        n_eval_queries=12,
        n_test_queries=12,
        padding_atoms=6,
        padding_states=16,
        batch_size=4,
        max_total_vars=256,
        atom_embedder="transe",
        state_embedder="sum",
        constant_embedding_size=32,
        predicate_embedding_size=32,
        atom_embedding_size=32,
        learn_embeddings=True,
        variable_no=64,
        corruption_mode=True,
        corruption_scheme=["head", "tail"],
        train_neg_ratio=0.0,
        seed_run_i=7,
        max_depth=12,
        skip_unary_actions=True,
        reward_type=1,
    )

    data_handler = DataHandler(
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
    )

    index_manager = IndexManager(
        constants=data_handler.constants,
        predicates=data_handler.predicates,
        max_total_runtime_vars=args.max_total_vars,
        max_arity=data_handler.max_arity,
        padding_atoms=args.padding_atoms,
        device=device,
    )
    data_handler.materialize_indices(im=index_manager, device=device)

    sampler = Sampler.from_data(
        all_known_triples_idx=data_handler.all_known_triples_idx,
        num_entities=index_manager.constant_no,
        num_relations=index_manager.predicate_no,
        device=device,
        default_mode="both",
        seed=args.seed_run_i,
    )

    embedder_getter = get_embedder(
        args=args,
        data_handler=data_handler,
        constant_no=index_manager.constant_no,
        predicate_no=index_manager.predicate_no,
        runtime_var_end_index=index_manager.runtime_var_end_index,
        constant_str2idx=index_manager.constant_str2idx,
        predicate_str2idx=index_manager.predicate_str2idx,
        constant_images_no=getattr(index_manager, "constant_images_no", 0),
        device=device,
    )
    embedder = embedder_getter.embedder
    embed_dim = getattr(embedder, "embed_dim", getattr(embedder, "embedding_dim", args.atom_embedding_size))

    unification_engine = UnificationEngine.from_index_manager(index_manager)
    valid_split = data_handler.get_materialized_split("valid")

    batch_limit = min(valid_split.queries.shape[0], args.batch_size * 3)

    queries = valid_split.queries[:batch_limit].to(device)
    labels = valid_split.labels[:batch_limit].to(device)
    depths = valid_split.depths[:batch_limit].to(device)

    eval_env = BatchedEnv(
        batch_size=args.batch_size,
        queries=queries,
        labels=labels,
        query_depths=depths,
        unification_engine=unification_engine,
        sampler=sampler,
        mode="eval",
        max_arity=index_manager.max_arity,
        padding_idx=index_manager.padding_idx,
        runtime_var_start_index=index_manager.runtime_var_start_index,
        total_vocab_size=index_manager.total_vocab_size,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=index_manager.true_pred_idx,
        false_pred_idx=index_manager.false_pred_idx,
        end_pred_idx=index_manager.end_pred_idx,
        skip_unary_actions=args.skip_unary_actions,
        reward_type=args.reward_type,
        max_depth=args.max_depth,
        memory_pruning=False,
        corruption_mode=False,
        device=device,
    )

    actor, _ = create_torchrl_modules(
        embedder=embedder,
        num_actions=args.padding_states,
        embed_dim=embed_dim,
        hidden_dim=64,
        num_layers=2,
        dropout_prob=0.0,
        device=device,
        enable_kge_action=False,
        kge_inference_engine=None,
        index_manager=index_manager,
    )
    actor.to(device)

    return SimpleNamespace(
        actor=actor,
        env=eval_env,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        data_handler=data_handler,
    )


def test_eval_pipeline_vectorized():
    """Test evaluation pipeline with ranking metrics."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")
    components = _build_eval_components(device)

    valid_split = components.data_handler.get_materialized_split("valid")
    n_test_queries = min(8, len(valid_split))
    test_queries_idx = valid_split.queries[:n_test_queries]
    query_triples = test_queries_idx[:, 0, :3].clone()

    print(f"Testing with {n_test_queries} validation queries")
    n_negatives = 5

    results = _evaluate_ranking_metrics(
        actor=components.actor,
        env=components.env,
        queries=query_triples,
        sampler=components.sampler,
        n_corruptions=n_negatives,
        corruption_modes=['head', 'tail'],
        deterministic=True,
        verbose=True,
        info_callback=None,
    )

    print("\n=== Test Results ===")
    assert 'head_metrics' in results
    assert 'tail_metrics' in results

    for corruption_type in ['head', 'tail']:
        metrics_key = f'{corruption_type}_metrics'
        if metrics_key in results:
            metrics = results[metrics_key]
            print(f"\n{corruption_type.capitalize()} Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
                if metric_name in ['mrr', 'h1', 'h3', 'h10']:
                    assert 0.0 <= value <= 1.0

    if 'mrr_mean' in results:
        print(f"\nOverall Metrics:")
        print(f"  MRR: {results['mrr_mean']:.4f}")
        print(f"  Hits@1: {results['h1_mean']:.4f}")
        print(f"  Hits@3: {results['h3_mean']:.4f}")
        print(f"  Hits@10: {results['h10_mean']:.4f}")
        assert 0.0 <= results['mrr_mean'] <= 1.0
        assert 0.0 <= results['h1_mean'] <= 1.0

    print("\n=== Test passed! ===")


if __name__ == '__main__':
    test_eval_pipeline_vectorized()
