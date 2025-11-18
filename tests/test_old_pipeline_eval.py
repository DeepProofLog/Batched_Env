"""Integration test for the TorchRL evaluation pipeline using real components."""

from __future__ import annotations

import sys
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

# Ensure repository root is on sys.path so local imports resolve when running from tests/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_handler import DataHandler
from embeddings import get_embedder
from env import BatchedEnv
from index_manager import IndexManager
from model_eval import evaluate_ranking_metrics, evaluate_policy
from ppo.torchrl_model import create_torchrl_modules
from sampler import Sampler
from unification_engine import UnificationEngine


def _build_eval_components(device: torch.device):
    args = SimpleNamespace(
        dataset_name="countries_s3",
        data_path="data",
        janus_file=None,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        n_eval_queries=None,
        n_valid_negatives=10,  # Use 10 negatives per query for faster testing
        padding_atoms=6,
        padding_states=16,
        batch_size=128,
        max_total_vars=1000000,
        atom_embedder="transe",
        state_embedder="sum",
        constant_embedding_size=32,
        predicate_embedding_size=32,
        atom_embedding_size=32,
        learn_embeddings=True,
        variable_no=64,
        corruption_mode=True,
        corruption_scheme=["both"],
        train_neg_ratio=1,
        seed_run_i=7,
        max_depth=8,
        skip_unary_actions=False,
        reward_type=1,
        end_proof_action=True,
    )
    if args.dataset_name == "countries_s3":
        args.corruption_scheme = ["tail"]
    
    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_eval_queries=args.n_eval_queries,
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

    unification_engine = UnificationEngine.from_index_manager(
        index_manager, 
        stringifier_params=None,
        max_derived_per_state=args.padding_states,
        end_pred_idx=index_manager.end_pred_idx,
        end_proof_action=args.end_proof_action
    )
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
        memory_pruning=True,
        corruption_mode=False,
        device=device,
        end_proof_action=args.end_proof_action,
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
    """Test evaluation pipeline with ranking metrics - simplified version."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")
    components = _build_eval_components(device)
    
    # Use a much smaller test: just 10 queries with 10 negatives each
    query_triples = components.data_handler.get_materialized_split("valid").queries[:10].to(device)

    print(f"Testing with {query_triples.shape[0]} validation queries")

    results = evaluate_ranking_metrics(
        actor=components.actor,
        env=components.env,
        queries=query_triples,
        sampler=components.sampler,
        n_corruptions=10,  # Use fixed 10 negatives
        corruption_modes=['head'],  # Test just one mode
        deterministic=True,
        verbose=False,
    )

    print("\n=== Test Results ===")
    
    # The function returns a dict with aggregated metrics and per_mode breakdown
    # Keys: 'MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'per_mode'
    if 'per_mode' in results:
        for corruption_type in ['head']:
            if corruption_type in results['per_mode']:
                metrics = results['per_mode'][corruption_type]
                print(f"\n{corruption_type.capitalize()} Metrics:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                    if metric_name in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
                        assert 0.0 <= value <= 1.0

    # Overall aggregated metrics
    print(f"\nOverall Metrics:")
    print(f"  MRR: {results['MRR']:.4f}")
    print(f"  Hits@1: {results['Hits@1']:.4f}")
    print(f"  Hits@3: {results['Hits@3']:.4f}")
    print(f"  Hits@10: {results['Hits@10']:.4f}")
    assert 0.0 <= results['MRR'] <= 1.0
    assert 0.0 <= results['Hits@1'] <= 1.0

    # Debug reward sanity: run a tiny evaluation episode batch and ensure rewards are not all zero
    env_batch = int(components.env.batch_size[0]) if isinstance(components.env.batch_size, torch.Size) else int(components.env.batch_size)
    q_debug = min(env_batch, query_triples.shape[0])
    slot_lengths = torch.tensor(
        [1] * q_debug + [0] * (env_batch - q_debug),
        dtype=torch.long,
        device=device,
    )
    flat_depths = torch.zeros(q_debug, dtype=torch.long, device=device)
    flat_labels = torch.ones(q_debug, dtype=torch.long, device=device)
    components.env.set_eval_dataset(
        queries=query_triples[:q_debug],
        labels=flat_labels,
        query_depths=flat_depths,
        per_slot_lengths=slot_lengths,
    )
    debug_out = evaluate_policy(
        actor=components.actor,
        env=components.env,
        target_episodes=slot_lengths.tolist(),
        deterministic=True,
        verbose=0,
    )
    reward_matrix = debug_out["rewards"]
    if isinstance(reward_matrix, torch.Tensor):
        reward_sum = reward_matrix.abs().sum().item()
    else:
        reward_sum = np.abs(reward_matrix).sum()
    assert reward_sum > 0.0, "Evaluation rewards should contain non-zero values"

    print("\n=== Test passed! ===")


if __name__ == '__main__':
    test_eval_pipeline_vectorized()
