"""
TorchRL-based training script for Neural-guided Grounding.

This version is fully aligned with the refactored pipeline (DataHandler,
IndexManager, BatchedEnv, Sampler, TorchRL PPO stack).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from callbacks import (
    EvaluationCallback,
    RolloutProgressCallback,
    TorchRLCallbackManager,
    TrainingMetricsCallback,
)
from data_handler import DataHandler
from embeddings import get_embedder
from env import BatchedEnv
from index_manager import IndexManager
from model_eval import evaluate_ranking_metrics
from ppo.ppo import PPO
from ppo.model import create_actor_critic
from sampler import Sampler
from unification_engine import UnificationEngine
from utils import (
    _freeze_dropout_layernorm,
    _set_seeds,
    _warn_non_reproducible,
    get_device,
    print_eval_info,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install tensorboard for logging.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_corruption_mode(corruption_scheme: Optional[list[str]]) -> str:
    scheme = corruption_scheme or ["head", "tail"]
    has_head = "head" in scheme
    has_tail = "tail" in scheme
    if has_head and has_tail:
        return "both"
    if has_head:
        return "head"
    if has_tail:
        return "tail"
    return "both"


def _materialize_data_components(
    args: Any,
    device: torch.device,
) -> Tuple[DataHandler, IndexManager, Sampler, Any, UnificationEngine, dict]:
    """Build dataset, indices, sampler, embedder, and unification helpers."""
    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=getattr(args, "janus_file", None),
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        train_depth=getattr(args, "train_depth", None),
        valid_depth=getattr(args, "valid_depth", None),
        test_depth=getattr(args, "test_depth", None),
        prob_facts=getattr(args, "prob_facts", False),
        topk_facts=getattr(args, "topk_facts", None),
        topk_facts_threshold=getattr(args, "topk_facts_threshold", None),
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
        default_mode=_default_corruption_mode(args.corruption_scheme),
        seed=args.seed_run_i,
        domain_heads=getattr(data_handler, "allowed_heads_per_rel", None) or None,
        domain_tails=getattr(data_handler, "allowed_tails_per_rel", None) or None,
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
    if hasattr(embedder, "to"):
        embedder = embedder.to(device)

    stringifier_params = index_manager.get_stringifier_params()
    # Get end_proof_action from args and pass to engine
    end_proof_action = getattr(args, "end_proof_action", False)
    end_pred_idx = index_manager.end_pred_idx if end_proof_action else None
    unification_engine = UnificationEngine.from_index_manager(
        index_manager, 
        stringifier_params=stringifier_params,
        end_pred_idx=end_pred_idx,
        end_proof_action=end_proof_action,
        max_derived_per_state=args.padding_states
    )

    return data_handler, index_manager, sampler, embedder, unification_engine, stringifier_params


def _make_env_from_split(
    split,
    *,
    mode: str,
    batch_size: int,
    args: Any,
    index_manager: IndexManager,
    sampler: Sampler,
    unification_engine: UnificationEngine,
    stringifier_params: dict,
    device: torch.device,
) -> BatchedEnv:
    """Instantiate a BatchedEnv for a specific dataset split."""
    env = BatchedEnv(
        batch_size=int(batch_size),
        queries=split.queries,
        labels=split.labels,
        query_depths=split.depths,
        unification_engine=unification_engine,
        sampler=sampler,
        mode=mode,
        max_arity=index_manager.max_arity,
        padding_idx=index_manager.padding_idx,
        runtime_var_start_index=index_manager.runtime_var_start_index,
        total_vocab_size=index_manager.total_vocab_size,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=index_manager.true_pred_idx,
        false_pred_idx=index_manager.false_pred_idx,
        end_pred_idx=index_manager.end_pred_idx if getattr(args, "end_proof_action", False) else None,
        stringifier_params=stringifier_params,
        corruption_mode=args.corruption_mode if mode == "train" else False,
        corruption_scheme=args.corruption_scheme,
        train_neg_ratio=args.train_neg_ratio if mode == "train" else 0.0,
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning if mode == "train" else False,
        eval_pruning=getattr(args, "eval_pruning", False),
        end_proof_action=getattr(args, "end_proof_action", False),
        skip_unary_actions=args.skip_unary_actions,
        reward_type=args.reward_type,
        verbose=getattr(args, "verbose_env", 0),
        prover_verbose=getattr(args, "verbose_prover", 0),
        device=device,
    )
    env.index_manager = index_manager  # For diagnostics / evaluation tooling
    return env


def _extract_query_triples(split, limit: Optional[int]) -> torch.Tensor:
    if split.queries.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=split.queries.device)
    triples = split.queries[:, 0, :3]
    if limit is not None:
        limit = min(int(limit), triples.shape[0])
        triples = triples[:limit]
    return triples


def _evaluate_split(
    policy: nn.Module,
    env: Optional[BatchedEnv],
    sampler: Sampler,
    split,
    *,
    n_queries: Optional[int],
    n_corruptions: Optional[int],
    corruption_scheme: list[str],
    verbose: bool = False,
) -> dict:
    if env is None or split is None or split.queries.shape[0] == 0:
        return {}
    if not n_corruptions or n_corruptions <= 0:
        return {}

    queries = _extract_query_triples(split, n_queries)
    if queries.shape[0] == 0:
        return {}

    device = getattr(env, "_device", torch.device("cpu"))
    queries = queries.to(device)

    metrics = evaluate_ranking_metrics(
        actor=policy.policy_net,
        env=env,
        queries=queries,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=corruption_scheme,
        deterministic=True,
        verbose=verbose,
    )
    
    # Transform keys to match expected format (lowercase with _mean suffix)
    # evaluate_ranking_metrics returns: MRR, Hits@1, Hits@3, Hits@10, per_mode
    transformed_metrics = {
        'mrr_mean': metrics.get('MRR', 0.0),
        'hits1_mean': metrics.get('Hits@1', 0.0),
        'hits3_mean': metrics.get('Hits@3', 0.0),
        'hits10_mean': metrics.get('Hits@10', 0.0),
    }
    
    # Add per-mode metrics if available
    if 'per_mode' in metrics:
        per_mode = metrics['per_mode']
        for mode, mode_metrics in per_mode.items():
            transformed_metrics[f'{mode}_mrr_mean'] = mode_metrics.get('MRR', 0.0)
            transformed_metrics[f'{mode}_hits1_mean'] = mode_metrics.get('Hits@1', 0.0)
            transformed_metrics[f'{mode}_hits3_mean'] = mode_metrics.get('Hits@3', 0.0)
            transformed_metrics[f'{mode}_hits10_mean'] = mode_metrics.get('Hits@10', 0.0)
    
    return transformed_metrics


def _evaluate(
    args: Any,
    policy: nn.Module,
    sampler: Sampler,
    data_handler: DataHandler,
    eval_env: Optional[BatchedEnv],
    test_env: Optional[BatchedEnv],
) -> Tuple[dict, dict, dict]:
    policy.eval()

    metrics_valid = _evaluate_split(
        policy,
        eval_env,
        sampler,
        data_handler.get_materialized_split("valid"),
        n_queries=args.n_eval_queries,
        n_corruptions=args.eval_neg_samples,
        corruption_scheme=args.corruption_scheme,
        verbose=getattr(args, "depth_info", False),
    )
    if metrics_valid:
        print_eval_info("VALID", metrics_valid)

    metrics_test = _evaluate_split(
        policy,
        test_env,
        sampler,
        data_handler.get_materialized_split("test"),
        n_queries=args.n_test_queries,
        n_corruptions=args.test_neg_samples or args.eval_neg_samples,
        corruption_scheme=args.corruption_scheme,
        verbose=getattr(args, "depth_info", False),
    )
    if metrics_test:
        print_eval_info("TEST", metrics_test)

    metrics_train = {key: 0.0 for key in metrics_valid.keys()} if metrics_valid else {}
    return metrics_train, metrics_valid, metrics_test


def _model_dir(args: Any, date: str) -> Path:
    return Path(args.models_path) / args.run_signature / f"seed_{args.seed_run_i}" / date


def _resolve_ckpt_to_load(root: Path, restore_best: bool) -> Optional[Path]:
    if not root.exists():
        return None
    keyword = "best_eval" if restore_best else "last_epoch"
    matches = sorted(root.glob(f"*{keyword}*.pt"))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Simple logger that writes to disk and (optionally) TensorBoard."""

    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"

        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        if self.use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        self.history = {"train": [], "eval": []}

    def log_scalar(self, name: str, value: float, step: int, category: str = "train") -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f"{category}/{name}", value, step)
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(f"[Step {step}] {category}/{name}: {value:.6f}\n")

    def log_dict(self, metrics: dict, step: int, category: str = "train", prefix: str = "") -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                full_name = f"{prefix}{key}" if prefix else key
                self.log_scalar(full_name, float(value), step, category)

    def log_training_step(
        self,
        global_step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
    ) -> None:
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
        self.log_dict(metrics, global_step, category="train")

    def log_evaluation(self, iteration: int, global_step: int, metrics: dict, prefix: str = "eval") -> None:
        self.log_dict(metrics, global_step, category="eval", prefix=f"{prefix}/")
        self.history["eval"].append(
            {
                "iteration": iteration,
                "step": global_step,
                "metrics": metrics.copy(),
            }
        )

    def save_history(self) -> None:
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        self.save_history()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    args: Any,
    policy: nn.Module,
    train_env: BatchedEnv,
    eval_env: Optional[BatchedEnv],
    sampler: Sampler,
    data_handler: DataHandler,
    model_path: Path,
    device: torch.device,
) -> nn.Module:
    if args.timesteps_train <= 0:
        print("No training requested (timesteps_train <= 0). Skipping optimization.")
        return policy

    logger = TrainingLogger(model_path, use_tensorboard=True)

    rollout_callback = RolloutProgressCallback(
        total_steps=args.n_steps * args.batch_size_env,
        n_envs=args.batch_size_env,
        update_interval=25,
        verbose=True,
    )

    eval_callback = EvaluationCallback(
        eval_env=eval_env,
        sampler=sampler,
        eval_data=data_handler.valid_queries,
        eval_data_depths=getattr(data_handler, "valid_depths", None),
        n_corruptions=args.eval_neg_samples,
        eval_freq=1,
        best_metric=getattr(args, "eval_best_metric", "mrr_mean"),
        save_path=model_path,
        verbose=True,
        collect_detailed=getattr(args, "depth_info", False),
        verbose_cb=getattr(args, "verbose_cb", False),
    )

    train_callback = TrainingMetricsCallback(
        log_interval=1,
        verbose=True,
        verbose_cb=getattr(args, "verbose_cb", False),
        collect_detailed=getattr(args, "depth_info", False),
    )

    callback_manager = TorchRLCallbackManager(
        rollout_callback=rollout_callback,
        eval_callback=eval_callback if eval_env is not None else None,
        train_callback=train_callback,
    )
    callback_manager.on_training_start()

    use_amp = getattr(args, "use_amp", device.type == "cuda")
    setattr(args, "use_amp", use_amp)

    # Set AMP on policy
    policy.use_amp = use_amp

    # Create PPO agent with SB3-style implementation
    ppo_agent = PPO(
        policy=policy,
        env=train_env,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        clip_range_vf=None,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=args.lr,
        device=device,
        verbose=1,
    )

    # Train the agent
    ppo_agent.learn(
        total_timesteps=args.timesteps_train,
        log_interval=1,
        reset_num_timesteps=True,
    )

    logger.close()

    return policy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args, log_filename, use_logger, use_WB, WB_path, date):
    del log_filename, use_logger, use_WB, WB_path  # Unused legacy parameters

    _warn_non_reproducible(args)
    _set_seeds(args.seed_run_i)

    args.corruption_scheme = list(args.corruption_scheme or ["head", "tail"])

    device = get_device(args.device)
    print(
        f"Device: {device}. CUDA available: {torch.cuda.is_available()}, "
        f"Device count: {torch.cuda.device_count()}"
    )

    (
        data_handler,
        index_manager,
        sampler,
        embedder,
        unification_engine,
        stringifier_params,
    ) = _materialize_data_components(args, device)

    train_split = data_handler.get_materialized_split("train")
    valid_split = data_handler.get_materialized_split("valid")
    test_split = data_handler.get_materialized_split("test")

    train_env = _make_env_from_split(
        train_split,
        mode="train",
        batch_size=args.batch_size_env,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        unification_engine=unification_engine,
        stringifier_params=stringifier_params,
        device=device,
    )
    eval_env = _make_env_from_split(
        valid_split,
        mode="eval",
        batch_size=args.batch_size_env_eval,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        unification_engine=unification_engine,
        stringifier_params=stringifier_params,
        device=device,
    )
    test_env = _make_env_from_split(
        test_split,
        mode="eval",
        batch_size=args.batch_size_env_eval,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        unification_engine=unification_engine,
        stringifier_params=stringifier_params,
        device=device,
    )

    embed_dim = getattr(embedder, "embed_dim", getattr(embedder, "atom_embedding_size", args.atom_embedding_size))
    
    use_amp = getattr(args, "use_amp", device.type == "cuda")
    use_compile = getattr(args, "use_compile", False)
    
    # Create actor-critic policy using PPO SB3-style implementation
    policy = create_actor_critic(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=8,
        dropout_prob=0.2,
        device=device,
        use_compile=use_compile,
        use_amp=use_amp,
    )

    # Note: PPO creates its own optimizer internally, no need to create one here
    # Wrap policy with DataParallel for multi-GPU training if needed
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        policy = torch.nn.DataParallel(policy)

    model_path = _model_dir(args, date)
    model_path.mkdir(parents=True, exist_ok=True)

    if args.load_model:
        ckpt = _resolve_ckpt_to_load(model_path, restore_best=args.restore_best_val_model)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {model_path}")
        # Load checkpoint into policy
        checkpoint = torch.load(ckpt, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model from {ckpt}")
    else:
        policy = train(
            args,
            policy,
            train_env,
            eval_env,
            sampler,
            data_handler,
            model_path,
            device,
        )

    policy.apply(_freeze_dropout_layernorm)
    policy.eval()

    metrics_train, metrics_valid, metrics_test = _evaluate(
        args,
        policy,
        sampler,
        data_handler,
        eval_env,
        test_env,
    )

    return metrics_train, metrics_valid, metrics_test
