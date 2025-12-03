"""
Simplified training script for comparing batched vs sb3 implementation.

This version closely mimics sb3/train.py but uses the batched environment and
TorchRL components. It includes verbose logging to compare intermediate values.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict

from data_handler import DataHandler
from callbacks import print_formatted_metrics
from embeddings import get_embedder
from env import BatchedEnv as BatchedEnv
from index_manager import IndexManager
from sampler import Sampler
from unification import UnificationEngine
from model import create_actor_critic  # Use new clean model
from ppo import PPO  # Use new clean PPO
from model_eval import eval_corruptions

from utils.utils import (
    _freeze_dropout_layernorm,
    _set_seeds,
    _warn_non_reproducible,
    get_device,
    is_variable,
    print_eval_info,
)
from utils.trace_utils import TraceRecorder, _to_python_scalar


def _default_corruption_mode(corruption_scheme):
    """Convert corruption scheme list to mode string."""
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


def _materialize_data_components(args, device, embed_device=None):
    """Build dataset, indices, sampler, embedder, and unification helpers."""
    embed_device = embed_device or device
    print("\n[DATA] Loading data handler...")
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
        load_depth_info=getattr(args, "load_depth_info", True),
        prob_facts=getattr(args, "prob_facts", False),
        topk_facts=getattr(args, "topk_facts", None),
        topk_facts_threshold=getattr(args, "topk_facts_threshold", None),
        corruption_mode=args.corruption_mode,
        filter_queries_by_rules=getattr(args, "filter_queries_by_rules", True),
    )
    print(f"[DATA] Loaded {len(data_handler.train_queries)} train, {len(data_handler.valid_queries)} valid, {len(data_handler.test_queries)} test queries")
    print(f"DEBUG: Batched DataHandler n_train_queries={args.n_train_queries}")
    print(f"DEBUG: Batched DataHandler predicates ({len(data_handler.predicates)}): {sorted(list(data_handler.predicates))}")

    print("\n[INDEX] Building index manager...")
    index_manager = IndexManager(
        constants=data_handler.constants,
        predicates=data_handler.predicates,
        max_total_runtime_vars=args.max_total_vars,
        max_arity=data_handler.max_arity,
        padding_atoms=args.padding_atoms,
        device=device,
        include_kge_predicates=getattr(args, 'kge_action', False),
    )
    print(f"[INDEX] Constants: {index_manager.constant_no}, Predicates: {index_manager.predicate_no}")

    print("\n[INDEX] Materializing indices...")
    data_handler.materialize_indices(im=index_manager, device=device)
    # Align runtime variable start with SB3: do not offset by head variables
    head_vars = {arg for rule in data_handler.rules for arg in getattr(rule.head, "args", ()) if is_variable(arg)}
    index_manager.adjust_runtime_start_for_head_vars(len(head_vars))
    print(f"[INDEX] Runtime var start: {index_manager.runtime_var_start_index} (head_vars={len(head_vars)})")

    print("\n[SAMPLER] Creating negative sampler...")
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
    print(f"[SAMPLER] Initialized with mode: {sampler.default_mode}")

    print("\n[EMBEDDER] Creating embedder...")
    # Reseed to align embedder init with SB3 regardless of prior torch usage
    torch.manual_seed(args.seed_run_i)
    embedder_getter = get_embedder(
        args=args,
        data_handler=data_handler,
        constant_no=index_manager.constant_no,
        predicate_no=index_manager.predicate_no,
        runtime_var_end_index=index_manager.runtime_var_end_index,
        constant_str2idx=index_manager.constant_str2idx,
        predicate_str2idx=index_manager.predicate_str2idx,
        constant_images_no=getattr(index_manager, "constant_images_no", 0),
        device=embed_device,
    )
    embedder = embedder_getter.embedder
    if hasattr(embedder, "to"):
        embedder = embedder.to(embed_device)
    print(f"[EMBEDDER] Embedding dim: {embedder.embedding_dim}")

    stringifier_params = index_manager.get_stringifier_params()
    end_proof_action = getattr(args, "end_proof_action", False)
    end_pred_idx = index_manager.end_pred_idx if end_proof_action else None
    
    print("\n[ENGINE] Creating unification engine...")
    unification_engine = UnificationEngine.from_index_manager(
        index_manager,
        stringifier_params=stringifier_params,
        end_pred_idx=end_pred_idx,
        end_proof_action=end_proof_action,
        max_derived_per_state=args.padding_states,
        sort_states=True
    )
    print(f"[ENGINE] Max derived per state: {args.padding_states}")

    return data_handler, index_manager, sampler, embedder, unification_engine, stringifier_params


def _format_eval_value(val: Any, key: str) -> str:
    """Format eval values to match SB3-style table output."""
    if isinstance(val, str):
        return val
    try:
        fval = float(val)
        if "timesteps" in key:
            return f"{int(fval)}"
        return f"{fval:.3f}"
    except Exception:
        return str(val)


def _print_eval_table(metrics: Dict[str, Any], total_timesteps: Any) -> None:
    """Print eval metrics in the exact order/format shown in SB3 logs."""
    if not metrics:
        return
    # Desired order (remaining metrics will be appended later)
    ordered_keys = [
        "_mrr",
        "auc_pr",
        "len_neg",
        "len_pos",
        "length mean +/- std",
        "proven_d_1_pos",
        "proven_d_2_pos",
        "proven_d_3_pos",
        "proven_d_4_pos",
        "proven_d_none_neg",
        "proven_neg",
        "proven_pos",
        "reward_d_1_pos",
        "reward_d_2_pos",
        "reward_d_3_pos",
        "reward_d_4_pos",
        "reward_label_neg",
        "reward_label_pos",
        "reward_overall",
        "success_rate",
        "total_timesteps",
    ]
    def _row(key: str, val: Any) -> None:
        print(f"|    {key:<20} | {_format_eval_value(val, key):<22} |")

    print("-------------------------------------------------")
    print(f"| {'eval/':<23} | {'':<22} |")
    for key in ordered_keys:
        value = metrics.get(key)
        if key == "total_timesteps":
            value = total_timesteps if value is None else value
        if value is not None:
            _row(key, value)
    print("-------------------------------------------------")


def _make_eval_callback(
    args,
    policy: nn.Module,
    eval_env: Optional[BatchedEnv],
    sampler: Sampler,
    data_handler: DataHandler,
):
    """
    Build a callback that mirrors SB3's periodic evaluation output.
    
    Runs evaluation every eval_freq timesteps and prints a formatted table with
    depth/label breakdowns.
    """
    if eval_env is None:
        return None
    eval_freq = getattr(args, "eval_freq", None) or (args.n_steps * args.batch_size_env)
    last_eval_step = 0

    def _print_eval_table(metrics: Dict[str, Any], total_timesteps: int):
        if not metrics:
            return
        metrics_section = {
            k: v
            for k, v in metrics.items()
            if k.startswith(("len_", "reward_", "length", "proven_"))
            or k == "success_rate"
            or isinstance(v, str)
        }
        keep_extra = {
            "_mrr",
            "mrr_mean",
            "hits1_mean",
            "hits3_mean",
            "hits10_mean",
            "auc_pr",
            "total_timesteps",
            "tail_mrr_mean",
            "tail_hits1_mean",
            "tail_hits3_mean",
            "tail_hits10_mean",
            "head_mrr_mean",
            "head_hits1_mean",
            "head_hits3_mean",
            "head_hits10_mean",
        }
        extra_section = {k: v for k, v in metrics.items() if k in keep_extra}
        if "total_timesteps" not in extra_section:
            extra_section["total_timesteps"] = total_timesteps
        print_formatted_metrics(metrics=metrics_section, prefix="eval", extra_metrics=extra_section)

    def _callback(locals_, globals_):
        nonlocal last_eval_step
        total_steps = (
            locals_.get("total_steps_done")
            or locals_.get("num_timesteps")
            or locals_.get("total_timesteps")
            or 0
        )
        if total_steps is None or total_steps <= 0:
            return True
        if (total_steps - last_eval_step) < eval_freq:
            return True

        last_eval_step = total_steps
        split = data_handler.get_materialized_split("valid")
        start = time.time()
        print("---------------evaluation started---------------")
        metrics_valid = _evaluate_split(
            policy,
            eval_env,
            sampler,
            split,
            n_queries=args.n_eval_queries,
            n_corruptions=args.eval_neg_samples,
            corruption_scheme=args.corruption_scheme,
            verbose=getattr(args, "depth_info", False),
            total_timesteps=total_steps,
        )
        _print_eval_table(metrics_valid, total_steps)
        print(f"---------------evaluation finished---------------  took {time.time() - start:.2f} seconds")
        return True

    return _callback


def _make_env_from_split(split, *, mode, batch_size, args, index_manager, sampler, unification_engine, stringifier_params, device):
    """Create environment from split."""
    verbose = getattr(args, "verbose_env", 0) if mode == "train" else 0
    
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
        memory_pruning=args.memory_pruning,
        eval_pruning=getattr(args, "eval_pruning", False),
        end_proof_action=getattr(args, "end_proof_action", False),
        skip_unary_actions=args.skip_unary_actions,
        reward_type=args.reward_type,
        verbose=verbose,
        prover_verbose=getattr(args, "verbose_prover", 0),
        device=device,
        use_exact_memory=getattr(args, "use_exact_memory", False),
    )
    env.index_manager = index_manager  # For diagnostics / evaluation tooling
    return env


def _model_dir(args, date):
    return Path(args.models_path) / args.run_signature / f"seed_{args.seed_run_i}"


def _resolve_ckpt_to_load(root, restore_best):
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    keyword = "best_eval" if restore_best else "last_epoch"
    candidates = sorted([p for p in root.glob(f"*{keyword}*.zip")])
    return candidates[-1] if candidates else None


def collect_rollouts_verbose(
    policy,
    env,
    rollout_buffer,
    n_steps,
    gamma,
    device,
    verbose=False,
):
    """
    Collect rollouts with verbose logging to match sb3 behavior.
    
    This mimics sb3's collect_rollouts but with detailed logging for comparison.
    """
    print(f"\n[ROLLOUT] Starting rollout collection for {n_steps} steps...")
    
    policy.eval()
    rollout_buffer.reset()
    
    n_collected = 0
    total_reward = 0.0
    episode_count = 0
    episode_lengths = []
    episode_rewards = []
    
    current_obs = env.reset()
    current_episode_reward = torch.zeros(env.batch_size, device=device)
    current_episode_length = torch.zeros(env.batch_size, dtype=torch.long, device=device)
    episode_starts = torch.ones(env.batch_size, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        while n_collected < n_steps:
            if verbose and n_collected % max(n_steps // 5, 1) == 0:
                print(f"[ROLLOUT] Step {n_collected}/{n_steps}")
            
            # Get action from policy
            actions, values, log_probs = policy(current_obs, deterministic=True)
            
            # Step environment - need to wrap action in TensorDict
            action_td = TensorDict({"action": actions}, batch_size=current_obs.batch_size)
            # IMPORTANT: Use step_and_maybe_reset to automatically reset done environments
            step_result, next_obs = env.step_and_maybe_reset(action_td)
            
            # step_and_maybe_reset returns (step_result, next_obs) where next_obs
            # already has reset observations for done environments
            # step_result should KEEP the done=True flag for episode tracking
            
            # Extract done/reward from the correct location
            # step_result has structure: {'action': ..., 'next': {...done, reward, ...}}
            if "next" in step_result.keys():
                step_info = step_result["next"]
            else:
                step_info = step_result
            
            rewards = step_info.get("reward", torch.zeros(env.batch_size_int, device=device))
            dones = step_info.get("done", torch.zeros(env.batch_size_int, dtype=torch.bool, device=device))
            
            # Squeeze done and rewards to ensure they're [batch_size] not [batch_size, 1]
            if rewards.dim() > 1:
                rewards = rewards.squeeze(-1)
            if dones.dim() > 1:
                dones = dones.squeeze(-1)
            
            infos = {}
            
            # Store transition
            rollout_buffer.add(
                obs=current_obs,
                action=actions,
                reward=rewards,
                episode_start=episode_starts,
                value=values,
                log_prob=log_probs
            )

            # Trace recording for parity debugging
            if isinstance(trace_recorder, TraceRecorder):
                sub_index = current_obs.get("sub_index")
                derived_sub_indices = current_obs.get("derived_sub_indices")
                action_mask = current_obs.get("action_mask")
                trace_recorder.log_step(
                    phase="train",
                    iteration=0,
                    step=n_collected,
                    env=0,
                    action=_to_python_scalar(actions[0]),
                    reward=_to_python_scalar(rewards[0]),
                    done=_to_python_scalar(dones[0]),
                    length=int(current_episode_length[0].item() + 1),
                    episode=int(episode_count),
                    value=_to_python_scalar(values[0]),
                    log_prob=_to_python_scalar(log_probs[0]),
                    sub_index=_to_python_scalar(sub_index),
                    derived_sub_indices=_to_python_scalar(derived_sub_indices),
                    action_mask=_to_python_scalar(action_mask),
                )
            
            # Update statistics
            current_episode_reward += rewards
            current_episode_length += 1
            n_collected += 1
            
            # Check for episode ends
            if dones.any():
                for idx in torch.where(dones)[0]:
                    ep_reward = current_episode_reward[idx].item()
                    ep_length = current_episode_length[idx].item()
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episode_count += 1
                    
                    if verbose:
                        print(f"[ROLLOUT]   Episode {episode_count} done: reward={ep_reward:.3f}, length={ep_length}")
                    
                    # Reset episode stats
                    current_episode_reward[idx] = 0.0
                    current_episode_length[idx] = 0
                
                # Mark episode starts for next step
                episode_starts = dones.float()
            else:
                episode_starts = torch.zeros(env.batch_size, dtype=torch.float32, device=device)
            
            current_obs = next_obs
    
    # Compute advantages and returns using last value for bootstrapping
    with torch.no_grad():
        _, last_values, _ = policy(current_obs)
    
    # Ensure dones has correct shape (n_envs,)
    if dones.dim() > 1:
        dones = dones.squeeze(-1)
    
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
    
    # Debug: Check buffer state
    print(f"[ROLLOUT] Buffer state: pos={rollout_buffer.pos}, full={rollout_buffer.full}, buffer_size={rollout_buffer.buffer_size}")
    
    # Print rollout statistics
    print(f"\n[ROLLOUT] Completed {n_collected} steps:")
    print(f"[ROLLOUT]   Episodes completed: {episode_count}")
    if episode_rewards:
        print(f"[ROLLOUT]   Mean episode reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"[ROLLOUT]   Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"[ROLLOUT]   Reward range: [{min(episode_rewards):.3f}, {max(episode_rewards):.3f}]")
    
    return {
        'episode_count': episode_count,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }


def train_ppo_verbose(
    args,
    policy,
    train_env,
    eval_env,
    sampler,
    data_handler,
    model_path,
    device,
    trace_recorder: TraceRecorder = None,
):
    """
    Training loop that uses PPO's learn() method - matches SB3 exactly.
    """
    print("\n" + "="*80)
    print("[TRAIN] Initializing PPO training")
    print("="*80)
    
    # Create PPO optimizer
    ppo = PPO(
        policy=policy,
        env=train_env,
        n_steps=args.n_steps,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        device=device,
        verbose=True,  # Enable verbose logging
        trace_recorder=trace_recorder,
        trace_dir=getattr(args, "trace_dir", None),
        trace_prefix="batched",
    )
    
    total_timesteps = args.timesteps_train
    
    print(f"[TRAIN] Total timesteps: {total_timesteps}")
    print(f"[TRAIN] Rollout steps: {args.n_steps}")
    print(f"[TRAIN] Batch size (n_envs): {args.batch_size_env}")
    print(f"[TRAIN] Learning rate: {args.lr}")
    print(f"[TRAIN] N epochs: {args.n_epochs}")
    print(f"[TRAIN] Gamma: {args.gamma}")
    
    eval_callback = _make_eval_callback(args, policy, eval_env, sampler, data_handler)
    # Use the learn method which handles both rollout and training
    ppo.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    print(f"\n{'='*80}")
    print("[TRAIN] Training complete")
    print(f"{'='*80}\n")
    
    return policy


def evaluate_verbose(args, policy, sampler, data_handler, eval_env, test_env):
    """Run validation/test evaluation to mirror SB3 metrics."""
    print("\n" + "="*80)
    print("[EVAL] Starting evaluation")
    print("="*80)
    print("---------------evaluation started---------------")
    eval_start_time = time.time()

    def _print_table(name: str, metrics: Dict[str, Any]):
        if not metrics:
            return
        print(f"\n{name} metrics:")
        _print_eval_table(metrics, getattr(args, "timesteps_train", None))

    metrics_train = {"mrr_mean": 0.0, "hits1_mean": 0.0, "hits3_mean": 0.0, "hits10_mean": 0.0}
    metrics_valid, metrics_test = _evaluate(
        args,
        policy,
        sampler,
        data_handler,
        eval_env,
        test_env,
    )

    if metrics_valid:
        _print_table("[EVAL] Validation", metrics_valid)
    if metrics_test:
        _print_table("[EVAL] Test", metrics_test)

    print(f"---------------evaluation finished---------------  took {time.time() - eval_start_time:.2f} seconds")

    return metrics_train, metrics_valid, metrics_test


def _evaluate(
    args: Any,
    policy: nn.Module,
    sampler: Sampler,
    data_handler: DataHandler,
    eval_env: Optional[BatchedEnv],
    test_env: Optional[BatchedEnv],
) -> Tuple[dict, dict]:
    policy.eval()

    # metrics_valid = _evaluate_split(
    #     policy,
    #     eval_env,
    #     sampler,
    #     data_handler.get_materialized_split("valid"),
    #     n_queries=args.n_eval_queries,
    #     n_corruptions=args.eval_neg_samples,
    #     corruption_scheme=args.corruption_scheme,
    #     verbose=getattr(args, "depth_info", False),
    # ) if eval_env is not None else {}
    metrics_valid = _evaluate_split(
        policy,
        eval_env,
        sampler,
        data_handler.get_materialized_split("valid"),
        n_queries=args.n_eval_queries,
        n_corruptions=args.eval_neg_samples,
        corruption_scheme=args.corruption_scheme,
        verbose=getattr(args, "depth_info", False),
        total_timesteps=args.timesteps_train,
    ) if eval_env is not None else {}
    metrics_test = _evaluate_split(
        policy,
        test_env,
        sampler,
        data_handler.get_materialized_split("test"),
        n_queries=args.n_test_queries,
        n_corruptions=args.test_neg_samples,
        corruption_scheme=args.corruption_scheme,
        verbose=getattr(args, "depth_info", False),
        total_timesteps=args.timesteps_train,
    ) if test_env is not None else {}


    return metrics_valid, metrics_test


def _evaluate_split(
    policy: nn.Module,
    env: Optional[BatchedEnv],
    sampler: Sampler,
    split,
    *,
    n_queries: Optional[int],
    n_corruptions: Optional[int],
    corruption_scheme,
    verbose: bool = False,
    total_timesteps: Optional[int] = None,
) -> Dict[str, float]:
    if env is None or split is None or len(split) == 0:
        return {}

    queries = split.queries
    total = queries.shape[0]
    if n_queries is not None:
        queries = queries[: min(n_queries, total)]
    queries = queries.to(env._device)
    query_depths = split.depths[: queries.shape[0]].to(env._device)

    class _EvalActor(nn.Module):
        def __init__(self, base_policy: nn.Module):
            super().__init__()
            self.policy = base_policy

        def forward(self, obs_td: TensorDict, deterministic: bool = True) -> TensorDict:
            actions, _, log_probs = self.policy(obs_td, deterministic=deterministic)
            return TensorDict(
                {"action": actions, "sample_log_prob": log_probs},
                batch_size=obs_td.batch_size,
            )

    eval_actor = _EvalActor(policy)

    # Collect rollout-style metrics with depth/label breakdown during eval
    from callbacks import DetailedMetricsCollector
    collector = DetailedMetricsCollector(collect_detailed=True, verbose=False)

    q_len = queries.shape[0]

    def _info_cb(td):
        # td contains next observations/rewards/done flags
        done = td.get("done") or td.get(("next", "done"))
        if done is None:
            return
        done = done.view(-1).bool()
        if not done.any():
            return
        rewards = td.get("reward") or td.get(("next", "reward"))
        lengths = td.get("length") or td.get(("next", "length"))
        labels = td.get("label") or td.get(("next", "label"))
        depths = td.get("query_depth") or td.get(("next", "query_depth"))
        success = td.get("is_success") or td.get(("next", "is_success"))
        infos = []
        for idx in torch.where(done)[0]:
            if idx.item() >= q_len:
                continue
            info = {"episode": {}}
            if rewards is not None:
                info["episode"]["r"] = float(rewards[idx].item())
            if lengths is not None:
                info["episode"]["l"] = float(lengths[idx].item())
            if labels is not None:
                info["label"] = int(labels[idx].item())
            if depths is not None:
                info["query_depth"] = int(depths[idx].item())
            if success is not None:
                info["is_success"] = bool(success[idx].item())
            infos.append(info)
        if infos:
            collector.accumulate(infos)

    # Limit corruptions to keep parity with SB3 logs; default to 3 when unset.
    corruption_count = n_corruptions
    if corruption_count is None:
        corruption_count = 3

    metrics = eval_corruptions(
        actor=eval_actor,
        env=env,
        queries=queries,
        sampler=sampler,
        query_depths=query_depths,
        n_corruptions=corruption_count,
        corruption_modes=corruption_scheme,
        deterministic=True,
        verbose=verbose,
        info_callback=_info_cb,
    )
    rollout_metrics = collector.compute_metrics()
    # Filter rollout metrics to SB3-style keys
    allowed_base = {
        "len_pos",
        "len_neg",
        "length mean +/- std",
        "proven_pos",
        "proven_neg",
        "reward_label_pos",
        "reward_label_neg",
        "reward_overall",
        "success_rate",
        "ep_len_mean",
        "ep_rew_mean",
    }
    filtered_rollout = {}
    for k, v in rollout_metrics.items():
        if k in allowed_base:
            filtered_rollout[k] = v
        elif k.startswith(("len_d_", "proven_d_", "reward_d_")):
            filtered_rollout[k] = v
    # Always include total_timesteps for display (fallback to None if unknown)
    filtered_rollout["total_timesteps"] = getattr(args, "timesteps_train", None) if "args" in locals() else None

    results = {
        "mrr_mean": float(metrics.get("MRR", 0.0)),
        "hits1_mean": float(metrics.get("Hits@1", 0.0)),
        "hits3_mean": float(metrics.get("Hits@3", 0.0)),
        "hits10_mean": float(metrics.get("Hits@10", 0.0)),
    }
    # Pull through any preformatted rollout-style stats from evaluate_ranking_metrics
    for key, val in metrics.items():
        if key not in results:
            results[key] = val

    per_mode = metrics.get("per_mode", {})
    for mode, mode_metrics in per_mode.items():
        results[f"{mode}_mrr_mean"] = float(mode_metrics.get("MRR", 0.0))
        results[f"{mode}_hits1_mean"] = float(mode_metrics.get("Hits@1", 0.0))
        results[f"{mode}_hits3_mean"] = float(mode_metrics.get("Hits@3", 0.0))
        results[f"{mode}_hits10_mean"] = float(mode_metrics.get("Hits@10", 0.0))

    # Merge rollout-style metrics from the collector (already formatted)
    if total_timesteps is not None:
        filtered_rollout["total_timesteps"] = total_timesteps
    results.update({k: v for k, v in filtered_rollout.items() if k not in results})

    return results


def main(args, log_filename, use_logger, use_WB, WB_path, date):
    """
    Main entry point that mimics sb3/train.py main function.
    """
    _warn_non_reproducible(args)
    _set_seeds(args.seed_run_i)

    args.corruption_scheme = list(args.corruption_scheme or ["head", "tail"])

    device = get_device(args.device)
    print(f"\n[SETUP] Device: {device}")
    print(f"[SETUP] CUDA available: {torch.cuda.is_available()}")
    print(f"[SETUP] Device count: {torch.cuda.device_count()}")

    # Determine rollout device
    rollout_device = device
    if hasattr(args, 'rollout_device') and args.rollout_device is not None:
        rollout_device = torch.device(args.rollout_device)
        print(f"[SETUP] Using separate rollout device: {rollout_device}")

    # Build data components on the rollout device (env side) while keeping the
    # model/embedder on the model device.
    (
        data_handler,
        index_manager,
        sampler,
        embedder,
        unification_engine,
        stringifier_params,
    ) = _materialize_data_components(
        args,
        device=rollout_device,
        embed_device=device,
    )

    train_split = data_handler.get_materialized_split("train")
    valid_split = data_handler.get_materialized_split("valid")
    test_split = data_handler.get_materialized_split("test")
    
    print("\n[ENV] Creating environments...")
    train_env = _make_env_from_split(
        train_split,
        mode="train",
        batch_size=args.batch_size_env,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        unification_engine=unification_engine,
        stringifier_params=stringifier_params,
        device=rollout_device,
    )
    print(f"[ENV] Train env: batch_size={train_env.batch_size}")
    
    eval_env = _make_env_from_split(
        valid_split,
        mode="eval",
        batch_size=args.batch_size_env_eval,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        unification_engine=unification_engine,
        stringifier_params=stringifier_params,
        device=rollout_device,
    )
    print(f"[ENV] Eval env: batch_size={eval_env.batch_size}")
    
    test_env = _make_env_from_split(
        test_split,
        mode="eval",
        batch_size=args.batch_size_env_eval,
        args=args,
        index_manager=index_manager,
        sampler=sampler,
        unification_engine=unification_engine,
        stringifier_params=stringifier_params,
        device=rollout_device,
    )
    print(f"[ENV] Test env: batch_size={test_env.batch_size}")

    embed_dim = getattr(embedder, "embed_dim", getattr(embedder, "atom_embedding_size", args.atom_embedding_size))
    
    # Disable optimizations for debugging
    use_amp = False
    use_compile = False
    
    print("\n[MODEL] Creating policy...")
    torch.manual_seed(args.seed_run_i)
    hidden_dim = getattr(args, "model_hidden_dim", 128)
    num_layers = getattr(args, "model_num_layers", 8)
    dropout_prob = getattr(args, "model_dropout", 0.2)
    policy = create_actor_critic(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        device=device,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        max_arity=index_manager.max_arity,
        total_vocab_size=index_manager.total_vocab_size,
        init_seed=args.seed,
        match_sb3_init=False,
    )
    torch.save(policy.state_dict(), "batched_init.pt")
    
    # Verify weights
    total_params = sum(p.sum() for p in policy.parameters())
    print(f"DEBUG: Batched Policy Weight Sum: {total_params.item()}")
    print(f"[MODEL] Policy created on device: {device}")

    model_path = _model_dir(args, date)
    model_path.mkdir(parents=True, exist_ok=True)

    trace_recorder = TraceRecorder(args.trace_dir, prefix="batched") if getattr(args, "trace_dir", None) else None

    if args.load_model:
        ckpt = _resolve_ckpt_to_load(model_path, restore_best=args.restore_best_val_model)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {model_path}")
        checkpoint = torch.load(ckpt, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"[MODEL] Loaded model from {ckpt}")
    else:
        policy = train_ppo_verbose(
            args,
            policy,
            train_env,
            eval_env,
            sampler,
            data_handler,
            model_path,
            device,
            trace_recorder=trace_recorder,
        )

    policy.apply(_freeze_dropout_layernorm)
    policy.eval()

    metrics_train, metrics_valid, metrics_test = evaluate_verbose(
        args,
        policy,
        sampler,
        data_handler,
        eval_env,
        test_env,
    )
    if trace_recorder is not None:
        if metrics_valid:
            trace_recorder.log_eval("valid", metrics_valid)
        if metrics_test:
            trace_recorder.log_eval("test", metrics_test)
        trace_recorder.flush()

    return metrics_train, metrics_valid, metrics_test
