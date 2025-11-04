# -*- coding: utf-8 -*-
# model_eval.py — SB3-independent TorchRL evaluation utilities (compiled/AMP-ready)
#
# Design goals:
# - Pure TorchRL (no SB3).
# - Minimal CPU↔GPU transfers.
# - Deterministic masked-argmax path when not sampling.
# - Worker-index–ordered results for stable corruption batching.
# - Batched corruption evaluation with ParallelEnv.
# - Optional torch.compile and autocast (AMP) for speed.
#
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import ParallelEnv, SerialEnv, EnvBase

# ----------------------------
# Utilities
# ----------------------------

def _device_of(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (argmax indices, logprob at argmax) under a boolean mask."""
    mask = mask.to(torch.bool)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
    # Argmax & its log-prob under log-softmax
    action_idx = torch.argmax(masked_logits, dim=-1)
    logp = torch.log_softmax(masked_logits, dim=-1).gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    return action_idx, logp


def _unwrap_actor_critic(actor: nn.Module) -> nn.Module:
    """
    Best-effort unwrapping to reach a module exposing `forward_actor(td)`.
    Works with stacks like: ProbabilisticActor -> TensorDictModule -> custom backbone.
    """
    if hasattr(actor, "forward_actor"):
        return actor
    
    # Try common wrapping attributes
    for attr in ("actor_critic_model", "module", "_module", "backbone", "operator", "0"):
        inner = getattr(actor, attr, None)
        if inner is None:
            continue
        if hasattr(inner, "forward_actor"):
            return inner
        # Second level unwrapping
        for attr2 in ("actor_critic_model", "module", "_module", "backbone", "operator", "0"):
            inner2 = getattr(inner, attr2, None)
            if inner2 is not None and hasattr(inner2, "forward_actor"):
                return inner2
    
    # If actor is a ModuleList or Sequential, try first element
    if isinstance(actor, (nn.ModuleList, nn.Sequential)):
        if len(actor) > 0 and hasattr(actor[0], "forward_actor"):
            return actor[0]
    
    # Fallback: assume it can directly consume a TD and emit logits
    return actor


def _maybe_compile(fn: Callable, enable: bool, mode: str = "max-autotune", fullgraph: bool = False) -> Callable:
    if not enable:
        return fn
    if not hasattr(torch, "compile"):
        return fn  # PyTorch < 2.0
    try:
        return torch.compile(fn, mode=mode, fullgraph=fullgraph)
    except Exception:
        return fn


def _autocast_context(enable: bool, device: torch.device, dtype_str: str = "bf16"):
    if not enable:
        return torch.autocast("cpu", enabled=False)  # inert context
    devtype = "cuda" if device.type == "cuda" else "cpu"
    # Prefer bf16 when available (safer numerics)
    if dtype_str.lower() in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.autocast(devtype, dtype=dtype, enabled=True)


@dataclass
class EvalEpisodeResult:
    success: bool
    length: int
    reward: float
    logp_sum: float


# ----------------------------
# Thin policy helper
# ----------------------------

class TorchRLPolicyWrapper:
    """
    Minimal wrapper exposing `.act(obs_td, deterministic)`.
    Adds optional torch.compile + autocast.
    """
    def __init__(
        self,
        actor: nn.Module,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
        compile_mode: str = "max-autotune",
        compile_fullgraph: bool = False,
        use_autocast: bool = False,
        autocast_dtype: str = "bf16",
    ):
        self.actor = actor
        self.device = device if device is not None else _device_of(actor)
        self._ac = _unwrap_actor_critic(actor)
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

        # Prepare a small forward function we can optionally compile
        def _forward(td: TensorDict) -> torch.Tensor:
            return self._ac.forward_actor(td)

        self._forward = _maybe_compile(_forward, enable=use_compile, mode=compile_mode, fullgraph=compile_fullgraph)

    @torch.inference_mode()
    def act(self, td_obs: TensorDict, deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        td = TensorDict(
            {
                "sub_index": td_obs["sub_index"].to(self.device, non_blocking=True),
                "derived_sub_indices": td_obs["derived_sub_indices"].to(self.device, non_blocking=True),
                "action_mask": td_obs["action_mask"].to(self.device, non_blocking=True),
            },
            batch_size=td_obs.batch_size,
        )
        with _autocast_context(self.use_autocast, self.device, self.autocast_dtype):
            logits = self._forward(td)
        if deterministic:
            return _masked_argmax(logits, td["action_mask"])

        masked_logits = logits.masked_fill(~td["action_mask"].bool(), float("-inf"))
        masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp


# ----------------------------
# Plain TorchRL policy evaluation
# ----------------------------

@torch.inference_mode()
def evaluate_policy_torchrl(
    actor: torch.nn.Module,
    env: EnvBase | ParallelEnv | SerialEnv,
    n_eval_episodes: int = 1,
    deterministic: bool = True,
    target_episodes: Optional[np.ndarray] = None,
    verbose: int = 0,
    track_logprobs: bool = False,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    expected_num_envs: Optional[int] = None,  # NEW: explicitly pass number of envs if known
    # new perf flags:
    use_compile: bool = False,
    compile_mode: str = "max-autotune",
    compile_fullgraph: bool = False,
    use_autocast: bool = False,
    autocast_dtype: str = "bf16",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate `actor` on a (possibly parallel) TorchRL env and return arrays:
    (rewards, lengths, logps, mask, success).  **Order is worker index**.
    Assumes 1 episode per worker unless `target_episodes` is provided.
    """
    # Handle different ways to get number of environments
    # If caller knows the number, use it directly to avoid issues with unstarted ParallelEnv
    if expected_num_envs is not None:
        n_envs = expected_num_envs
    else:
        # Try to access num_envs, but if env is not started yet, handle gracefully
        n_envs = 1  # Default fallback
        try:
            num_envs_attr = getattr(env, "num_envs", None)
            if callable(num_envs_attr):
                n_envs = int(num_envs_attr())
            elif num_envs_attr is not None:
                n_envs = int(num_envs_attr)
            else:
                n_envs = 1
        except (RuntimeError, TypeError) as e:
            # ParallelEnv not started yet - use default
            n_envs = 1

    if target_episodes is None:
        target_episodes = np.ones(n_envs, dtype=np.int32) * int(math.ceil(n_eval_episodes / n_envs))
    assert (target_episodes == 1).all(), "This evaluator currently supports exactly 1 episode per worker."

    policy = TorchRLPolicyWrapper(
        actor,
        device=_device_of(actor),
        use_compile=use_compile,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        use_autocast=use_autocast,
        autocast_dtype=autocast_dtype,
    )
    device = policy.device

    td = env.reset()
    cur_logp_sum = torch.zeros(n_envs, device=device)
    cur_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
    cur_reward = torch.zeros(n_envs, device=device)

    rewards = torch.full((n_envs,), float("nan"), device=device)
    lengths = torch.zeros(n_envs, dtype=torch.int32, device=device)
    logps = torch.zeros(n_envs, device=device)
    success = torch.zeros(n_envs, dtype=torch.float32, device=device)
    finished = torch.zeros(n_envs, dtype=torch.bool, device=device)

    while not finished.all():
        # Actor forward + action selection with (optional) AMP
        with _autocast_context(use_autocast, device, autocast_dtype):
            sub_index = td["sub_index"].to(device, non_blocking=True)
            derived = td["derived_sub_indices"].to(device, non_blocking=True)
            mask = td["action_mask"].to(device, non_blocking=True)
            
            # Ensure tensors have correct batch dimension
            if sub_index.ndim == 3:  # Single env without batch dim
                sub_index = sub_index.unsqueeze(0)
                derived = derived.unsqueeze(0)
                mask = mask.unsqueeze(0)
            
            logits = policy._forward(TensorDict(
                {"sub_index": sub_index, "derived_sub_indices": derived, "action_mask": mask},
                batch_size=torch.Size([n_envs]),
            ))
            if deterministic:
                action_idx, logp = _masked_argmax(logits, mask.reshape(n_envs, -1))
            else:
                masked_logits = logits.masked_fill(~mask.reshape(n_envs, -1).bool(), float("-inf"))
                masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                logp = dist.log_prob(action_idx)

        # Construct action tensor for environment
        action_for_env = action_idx.to(torch.long).cpu()
        if n_envs == 1 and action_for_env.ndim == 1:
            # Single environment might not expect batch dimension
            action_batch_size = torch.Size([])
        else:
            action_batch_size = torch.Size([n_envs])
        
        step_td = TensorDict({"action": action_for_env}, batch_size=action_batch_size)
        next_td = env.step(step_td)
        nxt = next_td.get("next", next_td)

        # Handle reward/done extraction for both single and parallel envs
        r = nxt["reward"]
        d = nxt["done"]
        if r.ndim > 1:
            r = r.reshape(n_envs, -1)[:, 0].to(device)
        else:
            r = r.reshape(n_envs).to(device)
        if d.ndim > 1:
            d = d.reshape(n_envs, -1)[:, 0].to(torch.bool)
        else:
            d = d.reshape(n_envs).to(torch.bool)
        cur_logp_sum += logp
        cur_len += 1
        cur_reward += r

        if d.any():
            finished_infos = []
            idxs = torch.where(d)[0].tolist()
            for i in idxs:
                if not finished[i]:
                    rewards[i] = cur_reward[i]
                    lengths[i] = cur_len[i]
                    if track_logprobs:
                        logps[i] = cur_logp_sum[i]
                    # Success flag if available
                    is_success = 0.0
                    if "is_success" in nxt.keys():
                        is_success = float(nxt["is_success"][i].item()) if nxt["is_success"][i].numel() > 0 else 0.0
                    success[i] = is_success
                    finished[i] = True

                    # Optional info callback
                    label = int(nxt.get("label", torch.tensor([1]))[i].item()) if "label" in nxt.keys() else 1
                    qdepth = int(nxt.get("query_depth", torch.tensor([-1]))[i].item()) if "query_depth" in nxt.keys() else -1
                    finished_infos.append({
                        "episode": {"r": float(rewards[i].item()), "l": int(lengths[i].item())},
                        "label": label,
                        "query_depth": (None if qdepth in (-1, None) else qdepth),
                        "is_success": bool(is_success),
                    })

                    cur_logp_sum[i] = 0.0
                    cur_len[i] = 0
                    cur_reward[i] = 0.0

            if info_callback is not None and finished_infos:
                info_callback(finished_infos)

        td = nxt

    rewards_np = rewards.detach().cpu().numpy().astype(np.float32)
    lengths_np = lengths.detach().cpu().numpy().astype(np.int32)
    logps_np = logps.detach().cpu().numpy().astype(np.float32) if track_logprobs else np.zeros_like(rewards_np)
    mask_np = np.ones_like(rewards_np, dtype=np.bool_)
    success_np = success.detach().cpu().numpy().astype(np.float32)
    return rewards_np, lengths_np, logps_np, mask_np, success_np


# ----------------------------
# Corruption-based evaluation
# ----------------------------

def _extract_base_env_kwargs(base_env: EnvBase) -> Dict[str, Any]:
    """Probe env to recover constructor params we must replicate."""
    # Don't try to access child environments if ParallelEnv isn't started
    # Just use the base environment's attributes directly
    e0 = base_env
    
    # Helper to safely extract values (handles dispatch_caller objects)
    def safe_getattr(obj, attr, default):
        try:
            # First try __dict__ to avoid dispatch_caller
            if hasattr(obj, '__dict__') and attr in obj.__dict__:
                val = obj.__dict__[attr]
                if not (hasattr(val, '__call__') and not isinstance(val, (int, float, str, bool, type(None)))):
                    return val
            # Fallback to regular getattr
            val = getattr(obj, attr, default)
            # Handle dispatch_caller or other non-primitive types
            if hasattr(val, '__call__') and not isinstance(val, (int, float, str, bool, type(None))):
                return default
            return val
        except (RuntimeError, AttributeError, TypeError):
            return default
    
    # Get critical values with proper defaults
    padding_atoms = safe_getattr(e0, "padding_atoms", 6)
    padding_states = safe_getattr(e0, "padding_states", 32)
    max_depth = safe_getattr(e0, "max_depth", 20)
    
    return {
        "index_manager": safe_getattr(e0, "index_manager", None),
        "data_handler": safe_getattr(e0, "data_handler", None),
        "facts": safe_getattr(e0, "facts", None),
        "engine": safe_getattr(e0, "engine", "python"),
        "kge_action": safe_getattr(e0, "kge_action", False),
        "kge_inference_engine": safe_getattr(e0, "kge_inference_engine", None),
        "reward_type": safe_getattr(e0, "reward_type", 0),
        "shaping_beta": safe_getattr(e0, "shaping_beta", 0.0),
        "shaping_gamma": safe_getattr(e0, "shaping_gamma", None),
        "endf_action": safe_getattr(e0, "endf_action", 0),
        "endt_action": safe_getattr(e0, "endt_action", 0),
        "skip_unary_actions": safe_getattr(e0, "skip_unary_actions", False),
        "padding_atoms": padding_atoms,
        "padding_states": padding_states,
        "device": torch.device("cpu"),
        "verbose": safe_getattr(e0, "verbose", 0),
        "prover_verbose": safe_getattr(e0, "prover_verbose", 0),
        "max_depth": max_depth,
        "corruption_mode": safe_getattr(e0, "corruption_mode", None),
        "corruption_scheme": safe_getattr(e0, "corruption_scheme", None),
        "train_neg_ratio": safe_getattr(e0, "train_neg_ratio", 1),
    }


def _make_eval_envs_like(
    base_env: EnvBase,
    queries: Sequence[Any],
    labels: Sequence[int],
    depths: Optional[Sequence[int]] = None,
    mode: str = "eval",
    index_manager: Optional[Any] = None,
    data_handler: Optional[Any] = None,
) -> ParallelEnv | SerialEnv:
    """Build a temporary batched TorchRL env (ParallelEnv) with given (q,label,depth)."""
    from env import LogicEnv_gym as LogicEnv  # local import; same class used elsewhere
    base_kwargs = _extract_base_env_kwargs(base_env)
    
    # Override index_manager and data_handler if explicitly provided
    if index_manager is not None:
        base_kwargs["index_manager"] = index_manager
    if data_handler is not None:
        base_kwargs["data_handler"] = data_handler
        # Also get facts from data_handler if available
        if hasattr(data_handler, 'facts') and data_handler.facts is not None:
            base_kwargs["facts"] = data_handler.facts

    def _factory(q_term, y, depth):
        def _init():
            env = LogicEnv(
                mode=mode,
                queries=[q_term],
                labels=[int(y)],
                query_depths=[depth] if depth is not None else None,
                facts=base_kwargs["facts"],
                index_manager=base_kwargs["index_manager"],
                data_handler=base_kwargs["data_handler"],
                corruption_mode=base_kwargs["corruption_mode"],
                corruption_scheme=base_kwargs["corruption_scheme"],
                train_neg_ratio=base_kwargs["train_neg_ratio"],
                max_depth=base_kwargs["max_depth"],
                endf_action=base_kwargs["endf_action"],
                endt_action=base_kwargs["endt_action"],
                skip_unary_actions=base_kwargs["skip_unary_actions"],
                padding_atoms=base_kwargs["padding_atoms"],
                padding_states=base_kwargs["padding_states"],
                device=base_kwargs["device"],
                engine=base_kwargs["engine"],
                kge_action=base_kwargs["kge_action"],
                reward_type=base_kwargs["reward_type"],
                shaping_beta=base_kwargs["shaping_beta"],
                shaping_gamma=base_kwargs["shaping_gamma"],
                kge_inference_engine=base_kwargs["kge_inference_engine"],
                verbose=base_kwargs["verbose"],
                prover_verbose=base_kwargs["prover_verbose"],
            )
            return env
        return _init

    env_fns = []
    for i, q in enumerate(queries):
        d = depths[i] if depths is not None else None
        env_fns.append(_factory(q, int(labels[i]), d))

    # Use SerialEnv instead of ParallelEnv to avoid multiprocessing issues during evaluation
    # SerialEnv is slower but more reliable and easier to debug
    if len(env_fns) == 1:
        return SerialEnv(1, env_fns)
    # Always use SerialEnv for evaluation to avoid worker crashes
    return SerialEnv(len(env_fns), env_fns)


def _score_from_traj(success: torch.Tensor, logp_sum: torch.Tensor) -> torch.Tensor:
    # Success-first ordering; tie-break by log-prob mass
    return success.to(logp_sum.dtype) * 1e6 + logp_sum


def _rank_of_true(scores: torch.Tensor, true_index: int = 0) -> int:
    # Higher score is better. Rank starts at 1.
    order = torch.argsort(scores, dim=-1, descending=True)
    pos = int((order == true_index).nonzero(as_tuple=True)[0].item())
    return pos + 1


def _generate_negatives(
    sampler: Any,
    state: Any,
    num_negs: int,
    device: Optional[torch.device] = None,
) -> List[Any]:
    """
    Robust negative generator that handles different sampler APIs.
    Expected to return a list of `num_negs` corrupted queries (states).
    """
    # Preferred API
    getter = getattr(sampler, "get_negatives_from_states", None)
    if callable(getter):
        out = getter(state, device if device is not None else torch.device("cpu"), num_negs=num_negs)
        if isinstance(out, list):
            return out[:num_negs]
        return [out][:num_negs]

    # Fallbacks
    getter = getattr(sampler, "get_negatives", None)
    if callable(getter):
        out = getter(state, num_negs=num_negs)
        return out[:num_negs] if isinstance(out, list) else [out][:num_negs]

    raise RuntimeError("Sampler must provide `get_negatives_from_states(state, device, num_negs)` or a compatible API.")


@torch.inference_mode()
def eval_corruptions_torchrl(
    actor: torch.nn.Module,
    env: EnvBase | ParallelEnv | SerialEnv,
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,  # reserved
    kge_inference_engine: Any = None,  # signature compatibility
    evaluation_mode: str = 'rl_only',  # signature compatibility
    corruption_scheme: Optional[List[str]] = None,  # signature compatibility
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    data_depths: Optional[List[int]] = None,
    hybrid_kge_weight: float = 2.0,  # unused in pure RL
    hybrid_rl_weight: float = 1.0,   # unused in pure RL
    hybrid_success_only: bool = True,# unused in pure RL
    index_manager: Optional[Any] = None,  # can be provided explicitly
    data_handler: Optional[Any] = None,  # can be provided explicitly
    group_size: int = 1,  # number of positive queries to evaluate per batch (default 1 to avoid OOM)
    # new perf flags (forwarded to evaluate_policy_torchrl):
    use_compile: bool = False,
    compile_mode: str = "max-autotune",
    compile_fullgraph: bool = False,
    use_autocast: bool = False,
    autocast_dtype: str = "bf16",
) -> Dict[str, Any]:
    """
    Pure TorchRL corruption-based evaluator.

    For each positive query q, generate negatives, evaluate [q]+negs in a *single*
    ParallelEnv (batched by `group_size`) to compute an RL score per candidate,
    then aggregate MRR/Hits.
    """
    device = _device_of(actor)

    # Infer default corruption count
    if n_corruptions is None:
        n_corruptions = int(getattr(sampler, "num_negs_per_pos", 50)) or 50

    # Accumulators
    mrrs: List[float] = []
    hits1: List[float] = []
    hits3: List[float] = []
    hits10: List[float] = []
    succ_flags: List[float] = []
    lens_pos: List[int] = []
    lens_neg: List[int] = []
    rew_pos: List[float] = []
    rew_neg: List[float] = []

    # Process in groups for higher throughput
    N = len(data)
    if group_size <= 0:
        group_size = 1
    for start in range(0, N, group_size):
        end = min(N, start + group_size)
        group_queries = data[start:end]
        group_depths = data_depths[start:end] if data_depths is not None else [None] * (end - start)

        # Build candidate pool for this group
        cand_queries: List[Any] = []
        cand_labels: List[int] = []
        cand_depths: List[Optional[int]] = []
        block_sizes: List[int] = []
        for q, d in zip(group_queries, group_depths):
            negatives = _generate_negatives(sampler, q, n_corruptions, device=torch.device("cpu"))
            # Ensure we have exactly n_corruptions
            if len(negatives) < n_corruptions and len(negatives) > 0:
                negatives = negatives + [negatives[-1]] * (n_corruptions - len(negatives))
            negatives = negatives[:n_corruptions]

            # Block layout: [positive first] + negatives
            cand_queries.extend([q] + negatives)
            cand_labels.extend([1] + [0] * len(negatives))
            # Use same depth for positive and negatives (or -1 if depth is None)
            depth_to_use = d if d is not None else -1
            cand_depths.extend([depth_to_use] * (1 + len(negatives)))
            block_sizes.append(1 + len(negatives))

        # Build a single ParallelEnv for all candidates in this group
        eval_env = _make_eval_envs_like(env, cand_queries, cand_labels, cand_depths, mode="eval", 
                                        index_manager=index_manager, data_handler=data_handler)

        # We know exactly how many environments we created
        num_eval_envs = len(cand_queries)

        # 1 episode per worker, in *worker index* order
        r, L, logp, _, success = evaluate_policy_torchrl(
            actor, eval_env, n_eval_episodes=1, deterministic=deterministic,
            target_episodes=None, verbose=0, track_logprobs=True, info_callback=info_callback,
            expected_num_envs=num_eval_envs,  # Pass known number of envs
            use_compile=use_compile, compile_mode=compile_mode, compile_fullgraph=compile_fullgraph,
            use_autocast=use_autocast, autocast_dtype=autocast_dtype,
        )

        # Convert to torch tensors for convenience
        r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
        L_t = torch.as_tensor(L, dtype=torch.int32, device=device)
        logp_t = torch.as_tensor(logp, dtype=torch.float32, device=device)
        succ_t = torch.as_tensor(success, dtype=torch.float32, device=device)

        # Evaluate each block independently
        offset = 0
        for bsz in block_sizes:
            block_slice = slice(offset, offset + bsz)
            # true candidate is at position 0 within block
            block_scores = succ_t[block_slice].to(logp_t.dtype) * 1e6 + logp_t[block_slice]
            order = torch.argsort(block_scores, descending=True)
            rank = int((order == 0).nonzero(as_tuple=True)[0].item()) + 1

            mrrs.append(1.0 / float(rank))
            hits1.append(1.0 if rank <= 1 else 0.0)
            hits3.append(1.0 if rank <= 3 else 0.0)
            hits10.append(1.0 if rank <= 10 else 0.0)

            succ_flags.append(float(succ_t[block_slice][0].item()))
            lens_pos.append(int(L_t[block_slice][0].item()))
            rew_pos.append(float(r_t[block_slice][0].item()))
            if bsz > 1:
                lens_neg.append(int(torch.mean(L_t[block_slice][1:].to(torch.float32)).item()))
                rew_neg.append(float(torch.mean(r_t[block_slice][1:]).item()))

            offset += bsz

        try:
            eval_env.close()
        except Exception:
            pass

    def _mean(x: List[float]) -> float:
        return float(np.mean(x)) if len(x) > 0 else float("nan")

    metrics = {
        "mrr_mean": _mean(mrrs),
        "hits1_mean": _mean(hits1),
        "hits3_mean": _mean(hits3),
        "hits10_mean": _mean(hits10),
        "success_pos_mean": _mean(succ_flags),
        "episode_len_pos_mean": _mean([float(v) for v in lens_pos]) if lens_pos else float("nan"),
        "episode_len_neg_mean": _mean([float(v) for v in lens_neg]) if lens_neg else float("nan"),
        "rewards_pos_mean": _mean(rew_pos) if rew_pos else float("nan"),
        "rewards_neg_mean": _mean(rew_neg) if rew_neg else float("nan"),
        "evaluated_queries": N,
        "corruptions_per_query": int(n_corruptions),
        "group_size": int(group_size),
    }
    if verbose:
        print(f"[eval] MRR={metrics['mrr_mean']:.4f}  H@1={metrics['hits1_mean']:.4f}  H@10={metrics['hits10_mean']:.4f}  "
              f"(N={N}, negs={n_corruptions}, group={group_size})")
    return metrics
