"""
model_eval.py — SB3-independent TorchRL evaluation utilities

This module provides **native TorchRL** evaluation functions with no dependency
on Stable-Baselines3. It keeps the speed of highly vectorized rollouts by
operating directly on TorchRL batched envs (e.g., ParallelEnv) and on the
underlying actor-critic model used for PPO.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import ParallelEnv, SerialEnv, EnvBase

# ----------------------------
# Utilities
# ----------------------------

def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return argmax indices and corresponding log-softmax values under mask."""
    mask = mask.to(torch.bool)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
    # Argmax
    action_idx = torch.argmax(masked_logits, dim=-1)
    # Log-prob of the chosen action (deterministic argmax): compute with log-softmax
    logp = torch.log_softmax(masked_logits, dim=-1).gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    return action_idx, logp

def _device_of(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _unwrap_actor_critic(actor: nn.Module) -> nn.Module:
    """
    Best-effort unwrapping to reach the underlying ActorCriticModel that exposes
    `forward_actor` and `forward_critic`. Works with our TorchRL stacks
    (ProbabilisticActor -> TensorDictModule -> TorchRLActorModule -> ActorCriticModel).
    """
    # Direct case
    if hasattr(actor, "forward_actor") and hasattr(actor, "forward_critic"):
        return actor
    # Common wrappers
    for attr in ("actor_critic_model", "module", "_module", "backbone", "operator"):
        inner = getattr(actor, attr, None)
        if inner is None:
            continue
        if hasattr(inner, "forward_actor") and hasattr(inner, "forward_critic"):
            return inner
        # one more hop
        for attr2 in ("actor_critic_model", "module", "_module", "backbone", "operator"):
            inner2 = getattr(inner, attr2, None)
            if inner2 is not None and hasattr(inner2, "forward_actor") and hasattr(inner2, "forward_critic"):
                return inner2
    # Fallback: assume actor itself handles TensorDict and returns 'action'/'sample_log_prob'
    return actor


@dataclass
class EvalEpisodeResult:
    success: bool
    length: int
    reward: float
    logp_sum: float


# ----------------------------
# Thin policy helper (kept for parity with existing imports)
# ----------------------------

class TorchRLPolicyWrapper:
    """
    Minimal wrapper that exposes a simple `.act(td, deterministic)` interface.
    This is **not** an SB3 policy and has no external dependencies.
    """
    def __init__(self, actor: nn.Module, device: Optional[torch.device] = None):
        self.actor = actor
        self.device = device if device is not None else _device_of(actor)
        self._ac = _unwrap_actor_critic(actor)

    @torch.inference_mode()
    def act(self, td_obs: TensorDict, deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given an observation tensordict with keys: sub_index, derived_sub_indices, action_mask,
        return (action_idx, log_prob).
        """
        td = TensorDict({
            "sub_index": td_obs["sub_index"].to(self.device),
            "derived_sub_indices": td_obs["derived_sub_indices"].to(self.device),
            "action_mask": td_obs["action_mask"].to(self.device),
        }, batch_size=td_obs.batch_size)

        # If we can access logits directly, do it for speed & determinism
        if hasattr(self._ac, "forward_actor"):
            logits = self._ac.forward_actor(td)
            if deterministic:
                action_idx, logp = _masked_argmax(logits, td["action_mask"])
            else:
                masked_logits = logits.masked_fill(~td["action_mask"].bool(), float("-inf"))
                masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                logp = dist.log_prob(action_idx)
            return action_idx, logp

        # Fallback: call actor to sample distribution-driven action
        out = self.actor(td.clone())
        action = out.get("action")
        logp = out.get("sample_log_prob")
        if action is None:
            raise RuntimeError("Actor did not produce 'action' in the TensorDict.")
        if logp is None:
            # try to reconstruct from logits if present
            logits = out.get("logits", None)
            if logits is not None:
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(action)
            else:
                logp = torch.zeros_like(action, dtype=torch.float32)
        return action, logp


# ----------------------------
# Plain TorchRL policy evaluation (no SB3)
# ----------------------------

@torch.inference_mode()
def evaluate_policy_torchrl(
    actor: torch.nn.Module,
    env: EnvBase | ParallelEnv | SerialEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: Optional[np.ndarray] = None,
    verbose: int = 0,
    track_logprobs: bool = False,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a TorchRL policy on a TorchRL batched environment using TensorDict only.
    Returns arrays for per-episode rewards, lengths, (optional) log ps, mask, and success.
    """
    # Support SerialEnv and single EnvBase uniformly
    if not hasattr(env, "num_envs"):
        n_envs = 1
    else:
        n_envs = int(env.num_envs)

    policy = TorchRLPolicyWrapper(actor)
    device = policy.device

    # Set uniform per-env episode targets unless provided
    if target_episodes is None:
        target_episodes = np.ones(n_envs, dtype=np.int32) * math.ceil(n_eval_episodes / n_envs)
    total_episodes = int(target_episodes.sum())

    rewards_out: List[float] = []
    lens_out: List[int] = []
    logps_out: List[float] = []
    success_out: List[float] = []

    # Reset env
    td = env.reset()

    # Accumulators per env for current running episode
    cur_logp_sum = torch.zeros(n_envs, device=device)
    cur_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
    cur_reward = torch.zeros(n_envs, device=device)

    # Loop until all envs hit their target
    while len(rewards_out) < total_episodes:
        # Compute masked logits on device; keep env data on CPU, move only needed parts
        sub_index = td["sub_index"].to(device, non_blocking=True)
        derived = td["derived_sub_indices"].to(device, non_blocking=True)
        mask = td["action_mask"].to(device, non_blocking=True)

        # Actor forward (deterministic or stochastic)
        logits = policy._ac.forward_actor(TensorDict({
            "sub_index": sub_index, "derived_sub_indices": derived, "action_mask": mask
        }, batch_size=torch.Size([n_envs])))

        if deterministic:
            action_idx, logp = _masked_argmax(logits, mask)
        else:
            masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
            masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()
            logp = dist.log_prob(action_idx)

        # Step env (env expects indices on CPU)
        step_td = TensorDict({"action": action_idx.to(torch.long).cpu()}, batch_size=torch.Size([n_envs]))
        next_td = env.step(step_td)
        nxt = next_td.get("next", next_td)

        # Update accumulators
        r = nxt["reward"].reshape(n_envs, -1)[:, 0].to(device)
        d = nxt["done"].reshape(n_envs, -1)[:, 0]
        cur_logp_sum += logp
        cur_len += 1
        cur_reward += r

        # Build infos for finished episodes
        if d.any():
            finished_infos = []
            for i in torch.where(d)[0].tolist():
                label = int(nxt.get("label", torch.tensor([1]))[i].item()) if "label" in nxt.keys() else 1
                qdepth = int(nxt.get("query_depth", torch.tensor([-1]))[i].item()) if "query_depth" in nxt.keys() else -1
                is_success = bool(nxt.get("is_success", torch.tensor([0]))[i].item()) if "is_success" in nxt.keys() else False

                # push episode aggregates
                rewards_out.append(float(cur_reward[i].item()))
                lens_out.append(int(cur_len[i].item()))
                if track_logprobs:
                    logps_out.append(float(cur_logp_sum[i].item()))
                success_out.append(1.0 if is_success else 0.0)

                finished_infos.append({
                    "episode": {"r": float(cur_reward[i].item()), "l": int(cur_len[i].item())},
                    "label": label,
                    "query_depth": (None if qdepth in (-1, None) else qdepth),
                    "is_success": is_success,
                })

                # reset per-env accumulators for next episode
                cur_logp_sum[i] = 0.0
                cur_len[i] = 0
                cur_reward[i] = 0.0

            # callback collection
            if info_callback is not None and finished_infos:
                info_callback(finished_infos)

        # Advance
        td = nxt  # next observation

    # Convert outputs
    rewards = np.asarray(rewards_out, dtype=np.float32)
    lengths = np.asarray(lens_out, dtype=np.int32)
    logps = np.asarray(logps_out, dtype=np.float32) if track_logprobs else np.zeros_like(rewards)
    mask = np.ones_like(rewards, dtype=np.bool_)  # present for compatibility
    success = np.asarray(success_out, dtype=np.float32)
    return rewards, lengths, logps, mask, success


# ----------------------------
# Corruption-based evaluation (no SB3)
# ----------------------------

def _extract_base_env_kwargs(base_env: EnvBase) -> Dict[str, Any]:
    """Probe env[0] (or the env itself) to recover constructor params we must replicate."""
    e0 = getattr(base_env, "envs", [base_env])[0]
    # Access with getattr to be robust
    return {
        "index_manager": getattr(e0, "index_manager", None),
        "data_handler": getattr(e0, "data_handler", None),
        "facts": getattr(e0, "facts", None),
        "rules": getattr(e0, "allowed_rules", getattr(e0, "rules", None)),
        "engine": getattr(e0, "engine", "python"),
        "kge_action": getattr(e0, "kge_action", False),
        "kge_inference_engine": getattr(e0, "kge_inference_engine", None),
        "reward_type": getattr(e0, "reward_type", 0),
        "shaping_beta": getattr(e0, "shaping_beta", 0.0),
        "shaping_gamma": getattr(e0, "shaping_gamma", None),
        "endf_action": getattr(e0, "endf_action", 0),
        "endt_action": getattr(e0, "endt_action", 0),
        "skip_unary_actions": getattr(e0, "skip_unary_actions", False),
        "padding_atoms": getattr(e0, "padding_atoms", 3),
        "padding_states": getattr(e0, "padding_states", 32),
        "device": torch.device("cpu"),  # individual envs run on CPU; batching handles device placement
        "verbose": getattr(e0, "verbose", 0),
        "prover_verbose": getattr(e0, "prover_verbose", 0),
        "max_depth": getattr(e0, "max_depth", 10),
        "corruption_mode": getattr(e0, "corruption_mode", None),
        "corruption_scheme": getattr(e0, "corruption_scheme", None),
        "train_neg_ratio": getattr(e0, "train_neg_ratio", 1),
    }

def _make_eval_envs_like(
    base_env: EnvBase,
    queries: Sequence[Any],
    labels: Sequence[int],
    depths: Optional[Sequence[int]] = None,
    mode: str = "eval",
) -> ParallelEnv | SerialEnv:
    """Build a temporary batched TorchRL env with the same settings as base_env but new (query,label,depth)."""
    from env import LogicEnv  # import locally; same class used in env_factory
    base_kwargs = _extract_base_env_kwargs(base_env)

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
        env_fns.append(_factory(q, labels[i], d))

    if len(env_fns) == 1:
        return SerialEnv(1, env_fns)
    return ParallelEnv(len(env_fns), env_fns, shared_memory=True)

def _score_from_traj(success: torch.Tensor, logp_sum: torch.Tensor) -> torch.Tensor:
    # Success-first ordering; tie break with logp_sum
    return success.to(logp_sum.dtype) * 1e6 + logp_sum

def _rank_of_true(scores: torch.Tensor, true_index: int = 0) -> int:
    # Higher score is better. Rank starts at 1.
    order = torch.argsort(scores, dim=-1, descending=True)
    pos = int((order == true_index).nonzero(as_tuple=True)[0].item())
    return pos + 1

@torch.inference_mode()
def eval_corruptions_torchrl(
    actor: torch.nn.Module,
    env: EnvBase | ParallelEnv | SerialEnv,
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,
    kge_inference_engine = None,   # kept for signature compatibility
    evaluation_mode: str = 'rl_only',
    corruption_scheme: Optional[List[str]] = None,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    data_depths: Optional[List[int]] = None,
    hybrid_kge_weight: float = 2.0,  # unused in pure RL
    hybrid_rl_weight: float = 1.0,   # unused in pure RL
    hybrid_success_only: bool = True,# unused in pure RL
    index_manager: Optional[Any] = None,  # NEW: pass index_manager directly
) -> Dict[str, Any]:
    """
    Pure TorchRL corruption-based evaluator.

    For each positive query q, generate negatives, evaluate [q]+negs in a batched env,
    compute an RL score per candidate, and aggregate MRR/Hits.
    """
    device = _device_of(actor)

    # Try to get IndexManager from parameter, base env, or sampler
    im = index_manager
    if im is None:
        # Try to access from sampler
        if hasattr(sampler, "index_manager"):
            im = sampler.index_manager
        elif hasattr(env, "index_manager"):
            im = env.index_manager
        # For ParallelEnv, we might need to get it from a different location
        elif hasattr(env, "_workers") and len(env._workers) > 0:
            # Try to get from first worker (this might not work with multiprocessing)
            pass
    
    if im is None:
        raise RuntimeError("IndexManager not found. Please pass index_manager parameter or ensure it's available in sampler/env.")

    MRRs, H1s, H3s, H10s = [], [], [], []
    success_flags = []
    episode_len_pos, episode_len_neg = [], []
    rewards_pos, rewards_neg = [], []

    # For each positive query build candidates
    for qi, q in enumerate(data):
        # --- build negatives ---
        # Convert the single query into sub_index tensor via IndexManager (API from your codebase)
        pos_sub_idx = im.get_atom_sub_index(q)    # shape: (1, padding_atoms, max_arity+1)
        neg_sub_indices = sampler.get_negatives(
            pos_sub_idx.unsqueeze(0),  # batch: (B=1, ...)
            n_corruptions=n_corruptions,
            corruption_scheme=corruption_scheme or ['head', 'tail'],
        )

        # Flatten negatives to a tensor of sub_indices
        if isinstance(neg_sub_indices, list):
            negs = torch.cat(neg_sub_indices, dim=0) if neg_sub_indices else pos_sub_idx.new_empty((0, *pos_sub_idx.shape[1:]))
        else:
            negs = neg_sub_indices

        # Compose candidate sub_indices: true + negs
        cand_sub = torch.cat([pos_sub_idx.unsqueeze(0), negs], dim=0)  # (C, padding_atoms, max_arity+1)
        # Convert to Term objects for env
        cand_terms = im.subindices_to_terms(cand_sub)  # -> List[Term] or List[List[Term]]
        labels = [1] + [0] * (len(cand_terms) - 1)
        depths = [data_depths[qi]] + [data_depths[qi]] * (len(cand_terms) - 1) if data_depths is not None else None

        # --- build a temporary eval env for these candidates ---
        batch_env = _make_eval_envs_like(env, cand_terms, labels, depths, mode="eval")

        # Ensure exactly one episode per env (we stop when all are done)
        td = batch_env.reset()
        n_envs = int(getattr(batch_env, "num_envs", 1))

        # Unwrap the actor for fast logits
        ac_model = _unwrap_actor_critic(actor)

        # accumulators
        logp_sum = torch.zeros(n_envs, device=device)
        ep_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
        ep_reward = torch.zeros(n_envs, device=device)
        finished = torch.zeros(n_envs, dtype=torch.bool)

        last_next_td = None
        while not finished.all():
            # forward
            sub_index = td["sub_index"].to(device, non_blocking=True)
            derived = td["derived_sub_indices"].to(device, non_blocking=True)
            mask = td["action_mask"].to(device, non_blocking=True)

            logits = ac_model.forward_actor(TensorDict({
                "sub_index": sub_index, "derived_sub_indices": derived, "action_mask": mask
            }, batch_size=torch.Size([n_envs])))

            if deterministic:
                action_idx, logp = _masked_argmax(logits, mask)
            else:
                masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
                masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                logp = dist.log_prob(action_idx)

            step_td = TensorDict({"action": action_idx.to(torch.long).cpu()}, batch_size=torch.Size([n_envs]))
            next_td = batch_env.step(step_td)
            nxt = next_td.get("next", next_td)

            r = nxt["reward"].reshape(n_envs, -1)[:, 0].to(device)
            d = nxt["done"].reshape(n_envs, -1)[:, 0].to(torch.bool)

            # update
            logp_sum += logp * (~finished).to(device)
            ep_len += (~finished).to(ep_len.dtype)
            ep_reward += r * (~finished).to(device)
            finished |= d

            td = nxt
            last_next_td = nxt

        # collect success flags (prefer next_td if provided)
        if last_next_td is None:
            last_next_td = td
        is_success = last_next_td.get("is_success", torch.zeros(n_envs, dtype=torch.bool)).to(torch.bool).to(device)

        # RL score
        scores = _score_from_traj(is_success, logp_sum)

        # rank of the true query (assume 0)
        rank = _rank_of_true(scores, true_index=0)
        MRRs.append(1.0 / rank)
        H1s.append(1.0 if rank <= 1 else 0.0)
        H3s.append(1.0 if rank <= 3 else 0.0)
        H10s.append(1.0 if rank <= 10 else 0.0)
        success_flags.append(float(is_success[0].item()))

        # collect simple aggregates for display parity
        # pos is env 0
        episode_len_pos.append(int(ep_len[0].item()))
        rewards_pos.append(float(ep_reward[0].item()))
        # aggregate negatives if present
        if n_envs > 1:
            neg_lens = ep_len[1:].float().mean().item()
            neg_rews = ep_reward[1:].float().mean().item()
            episode_len_neg.append(neg_lens)
            rewards_neg.append(neg_rews)

        # stream per-env episode info to callback
        if info_callback is not None:
            infos = []
            for j in range(n_envs):
                qd = depths[j] if depths is not None else None
                infos.append({"episode": {"r": float(ep_reward[j].item()), "l": int(ep_len[j].item())},
                              "label": int(labels[j]),
                              "query_depth": qd,
                              "is_success": bool(is_success[j].item())})
            info_callback(infos)

        # close the temporary env
        batch_env.close()

    # aggregate metrics
    metrics: Dict[str, Any] = dict(
        mrr_mean=float(np.mean(MRRs)) if MRRs else 0.0,
        hits1_mean=float(np.mean(H1s)) if H1s else 0.0,
        hits3_mean=float(np.mean(H3s)) if H3s else 0.0,
        hits10_mean=float(np.mean(H10s)) if H10s else 0.0,
        success_rate=float(np.mean(success_flags)) if success_flags else 0.0,
    )
    if episode_len_pos:
        metrics["episode_len_pos_mean"] = float(np.mean(episode_len_pos))
        metrics["episode_len_pos_std"] = float(np.std(episode_len_pos))
    if episode_len_neg:
        metrics["episode_len_neg_mean"] = float(np.mean(episode_len_neg))
        metrics["episode_len_neg_std"] = float(np.std(episode_len_neg))
    if rewards_pos:
        metrics["rewards_pos_mean"] = float(np.mean(rewards_pos))
        metrics["rewards_pos_std"] = float(np.std(rewards_pos))
    if rewards_neg:
        metrics["rewards_neg_mean"] = float(np.mean(rewards_neg))
        metrics["rewards_neg_std"] = float(np.std(rewards_neg))

    if verbose:
        print(f"[TorchRL eval-corr] Pos={len(data)} | "
              f"MRR={metrics['mrr_mean']:.4f} | H@1={metrics['hits1_mean']:.4f} | "
              f"H@3={metrics['hits3_mean']:.4f} | H@10={metrics['hits10_mean']:.4f} | "
              f"Succ={metrics['success_rate']:.3f}")

    return metrics


__all__ = [
    "TorchRLPolicyWrapper",
    "evaluate_policy_torchrl",
    "eval_corruptions_torchrl",
]
