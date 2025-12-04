# ======================== model_eval.py ========================
from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase

from callbacks import _format_stat_string, _format_depth_key

DEBUG_EVAL = os.environ.get("NGG_DEBUG_EVAL", "").lower() in {"1", "true", "yes"}


# ------------------------------------------------------------
# Trace Types for detailed step-by-step comparison
# ------------------------------------------------------------

class EvalStepTrace(TypedDict, total=False):
    """Trace of a single step in evaluate_policy."""
    step: int                    # Global step index
    env_idx: int                 # Environment index in the batch
    episode_idx: int             # Episode index for this env
    action: int                  # Action taken
    reward: float                # Reward received
    done: bool                   # Whether episode ended
    success: bool                # Whether proof succeeded (if applicable)
    log_prob: float              # Log probability of the action
    cumulative_reward: float     # Cumulative reward so far in episode
    episode_length: int          # Steps so far in episode
    state_obs: Dict[str, Any]    # State observation (sub_index, derived_states, etc.)
    next_state_obs: Dict[str, Any]  # Next state observation
    value: float                 # Value estimate (if available)
    query: List[int]             # Query tensor [rel, head, tail]


class EvalCorruptionsTrace(TypedDict, total=False):
    """Trace of eval_corruptions evaluation."""
    batch_idx: int               # Batch index
    mode: str                    # Corruption mode (head/tail)
    query_idx: int               # Query index within batch
    query: List[int]             # Query tensor [rel, head, tail]
    negatives: List[List[int]]   # Generated negative samples
    num_negatives: int           # Number of negatives generated
    pos_logp: float              # Log probability for positive
    neg_logps: List[float]       # Log probabilities for negatives
    pos_success: bool            # Whether positive proof succeeded
    neg_successes: List[bool]    # Whether negative proofs succeeded
    rank: int                    # Rank of the positive query
    episode_traces: List[EvalStepTrace]  # Detailed traces per episode


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def infer_batch_size(env: EnvBase) -> int:
    bs = getattr(env, "batch_size", None)
    if bs is None:
        td0 = env.reset()
        return int(td0.batch_size[0])
    if isinstance(bs, torch.Size):
        return int(bs[0]) if len(bs) else 1
    return int(bs)


# ------------------------------------------------------------
# Policy evaluation — single batched env
# ------------------------------------------------------------

@torch.inference_mode()
def evaluate_policy(
    actor: nn.Module,
    env:  EnvBase,
    *,
    n_eval_episodes: Optional[int] = None,
    target_episodes: Optional[Sequence[int]] = None,
    deterministic: bool = True,
    track_logprobs: bool = False,
    collect_action_stats: bool = False,
    info_callback: Optional[Callable[[TensorDict], None]] = None,
    verbose: int = 0,
    return_traces: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a policy on a single TorchRL env with internal batch dimension.
    Uses step_and_maybe_reset semantics (no invalid-action patching).
    Returns rewards/lengths/logps/success/mask tensors shaped [B, T].
    
    Args:
        actor: Policy network
        env: TorchRL environment
        n_eval_episodes: Number of episodes to evaluate (if target_episodes not provided)
        target_episodes: Per-env target episode counts [B]
        deterministic: Use deterministic actions
        track_logprobs: Track log probabilities (always True internally)
        collect_action_stats: Collect action statistics
        info_callback: Callback for step info
        verbose: Verbosity level
        return_traces: If True, collect detailed step-by-step traces for comparison
        
    Returns:
        Dict with rewards/lengths/logps/success/mask tensors, and optionally 'traces' list
    """
    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    B = infer_batch_size(env)
    if target_episodes is not None:
        targets = torch.as_tensor(target_episodes, dtype=torch.long, device=device)
        if targets.numel() != B:
            # Pad or trim to match batch size
            if targets.numel() < B:
                pad = torch.zeros(B - targets.numel(), dtype=torch.long, device=device)
                targets = torch.cat([targets, pad], dim=0)
            else:
                targets = targets[:B]
    else:
        targets = torch.full((B,), n_eval_episodes if n_eval_episodes is not None else 1, dtype=torch.long, device=device)

    actor_was_training = actor.training
    actor.eval()
    verbose_level = max(int(verbose), 1 if DEBUG_EVAL else 0)

    T = int(targets.max().item()) if targets.numel() > 0 else 0
    rewards  = torch.zeros((B, T), dtype=torch.float32, device=device)
    lengths  = torch.zeros((B, T), dtype=torch.long,   device=device)
    logps    = torch.zeros((B, T), dtype=torch.float32, device=device)
    success  = torch.zeros((B, T), dtype=torch.float32, device=device)
    mask     = torch.zeros((B, T), dtype=torch.bool,    device=device)

    ep_count   = torch.zeros(B, dtype=torch.long, device=device)
    ep_return  = torch.zeros(B, dtype=torch.float32, device=device)
    ep_length  = torch.zeros(B, dtype=torch.long,   device=device)
    ep_logprob = torch.zeros(B, dtype=torch.float32, device=device)

    # Trace collection
    traces: List[EvalStepTrace] = [] if return_traces else []
    global_step = 0

    td = env.reset().to(device, non_blocking=True)

    def _get_step_value(step_td: TensorDict, key: str, default: torch.Tensor) -> torch.Tensor:
        if key in step_td.keys():
            return step_td.get(key)
        nxt = step_td.get("next", None)
        if nxt is not None and key in nxt.keys():
            return nxt.get(key)
        return default
    
    def _extract_state_obs(td: TensorDict, env_idx: int) -> Dict[str, Any]:
        """Extract state observation for tracing."""
        obs = {}
        # Try to get sub_index (state representation)
        if "sub_index" in td.keys():
            sub_idx = td.get("sub_index")
            if sub_idx is not None:
                obs["sub_index"] = sub_idx[env_idx].cpu().tolist() if sub_idx.dim() > 1 else sub_idx.cpu().tolist()
        # Try to get derived_states
        if "derived_states" in td.keys():
            derived = td.get("derived_states")
            if derived is not None:
                obs["derived_states"] = derived[env_idx].cpu().tolist() if derived.dim() > 1 else derived.cpu().tolist()
        # Try to get derived_sub_indices (alternative name)
        if "derived_sub_indices" in td.keys():
            derived = td.get("derived_sub_indices")
            if derived is not None:
                obs["derived_sub_indices"] = derived[env_idx].cpu().tolist() if derived.dim() > 1 else derived.cpu().tolist()
        # Try to get action_mask
        if "action_mask" in td.keys():
            mask = td.get("action_mask")
            if mask is not None:
                obs["action_mask"] = mask[env_idx].cpu().tolist() if mask.dim() > 1 else mask.cpu().tolist()
        # Try to get query
        if "query" in td.keys():
            query = td.get("query")
            if query is not None:
                obs["query"] = query[env_idx].cpu().tolist() if query.dim() > 1 else query.cpu().tolist()
        # Try to get n_derived
        if "n_derived" in td.keys():
            n_derived = td.get("n_derived")
            if n_derived is not None:
                obs["n_derived"] = int(n_derived[env_idx].item()) if n_derived.dim() > 0 else int(n_derived.item())
        return obs

    if verbose_level > 0:
        init_done = _get_step_value(td, "done", torch.zeros(B, 1, dtype=torch.bool, device=device)).view(-1).tolist()
        print("[evaluate_policy] initial done flags:", init_done)

    while bool((ep_count < targets).any()):
        # Extract pre-step state for tracing
        pre_step_obs = {}
        if return_traces:
            for env_idx in range(B):
                if ep_count[env_idx] < targets[env_idx]:
                    pre_step_obs[env_idx] = _extract_state_obs(td, env_idx)
        
        actor_device = next(actor.parameters(), torch.zeros((), device=device)).device if isinstance(actor, nn.Module) else device
        policy_td = td.clone().to(actor_device)
        
        out = actor(policy_td, deterministic=deterministic)
        
        # Handle both tuple output (actions, values, log_probs) and TensorDict output
        value_estimates = torch.zeros(B, device=device)
        if isinstance(out, tuple):
            # Model returns (actions, values, log_probs) like SB3
            action = out[0].view(-1).long().to(device)
            if len(out) > 1 and out[1] is not None:
                value_estimates = out[1].view(-1).to(device)
            log_probs = out[2].view(-1).to(device) if len(out) > 2 else torch.zeros(B, device=device)
        elif isinstance(out, TensorDict):
            action = out.get("action").view(-1).long().to(device)
            log_probs = out.get("sample_log_prob", torch.zeros(B, device=device)).view(-1).to(device)
            if "state_value" in out.keys():
                value_estimates = out.get("state_value").view(-1).to(device)
        else:
            # Fallback: assume action is in policy_td
            action = policy_td.get("action").view(-1).long().to(device)
            log_probs = policy_td.get("sample_log_prob", torch.zeros(B, device=device)).view(-1).to(device)
        
        action_td = TensorDict({"action": action.to(device)}, batch_size=env.batch_size).to(device)
        
        # Step environment (without auto-reset, so we can control which slots reset)
        step_td = env.step(action_td)

        rew = _get_step_value(step_td, "reward", torch.zeros(B, device=device)).view(-1)
        done_curr = _get_step_value(step_td, "done", torch.zeros(B, 1, dtype=torch.bool, device=device)).view(-1)
        success_curr = _get_step_value(step_td, "is_success", torch.zeros(B, 1, dtype=torch.bool, device=device)).view(-1)
        length_curr = _get_step_value(step_td, "length", torch.zeros(B, 1, device=device, dtype=torch.long)).view(-1)

        ep_return += rew
        ep_length += 1
        ep_logprob += log_probs
        
        # Collect traces for this step
        if return_traces:
            next_obs_td = step_td.get("next", step_td)
            for env_idx in range(B):
                if ep_count[env_idx] < targets[env_idx]:
                    trace: EvalStepTrace = {
                        "step": global_step,
                        "env_idx": env_idx,
                        "episode_idx": int(ep_count[env_idx].item()),
                        "action": int(action[env_idx].item()),
                        "reward": float(rew[env_idx].item()),
                        "done": bool(done_curr[env_idx].item()),
                        "success": bool(success_curr[env_idx].item()),
                        "log_prob": float(log_probs[env_idx].item()),
                        "cumulative_reward": float(ep_return[env_idx].item()),
                        "episode_length": int(ep_length[env_idx].item()),
                        "state_obs": pre_step_obs.get(env_idx, {}),
                        "next_state_obs": _extract_state_obs(next_obs_td, env_idx),
                        "value": float(value_estimates[env_idx].item()),
                    }
                    # Add query if available
                    if "query" in pre_step_obs.get(env_idx, {}):
                        trace["query"] = pre_step_obs[env_idx]["query"]
                    traces.append(trace)
            global_step += 1

        need_more = ep_count < targets
        finished_rows = done_curr & need_more
        if finished_rows.any():
            rows = finished_rows.nonzero(as_tuple=False).view(-1)
            curr_ep_idx = ep_count[rows]
            rewards[rows, curr_ep_idx] = ep_return[rows]
            lengths[rows, curr_ep_idx] = ep_length[rows]
            logps[rows, curr_ep_idx] = ep_logprob[rows]
            success[rows, curr_ep_idx] = success_curr[rows].float()
            mask[rows, curr_ep_idx] = True
            ep_count[rows] += 1
            ep_return[rows] = 0
            ep_length[rows] = 0
            ep_logprob[rows] = 0
        
        # Reset only the environments that finished AND need more episodes
        # This prevents resetting slots that have completed their target
        next_obs = step_td.get("next", step_td)
        reset_needed = done_curr & need_more  # Only reset if we need more episodes
        if reset_needed.any():
            # After episode tracking update, need_more changes, so recompute
            # Slots that just finished but still need more episodes should reset
            still_need_more = ep_count < targets
            reset_mask = done_curr & still_need_more
            if reset_mask.any():
                reset_td = env.reset(TensorDict({"_reset": reset_mask.view(-1, 1)}, batch_size=env.batch_size))
                next_td = _merge_rows(next_obs.clone(), reset_td, reset_mask)
            else:
                next_td = next_obs
        else:
            next_td = next_obs
        
        td = next_td.to(device)

        if info_callback is not None:
            info_callback(step_td)

    if actor_was_training:
        actor.train()

    result = {
        "rewards": rewards.cpu(),
        "lengths": lengths.cpu(),
        "logps":   logps.cpu(),
        "success": success.cpu(),
        "mask":    mask.cpu(),
    }
    
    if return_traces:
        result["traces"] = traces
    
    return result


def step_and_maybe_reset(
    env: EnvBase,
    action_td: TensorDict,
    *,
    reset_rows: torch.Tensor,
) -> Tuple[TensorDict, TensorDict]:
    step_td = env.step(action_td)
    next_obs = step_td.get("next", step_td)
    if reset_rows.any():
        reset_td = env.reset(TensorDict({"_reset": reset_rows.view(-1, 1)}, batch_size=env.batch_size))
        next_td = _merge_rows(next_obs.clone(), reset_td, reset_rows)
    else:
        next_td = next_obs
    return step_td, next_td


def _merge_rows(dst: TensorDict, src: TensorDict, mask: torch.Tensor) -> TensorDict:
    mask = mask.view(-1)
    for k in src.keys():
        if k not in dst.keys():
            dst.set(k, src.get(k))
            continue
        dv, sv = dst.get(k), src.get(k)
        if dv.shape[0] == mask.shape[0]:
            merged = dv.clone()
            merged[mask] = sv[mask]
            dst.set(k, merged)
        else:
            dst.set(k, sv)
    return dst


# ------------------------------------------------------------
# Ranking metrics — SB3-style aggregation
# ------------------------------------------------------------

@torch.inference_mode()
def eval_corruptions(
    actor: nn.Module,
    env:   EnvBase,
    *,
    queries: torch.Tensor,
    sampler: Any,
    query_depths: Optional[torch.Tensor] = None,
    n_corruptions: Optional[int] = 10,
    corruption_modes: Sequence[str] = ("head", "tail"),
    deterministic: bool = True,
    verbose: bool = False,
    info_callback: Optional[Callable] = None,
    return_traces: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model by ranking a positive query vs. its corruptions.
    
    Args:
        actor: Policy network
        env: TorchRL environment
        queries: Query tensors [N, A, D] or [N, D]
        sampler: Negative sampler for generating corruptions
        query_depths: Depth values for queries
        n_corruptions: Number of corruptions per type (None for all)
        corruption_modes: Tuple of corruption modes ('head', 'tail')
        deterministic: Use deterministic actions
        verbose: Print progress
        info_callback: Callback for step info
        return_traces: If True, collect detailed traces for comparison
        
    Returns:
        Dict with MRR, Hits@K metrics, and optionally 'traces' list
    """
    env_device = getattr(env, "_device", None)
    actor_device = None
    try:
        actor_device = next(actor.parameters()).device
    except StopIteration:
        actor_device = None
    device = env_device or actor_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = infer_batch_size(env)
    if queries.ndim == 2:
        queries = queries.unsqueeze(1)
    elif queries.ndim != 3:
        raise ValueError(f"Expected queries with 2 or 3 dims, got shape {tuple(queries.shape)}")
    N = int(queries.shape[0])
    A, D = int(queries.shape[1]), int(queries.shape[2])
    
    # Trace collection for eval_corruptions
    all_traces: List[EvalCorruptionsTrace] = [] if return_traces else []

    def finalize(ranks: List[int]) -> Dict[str, float]:
        arr = np.asarray(ranks, dtype=np.int64)
        if arr.size == 0:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        return {
            "MRR": float(np.mean(1.0 / arr)),
            "Hits@1": float(np.mean(arr <= 1)),
            "Hits@3": float(np.mean(arr <= 3)),
            "Hits@10": float(np.mean(arr <= 10)),
        }

    per_mode_ranks: Dict[str, List[int]] = {m: [] for m in corruption_modes}
    episode_records: List[Tuple[bool, str, float, float, float]] = []

    actor_was_training = actor.training
    actor.eval()
    rng = np.random.RandomState(0)

    with torch.inference_mode():
        for start in range(0, N, B):
            Q = min(B, N - start)
            pos = queries[start:start + Q].to(device)
            pos_triples = pos.squeeze(1) if (A == 1 and D == 3) else pos

            # Generate head and tail corruptions separately upfront (SB3-style)
            # This ensures we use the same negatives generation pattern as SB3
            if n_corruptions == 0:
                head_corrs_list = [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
                tail_corrs_list = [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
            elif hasattr(sampler, 'get_negatives_from_states_separate'):
                # Use SB3-style separate generation
                head_corrs_list, tail_corrs_list = sampler.get_negatives_from_states_separate(
                    pos_triples,
                    num_negatives=n_corruptions,
                    device=device,
                )
            else:
                # Fallback: generate separately using corrupt method
                if n_corruptions is None:
                    head_corrs_list_raw, _ = sampler.corrupt_all(pos_triples, mode='head')
                    _, tail_corrs_list_raw = sampler.corrupt_all(pos_triples, mode='tail')
                    head_corrs_list = head_corrs_list_raw if head_corrs_list_raw else [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
                    tail_corrs_list = tail_corrs_list_raw if tail_corrs_list_raw else [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
                else:
                    K = int(n_corruptions)
                    head_neg = sampler.corrupt(pos_triples, num_negatives=K, mode='head').to(device)
                    tail_neg = sampler.corrupt(pos_triples, num_negatives=K, mode='tail').to(device)
                    head_corrs_list = [head_neg[i] for i in range(Q)]
                    tail_corrs_list = [tail_neg[i] for i in range(Q)]

            for mode in corruption_modes:
                # Select the appropriate corruption list based on mode
                if mode == "head":
                    corrs_list = head_corrs_list
                else:  # tail
                    corrs_list = tail_corrs_list
                
                # Convert to ragged lists format and count valid negatives
                ragged_lists = []
                lengths_i = []
                for i in range(Q):
                    neg_tensor = corrs_list[i]
                    if neg_tensor.numel() == 0:
                        ragged_lists.append(torch.empty((0, 1, 3) if (A == 1 and D == 3) else (0, 3), dtype=pos.dtype, device=device))
                        lengths_i.append(0)
                    else:
                        # Add dimension if needed
                        if A == 1 and D == 3 and neg_tensor.ndim == 2:
                            neg_tensor = neg_tensor.unsqueeze(1)
                        ragged_lists.append(neg_tensor.to(device))
                        # Count valid (non-padding) negatives
                        # Use SB3-compatible logic: valid if ALL fields are non-zero (padding_idx=0)
                        # A row with ANY zero field is considered padding
                        flat_neg = neg_tensor.view(neg_tensor.shape[0], -1)
                        is_valid = (flat_neg != 0).all(dim=1)  # Valid if ALL fields are non-zero
                        valid_count = int(is_valid.sum().item())
                        lengths_i.append(valid_count)


                per_slot_lengths = [1 + li for li in lengths_i]
                slot_starts = []
                running = 0
                for li in per_slot_lengths:
                    slot_starts.append(running)
                    running += li
                flat_list = []
                for i in range(Q):
                    flat_list.append(pos[i].unsqueeze(0))
                    if per_slot_lengths[i] > 1:
                        # Only append the valid negatives (exclude padding)
                        flat_list.append(ragged_lists[i][:lengths_i[i]])
                flat_queries = torch.cat(flat_list, dim=0)
                flat_labels = torch.cat([
                    torch.cat([torch.tensor([1], dtype=torch.long, device=device),
                               torch.zeros(per_slot_lengths[i]-1, dtype=torch.long, device=device)])
                    for i in range(Q)
                ])

                if query_depths is not None:
                    pos_depths = query_depths[start:start + Q].to(device)
                    depth_list = []
                    for i in range(Q):
                        d = pos_depths[i]
                        pos_d = torch.tensor([d], dtype=torch.long, device=device)
                        neg_d = torch.full((per_slot_lengths[i] - 1,), -1, dtype=torch.long, device=device)
                        depth_list.append(torch.cat([pos_d, neg_d]))
                    flat_depths = torch.cat(depth_list, dim=0)
                else:
                    flat_depths = torch.full((flat_queries.shape[0],), -1, dtype=torch.long, device=device)

                slot_lengths = torch.tensor(per_slot_lengths + [0] * (B - Q), dtype=torch.long, device=device)
                env.set_eval_dataset(
                    queries=flat_queries,
                    labels=flat_labels,
                    query_depths=flat_depths,
                    per_slot_lengths=slot_lengths,
                )

                targets = slot_lengths.clone().tolist()
                out = evaluate_policy(
                    actor,
                    env,
                    target_episodes=targets,
                    deterministic=deterministic,
                    track_logprobs=True,
                    info_callback=info_callback,
                    return_traces=return_traces,
                )
                
                logps_out = out["logps"].to(device)
                msk = out["mask"].to(device)
                success = out.get("success")
                lengths_out = out.get("lengths")
                rewards_out = out.get("rewards")
                episode_traces = out.get("traces", []) if return_traces else []

                labels_matrix = torch.zeros_like(logps_out, dtype=torch.bool)
                for i, Ei in enumerate(per_slot_lengths):
                    if Ei == 0:
                        continue
                    labels_matrix[i, 0] = True
                    labels_matrix[i, 1:Ei] = False

                valid_mask = torch.zeros_like(msk, dtype=torch.bool)
                for i, Ei in enumerate(per_slot_lengths):
                    if Ei == 0:
                        continue
                    valid_mask[i, : min(Ei, valid_mask.shape[1])] = True

                Tmax = logps_out.shape[1]
                if success is not None:
                    # For penalty: penalize ALL unsuccessful proofs (matching SB3 behavior)
                    success_for_penalty = success.to(device).bool()
                    # For reporting: track which positive queries succeeded
                    success_mask = success.to(device).bool()
                else:
                    success_for_penalty = torch.ones_like(logps_out, dtype=torch.bool)
                    success_mask = None

                logps_out = logps_out.clone()
                logps_out[~success_for_penalty] -= 100.0

                # Align RNG with SB3 for parity in tie-breaking
                rng = np.random.RandomState(0)
                # SB3 generates random keys for the whole batch (B, Tmax)
                # We must match this generation order.
                batch_random_keys = rng.rand(Q, Tmax)

                ranks_this = []
                for i in range(Q):
                    Ei = per_slot_lengths[i]
                    cols = slice(0, min(Ei, Tmax))
                    l_i = logps_out[i, cols].cpu().numpy()
                    m_i = msk[i, cols].cpu().numpy()
                    lp_batch = np.where(m_i, l_i, -np.inf)
                    
                    # Use pre-generated keys for this row
                    random_keys = batch_random_keys[i, cols]
                    
                    sorted_indices = np.lexsort((-random_keys, -lp_batch))
                    rank = np.where(sorted_indices == 0)[0][0] + 1
                    ranks_this.append(rank)
                    
                    # Collect trace for this query
                    if return_traces:
                        # Extract positive query
                        pos_query = pos[i].squeeze().cpu().tolist() if pos[i].dim() > 1 else pos[i].cpu().tolist()
                        
                        # Extract negatives for this query
                        neg_list = []
                        if lengths_i[i] > 0:
                            neg_tensor = ragged_lists[i][:lengths_i[i]]
                            for j in range(min(lengths_i[i], neg_tensor.shape[0])):
                                neg_triple = neg_tensor[j].squeeze().cpu().tolist() if neg_tensor[j].dim() > 1 else neg_tensor[j].cpu().tolist()
                                neg_list.append(neg_triple)
                        
                        # Extract log probs (pos is at index 0, negs follow)
                        pos_logp = float(logps_out[i, 0].cpu().item()) if Ei > 0 else float('-inf')
                        neg_logps_list = [float(logps_out[i, j].cpu().item()) for j in range(1, min(Ei, Tmax))]
                        
                        # Extract success flags
                        pos_succ = bool(success_mask[i, 0].cpu().item()) if success_mask is not None and Ei > 0 else False
                        neg_succs = [bool(success_mask[i, j].cpu().item()) for j in range(1, min(Ei, Tmax))] if success_mask is not None else []
                        
                        # Filter episode traces for this query (env_idx == i)
                        query_episode_traces = [t for t in episode_traces if t.get("env_idx") == i]
                        
                        trace: EvalCorruptionsTrace = {
                            "batch_idx": start // B,
                            "mode": mode,
                            "query_idx": start + i,
                            "query": pos_query,
                            "negatives": neg_list,
                            "num_negatives": lengths_i[i],
                            "pos_logp": pos_logp,
                            "neg_logps": neg_logps_list,
                            "pos_success": pos_succ,
                            "neg_successes": neg_succs,
                            "rank": rank,
                            "episode_traces": query_episode_traces,
                        }
                        all_traces.append(trace)

                per_mode_ranks[mode].extend(ranks_this)

                if lengths_out is not None and rewards_out is not None:
                    lengths_out = lengths_out.to(device)
                    rewards_out = rewards_out.to(device)
                    for i in range(Q):
                        base_idx = slot_starts[i] if i < len(slot_starts) else 0
                        depth_val = int(flat_depths[base_idx].item()) if per_slot_lengths[i] > 0 else -1
                        for j in range(min(per_slot_lengths[i], lengths_out.shape[1])):
                            if not valid_mask[i, j]:
                                continue
                            is_pos = bool(labels_matrix[i, j].item())
                            depth_key = _format_depth_key(depth_val if is_pos else -1)
                            episode_records.append((
                                is_pos,
                                depth_key,
                                float(lengths_out[i, j].item()),
                                float(rewards_out[i, j].item()),
                                float(success_mask[i, j].item() if success_mask is not None else 0.0),
                            ))

                if verbose:
                    print(f"[batch {start//B:03d} | mode={mode}] Q={Q} mean_rank={np.mean(ranks_this):.2f}  MRR={np.mean(1/np.asarray(ranks_this)):.3f}")

    if actor_was_training:
        actor.train()

    per_mode = {m: finalize(per_mode_ranks[m]) for m in corruption_modes}
    agg = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    for m in corruption_modes:
        for k, v in per_mode[m].items():
            agg[k] += v
    for k in agg:
        agg[k] /= float(len(corruption_modes)) if corruption_modes else 1.0
    agg["per_mode"] = per_mode
    agg["_mrr"] = agg["MRR"]

    if episode_records:
        lengths_all = [e[2] for e in episode_records]
        rewards_all = [e[3] for e in episode_records]
        succ_all = [e[4] for e in episode_records]
        if lengths_all:
            agg["length mean +/- std"] = _format_stat_string(np.mean(lengths_all), np.std(lengths_all), len(lengths_all))
            agg["ep_len_mean"] = float(np.mean(lengths_all))
        if rewards_all:
            agg["reward_overall"] = _format_stat_string(np.mean(rewards_all), np.std(rewards_all), len(rewards_all))
            agg["ep_rew_mean"] = float(np.mean(rewards_all))
        if succ_all:
            agg["success_rate"] = float(np.mean(succ_all))

        for lbl_bool, lbl_key in [(True, "pos"), (False, "neg")]:
            items = [e for e in episode_records if e[0] == lbl_bool]
            lens = [e[2] for e in items]
            rews = [e[3] for e in items]
            succs = [e[4] for e in items]
            if lens:
                agg[f"len_{lbl_key}"] = _format_stat_string(np.mean(lens), np.std(lens), len(lens))
            if succs:
                agg[f"proven_{lbl_key}"] = _format_stat_string(np.mean(succs), np.std(succs), len(succs))
            if rews:
                agg[f"reward_label_{lbl_key}"] = _format_stat_string(np.mean(rews), np.std(rews), len(rews))

        depth_buckets: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
        for is_pos, depth_key, length_val, reward_val, succ_val in episode_records:
            lbl_key = "pos" if is_pos else "neg"
            depth_buckets.setdefault((depth_key, lbl_key), []).append((length_val, reward_val, succ_val))
        for (depth_key, lbl_key), vals in depth_buckets.items():
            lens = [v[0] for v in vals]
            rews = [v[1] for v in vals]
            succs = [v[2] for v in vals]
            if lens:
                agg[f"len_d_{depth_key}_{lbl_key}"] = _format_stat_string(np.mean(lens), np.std(lens), len(lens))
            if succs:
                agg[f"proven_d_{depth_key}_{lbl_key}"] = _format_stat_string(np.mean(succs), np.std(succs), len(succs))
            if rews:
                agg[f"reward_d_{depth_key}_{lbl_key}"] = _format_stat_string(np.mean(rews), np.std(rews), len(rews))

    # Add traces if requested
    if return_traces:
        agg["traces"] = all_traces

    return agg
