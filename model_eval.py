# ======================== model_eval.py ========================
from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase

DEBUG_EVAL = os.environ.get("NGG_DEBUG_EVAL", "").lower() in {"1", "true", "yes"}


# ------------------------------------------------------------
# Small utilities (single batched env only)
# ------------------------------------------------------------

def infer_batch_size(env: EnvBase) -> int:
    """Return the env's batch size (B) for a single TorchRL batched env."""
    bs = getattr(env, "batch_size", None)
    if bs is None:
        td0 = env.reset()
        return int(td0.batch_size[0])
    if isinstance(bs, torch.Size):
        return int(bs[0]) if len(bs) else 1
    return int(bs)


def compute_episode_targets(
    batch_size: int,
    n_eval_episodes: Optional[int],
    target_episodes: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
      targets [B], max_t, total
    """
    if target_episodes is not None:
        t = torch.as_tensor(target_episodes, dtype=torch.long)
        if t.numel() < batch_size:
            t = torch.cat([t, torch.zeros(batch_size - t.numel(), dtype=torch.long)], 0)
        elif t.numel() > batch_size:
            t = t[:batch_size]
        targets = t
    else:
        n_total = int(n_eval_episodes or 0)
        if n_total <= 0:
            raise ValueError("Provide n_eval_episodes > 0 or target_episodes per slot.")
        base, rem = divmod(n_total, batch_size)
        targets = torch.full((batch_size,), base, dtype=torch.long)
        if rem:
            targets[:rem] += 1

    max_t = int(targets.max().item()) if targets.numel() else 0
    total = int(targets.sum().item())
    return targets, max_t, total



def _merge_rows(dst: TensorDict, src: TensorDict, mask: torch.Tensor) -> TensorDict:
    """
    For every key present in src, overwrite rows where mask==True into dst.
    Assumes first dimension is the batch dimension.
    """
    if not mask.any():
        return dst
    for k in src.keys():
        sv = src.get(k)
        if k in dst.keys():
            dv = dst.get(k)
            # If batched on first dim, scatter rows; else replace whole tensor
            if dv.shape[:1] == mask.shape[:1]:
                dv = dv.clone()
                dv[mask] = sv[mask]
                dst.set(k, dv)
            else:
                dst.set(k, sv)
        else:
            dst.set(k, sv)
    return dst


def step_and_maybe_reset(
    env: EnvBase,
    action_td: TensorDict,
    *,
    reset_rows: torch.Tensor,          # boolean [B]: which rows STILL need episodes and finished now
) -> Tuple[TensorDict, TensorDict]:
    """
    One step, then partial reset ONLY for rows that still need work.
    Returns (step_result_td, next_observation_td).
    """
    step_td = env.step(action_td)  # -> contains reward/done/terminated/truncated + next obs under "next" key
    
    # Extract next observation from the "next" key (TorchRL convention)
    if "next" in step_td.keys():
        next_obs = step_td.get("next")
    else:
        # If no "next" key, the obs might be directly in step_td (older TorchRL or custom env)
        next_obs = step_td
    
    if reset_rows.any():
        reset_td = env.reset(TensorDict({"_reset": reset_rows.view(-1, 1)}, batch_size=env.batch_size))
        # Build next obs by merging reset obs into step output
        next_td = _merge_rows(next_obs.clone(), reset_td, reset_rows)
    else:
        next_td = next_obs
    return step_td, next_td


# ------------------------------------------------------------
# 1) Policy evaluation — single batched env
# ------------------------------------------------------------

@torch.inference_mode()
def evaluate_policy(
    actor: nn.Module,
    env:  EnvBase,
    *,
    n_eval_episodes: Optional[int] = None,
    target_episodes: Optional[Sequence[int]] = None,
    deterministic: bool = True,   # kept for API symmetry; use inside your actor if needed
    track_logprobs: bool = False, # left in for easy toggling; not required for ranking
    info_callback: Optional[Callable[[TensorDict], None]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate a policy on a single TorchRL env with internal batch dimension.
    Uses step_and_maybe_reset semantics (no invalid-action patching).
    """
    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    B = infer_batch_size(env)
    targets, T, total = compute_episode_targets(B, n_eval_episodes, target_episodes)
    targets = targets.to(device)  # Move targets to device

    actor_was_training = actor.training
    actor.eval()

    # Output buffers
    rewards  = torch.zeros((B, T), dtype=torch.float32, device=device)
    lengths  = torch.zeros((B, T), dtype=torch.long,   device=device)
    logps    = torch.zeros((B, T), dtype=torch.float32, device=device)
    success  = torch.zeros((B, T), dtype=torch.float32, device=device)
    mask     = torch.zeros((B, T), dtype=torch.bool,    device=device)

    # Per-slot accumulators
    ep_count   = torch.zeros(B, dtype=torch.long, device=device)
    ep_return  = torch.zeros(B, dtype=torch.float32, device=device)
    ep_length  = torch.zeros(B, dtype=torch.long,   device=device)
    ep_logprob = torch.zeros(B, dtype=torch.float32, device=device)

    # Full reset to start
    td = env.reset().to(device, non_blocking=True)

    def _get_step_value(step_td: TensorDict, key: str, default: torch.Tensor) -> torch.Tensor:
        if key in step_td.keys():
            return step_td.get(key)
        nxt = step_td.get("next", None)
        if nxt is not None and key in nxt.keys():
            return nxt.get(key)
        return default

    if DEBUG_EVAL:
        init_done_vals = _get_step_value(
            td,
            "done",
            torch.zeros(B, 1, dtype=torch.bool, device=device),
        ).view(-1).tolist()
        print("[evaluate_policy] initial done flags:", init_done_vals)

    prev_done = torch.zeros(B, dtype=torch.bool, device=device)

    while bool((ep_count < targets).any()):
        # 1) Policy
        out = actor(td)
        action = out.get("action")
        action = action.view(-1).long()
        action_td = TensorDict({"action": action}, batch_size=env.batch_size).to(device)

        log_probs = out.get("log_prob", torch.zeros(B, device=device)).view(-1)
        # 2) Step + maybe partial reset ONLY where we still need episodes
        done_prev = prev_done
        need_more = ep_count < targets
        if DEBUG_EVAL:
            print(
                "[evaluate_policy] pre-step done=",
                done_prev.tolist(),
                "need_more=",
                need_more.tolist(),
            )
        reset_rows = done_prev & need_more  # rows that finished an episode and still need work
        if DEBUG_EVAL and reset_rows.any():
            print(
                "[evaluate_policy] reset_rows=",
                reset_rows.nonzero(as_tuple=False).view(-1).tolist(),
                "ep_count=",
                ep_count.tolist(),
                "targets=",
                targets.tolist(),
            )

        step_td, next_td = step_and_maybe_reset(env, action_td, reset_rows=reset_rows)
        if info_callback is not None:
            info_callback(step_td)

        # 3) Accumulate
        r = _get_step_value(step_td, "reward", torch.zeros(B, 1, device=device)).view(-1)
        done_curr = _get_step_value(
            step_td,
            "done",
            torch.zeros(B, 1, dtype=torch.bool, device=device),
        ).view(-1).bool()
        if DEBUG_EVAL:
            print("[evaluate_policy] post-step done=", done_curr.tolist())
        suc = _get_step_value(step_td, "is_success", torch.zeros(B, device=device)).view(-1).float()

        active = ep_count < targets
        ep_return[active] += r[active]
        ep_length[active] += 1
        if track_logprobs:
            ep_logprob += log_probs * active


        # Close finished episodes (only for active rows)
        finished_now = done_curr & active
        if finished_now.any():
            if DEBUG_EVAL:
                print(
                    "[evaluate_policy] finished_rows=",
                    finished_now.nonzero(as_tuple=False).view(-1).tolist(),
                    "ep_count=",
                    ep_count.tolist(),
                    "targets=",
                    targets.tolist(),
                )
            rows = finished_now.nonzero(as_tuple=False).view(-1)
            cols = ep_count[rows]
            rewards[rows, cols] = ep_return[rows]
            lengths[rows, cols] = ep_length[rows]
            success[rows, cols] = suc[rows]
            mask[rows, cols]    = True
            logps[rows, cols] = ep_logprob[rows]

            ep_return[rows] = 0.0
            ep_length[rows] = 0
            ep_count[rows]  += 1
            ep_logprob[rows] = 0.0

        td = next_td  # observation for next step
        prev_done = done_curr.clone()
        prev_done[reset_rows] = False

        if not (ep_count < targets).any():
            break

    if actor_was_training:
        actor.train()

    return {
        "rewards": rewards.cpu(),
        "lengths": lengths.cpu(),
        "logps":   logps.cpu(),
        "success": success.cpu(),
        "mask":    mask.cpu(),
    }


# ------------------------------------------------------------
# 2) Ranking metrics — handles ragged negatives (corrupt_all)
# ------------------------------------------------------------

@torch.inference_mode()
def evaluate_ranking_metrics(
    actor: nn.Module,
    env:   EnvBase,
    *,
    queries: torch.Tensor,                     # [N, A, D] padded positives
    sampler: Any,                              # must provide corrupt() and/or corrupt_all()
    n_corruptions: Optional[int] = 10,         # if None -> use corrupt_all (ragged)
    corruption_modes: Sequence[str] = ("head", "tail"),
    deterministic: bool = True,
    verbose: bool = False,
    score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Link-prediction style eval (MRR/Hits@K) on a single batched env.

    Packing per minibatch × corruption mode:
      - Pin one root to one env slot.
      - For slot i, run E_i = 1 + (#negatives for that root) episodes.
      - Provide per-slot lengths to env via set_eval_dataset(..., per_slot_lengths=...).
      - Provide per-slot episode targets to evaluate_policy(target_episodes=...).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B      = infer_batch_size(env)
    if queries.ndim == 2:
        queries = queries.unsqueeze(1)  # [N, 3] -> [N, 1, 3]
    elif queries.ndim != 3:
        raise ValueError(f"Expected queries with 2 or 3 dims, got shape {tuple(queries.shape)}")
    N      = int(queries.shape[0])
    A, D   = int(queries.shape[1]), int(queries.shape[2])

    def default_score(succ: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        if succ.numel() == 0:
            return succ
        max_len = torch.clamp(lens.max(), min=1)
        return 2.0 * succ.float() - (lens.float() / (max_len + 1.0))
    score_fn = score_fn or default_score

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

    actor_was_training = actor.training
    actor.eval()

    with torch.inference_mode():
        for start in range(0, N, B):
            Q = min(B, N - start)
            pos = queries[start:start + Q].to(device)   # [Q, A, D]
            
            # For single-atom queries (link prediction), squeeze to [Q, 3] for sampler
            if A == 1 and D == 3:
                pos_triples = pos.squeeze(1)  # [Q, 3]
            else:
                pos_triples = pos

            for mode in corruption_modes:
                # ----------------------------
                # Negatives (fixed-K or ragged)
                # ----------------------------
                if n_corruptions is None:
                    if not hasattr(sampler, "corrupt_all"):
                        raise NotImplementedError("n_corruptions=None requires sampler.corrupt_all(...)")
                    
                    # corrupt_all returns (heads_list, tails_list) where each list contains [K_i, 3] tensors
                    heads_list, tails_list = sampler.corrupt_all(pos_triples, mode=mode)
                    
                    # Extract the appropriate list based on mode
                    if mode == 'head':
                        raw_lists = heads_list
                    elif mode == 'tail':
                        raw_lists = tails_list
                    else:
                        raise ValueError(f"Unsupported mode for corrupt_all: {mode}")
                    
                    # Convert [K_i, 3] to [K_i, 1, 3] to match padded format if needed
                    if A == 1 and D == 3:
                        ragged_lists = [t.unsqueeze(1).to(device) for t in raw_lists]  # [K_i, 3] -> [K_i, 1, 3]
                    else:
                        ragged_lists = [t.to(device) for t in raw_lists]
                    lengths_i = [int(t.shape[0]) for t in ragged_lists]
                else:
                    K = int(n_corruptions)
                    neg = sampler.corrupt(pos_triples, num_negatives=K, mode=mode).to(device)  # [Q, K, 3]
                    # Convert [Q, K, 3] to [Q, K, 1, 3] to match padded format if needed
                    if A == 1 and D == 3:
                        neg = neg.unsqueeze(2)  # [Q, K, 3] -> [Q, K, 1, 3]
                    ragged_lists = [neg[i] for i in range(Q)]
                    lengths_i = [K for _ in range(Q)]

                # ----------------------------
                # Build per-slot sequences
                # ----------------------------
                # For each slot i: sequence = [positive] + all its negatives
                per_slot_lengths = [1 + li for li in lengths_i]          # E_i
                # Flatten as concatenation of per-slot sequences (slot 0 block, slot 1 block, ...)
                flat_list = []
                for i in range(Q):
                    flat_list.append(pos[i].unsqueeze(0))                # [1, A, D]
                    if per_slot_lengths[i] > 1:
                        flat_list.append(ragged_lists[i])                # [K_i, A, D]
                flat_queries = torch.cat(flat_list, dim=0)               # [sum_i E_i, A, D]
                flat_labels  = torch.cat([
                    torch.cat([
                        torch.tensor([1], dtype=torch.long, device=device),
                        torch.zeros(per_slot_lengths[i]-1, dtype=torch.long, device=device)
                    ]) for i in range(Q)
                ])  # [sum_i E_i]

                # Depths (optional; set -1)
                flat_depths = torch.full((flat_queries.shape[0],), -1, dtype=torch.long, device=device)

                # ----------------------------
                # Wire dataset into the env
                # ----------------------------
                # Provide per-slot lengths for the first Q slots; others are 0.
                slot_lengths = torch.tensor(
                    per_slot_lengths + [0] * (B - Q),
                    dtype=torch.long, device=device
                )
                if hasattr(env, "set_eval_dataset") and callable(getattr(env, "set_eval_dataset")):
                    env.set_eval_dataset(
                        queries=flat_queries,
                        labels=flat_labels,
                        query_depths=flat_depths,
                        per_slot_lengths=slot_lengths,
                    )
                else:
                    raise RuntimeError("Your env must implement set_eval_dataset(...). See snippet below.")

                # Targets = per-slot episode quotas (same as per-slot lengths)
                targets = slot_lengths.clone().tolist()

                # ----------------------------
                # Run evaluation
                # ----------------------------
                out = evaluate_policy(
                    actor,
                    env,
                    target_episodes=targets,
                    deterministic=deterministic,
                    track_logprobs=False,
                )
                succ = out["success"].to(device)   # [B, T]
                lens = out["lengths"].to(device)   # [B, T]
                msk  = out["mask"].to(device)      # [B, T]
                Tmax = succ.shape[1]

                # Only first Q slots matter; each has E_i valid columns
                ranks_this = []
                for i in range(Q):
                    Ei = per_slot_lengths[i]
                    cols = slice(0, min(Ei, Tmax))
                    s_i = succ[i, cols]
                    l_i = lens[i, cols]
                    m_i = msk[i, cols]
                    # scores (higher is better)
                    scores = score_fn(s_i, l_i)
                    scores = torch.where(m_i, scores, torch.full_like(scores, -1e9))
                    pos_score = scores[0]
                    better = (scores[1:] > pos_score).sum().item()
                    rank = 1 + int(better)
                    ranks_this.append(rank)

                per_mode_ranks[mode].extend(ranks_this)

                if verbose:
                    print(f"[batch {start//B:03d} | mode={mode}] Q={Q} "
                          f"mean_rank={np.mean(ranks_this):.2f}  "
                          f"MRR={np.mean(1/np.asarray(ranks_this)):.3f}")

    if actor_was_training:
        actor.train()

    # aggregate metrics
    per_mode = {m: finalize(per_mode_ranks[m]) for m in corruption_modes}
    agg = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    for m in corruption_modes:
        for k, v in per_mode[m].items():
            agg[k] += v
    for k in agg:
        agg[k] /= float(len(corruption_modes)) if corruption_modes else 1.0
    agg["per_mode"] = per_mode
    return agg
