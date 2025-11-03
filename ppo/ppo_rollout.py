
# ppo_rollout.py — fast rollout via TorchRL SyncDataCollector
# This file REPLACES the previous Python-loop rollout.

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector


def _reshape_time_env(td: TensorDict, n_steps: int, n_envs: int) -> TensorDict:
    """Reshape a flat [n_steps*n_envs] batch to [n_steps, n_envs]."""
    return td.reshape(n_steps, n_envs)


def _extract_episode_infos(batch_td: TensorDict, n_steps: int, n_envs: int, device: torch.device) -> List[Dict[str, Any]]:
    """Lightweight episode stats extraction compatible with your callbacks."""
    infos: List[Dict[str, Any]] = []
    done = batch_td.get(('next', 'done'))
    reward = batch_td.get(('next', 'reward'))
    label = batch_td.get(('next', 'label'), None)
    depth = batch_td.get(('next', 'query_depth'), None)
    success = batch_td.get(('next', 'is_success'), None)

    done = done.reshape(n_steps, n_envs).squeeze(-1).to(torch.bool)
    reward = reward.reshape(n_steps, n_envs).squeeze(-1)

    # For each env, accumulate until done
    ep_ret = torch.zeros(n_envs, device=device, dtype=reward.dtype)
    ep_len = torch.zeros(n_envs, device=device, dtype=torch.long)
    for t in range(n_steps):
        ep_ret += reward[t]
        ep_len += 1
        finished = done[t]
        if finished.any():
            idxs = torch.nonzero(finished, as_tuple=False).reshape(-1)
            for i in idxs.tolist():
                info = {
                    "episode": {"r": float(ep_ret[i].item()), "l": int(ep_len[i].item())}
                }
                if label is not None:
                    info["label"] = int(label[t * n_envs + i].item())
                if depth is not None:
                    info["query_depth"] = int(depth[t * n_envs + i].item())
                if success is not None:
                    info["is_success"] = bool(success[t * n_envs + i].item())
                infos.append(info)
                ep_ret[i] = 0.0
                ep_len[i] = 0
    return infos


@torch.inference_mode()
def collect_rollouts(
    env,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    n_envs: int,
    n_steps: int,
    device: torch.device,
    rollout_callback: Optional[Callable] = None,
) -> Tuple[List[TensorDict], Dict[str, Any]]:
    """Collect one batch (n_envs * n_steps) using SyncDataCollector.

    Returns
    -------
    experiences : List[TensorDict]
        Length n_steps; each TD has batch_size [n_envs] and contains:
          sub_index, derived_sub_indices, action_mask, action (one-hot),
          sample_log_prob, state_value, and a 'next' sub-TD with
          next/sub_index, next/derived_sub_indices, next/action_mask, next/reward, next/done.
    stats : Dict[str, Any]
        Contains 'episode_info' list for callbacks.
    """
    frames_per_batch = int(n_envs) * int(n_steps)

    # The actor is already a ProbabilisticActor that consumes obs and returns action+log_prob.
    collector = SyncDataCollector(
        env=env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch,
        device=device,
        storing_device='cpu',
        split_trajs=False,
        pin_memory=True,
        prefetch=2,
    )

    batch_td = next(iter(collector))  # single batch
    collector.shutdown()

    # Add value estimates using the critic (operates on the observation at ROOT)
    critic(batch_td)  # adds 'state_value'

    # Reshape to [n_steps, n_envs] and split to list as expected by PPOAgent.learn
    batch_td_time = _reshape_time_env(batch_td, n_steps=n_steps, n_envs=n_envs)
    experiences: List[TensorDict] = []
    for t in range(n_steps):
        step_td = TensorDict({
            "sub_index": batch_td_time[t]["sub_index"],
            "derived_sub_indices": batch_td_time[t]["derived_sub_indices"],
            "action_mask": batch_td_time[t]["action_mask"],
            "action": batch_td_time[t]["action"],  # OneHotCategorical -> one-hot
            "sample_log_prob": batch_td_time[t].get("sample_log_prob"),
            "state_value": batch_td_time[t].get("state_value"),
            "next": TensorDict({
                "sub_index": batch_td_time[t]["next"]["sub_index"],
                "derived_sub_indices": batch_td_time[t]["next"]["derived_sub_indices"],
                "action_mask": batch_td_time[t]["next"]["action_mask"],
                "reward": batch_td_time[t]["next"]["reward"],
                "done": batch_td_time[t]["next"]["done"],
            }, batch_size=[n_envs]),
        }, batch_size=[n_envs])
        experiences.append(step_td)

        if rollout_callback is not None:
            rollout_callback(t)

    # Stats for callbacks
    episode_info = _extract_episode_infos(batch_td, n_steps=n_steps, n_envs=n_envs, device=device)
    stats = {
        "episode_info": episode_info,
    }
    return experiences, stats
