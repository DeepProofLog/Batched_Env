"""
Shared utilities for callbacks in both TorchRL and Stable-Baselines3 implementations.

This module provides common functionality for collecting and formatting detailed
episode metrics broken down by label (positive/negative) and depth.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _format_depth_key(depth_value: Any) -> str:
    """
    Normalize depth IDs so metrics share consistent naming.
    
    Args:
        depth_value: Raw depth value (int, None, -1, etc.)
        
    Returns:
        Normalized depth string ("0", "1", "2", ..., or "unknown")
    """
    if depth_value in (None, -1):
        return "unknown"
    try:
        return str(int(depth_value))
    except (TypeError, ValueError):
        return "unknown"


def _format_stat_string(mean: Optional[float], std: Optional[float], count: int) -> str:
    """
    Format statistics as a display string.
    
    Args:
        mean: Mean value
        std: Standard deviation
        count: Number of samples
        
    Returns:
        Formatted string "mean +/- std (count)" with fixed precision
    """
    if mean is None or std is None or count == 0:
        return "N/A"
    return f"{mean:.3f} +/- {std:.2f} ({count})"


def _sort_metric_key(key: str) -> Tuple[int, Union[int, float]]:
    """
    Generate sort key for depth-based metrics.
    
    Sorts metrics by:
    1. Label (pos before neg)
    2. Depth (0, 1, 2, ..., unknown last)
    
    Args:
        key: Metric name like "len_d_0_pos" or "proven_d_1_neg"
        
    Returns:
        Tuple of (label_order, depth_order) for sorting
    """
    parts = key.split('_')
    label = parts[-1]  # 'pos' or 'neg'
    label_order = 0 if label == 'pos' else 1
    
    # Extract depth number - should be parts[2] for format like "len_d_0_pos"
    depth_str = parts[2] if len(parts) > 2 else "unknown"
    if depth_str == "unknown":
        depth_order: Union[int, float] = float("inf")
    else:
        try:
            depth_order = int(depth_str)
        except (TypeError, ValueError):
            depth_order = float("inf")
    
    return (label_order, depth_order)


def print_formatted_metrics(
    metrics: Dict[str, Any],
    prefix: str = "rollout",
    extra_metrics: Optional[Dict[str, Any]] = None,
    global_step: Optional[int] = None,
) -> None:
    """
    Print metrics in a formatted table.
    
    This is a common function used by both training and evaluation callbacks
    to ensure consistent formatting.
    
    Args:
        metrics: Dictionary of rollout/episode metrics from DetailedMetricsCollector
        prefix: Prefix for the metrics section (e.g., "rollout", "eval")
        extra_metrics: Optional additional metrics to display (e.g., MRR, AUC, train losses)
        global_step: Optional global step/timestep to display
    """
    print("-" * 52)
    
    # Priority metrics (MRR, AUC, etc.) if provided in extra_metrics
    if extra_metrics:
        priority_keys = []
        for key in ["mrr_mean", "_mrr", "auc_pr", "auc_pr_mean", "success_rate"]:
            if key in extra_metrics:
                priority_keys.append(key)
        
        if priority_keys:
            print(f"| {prefix + '/':<23} | {'':<24} |")
            for key in priority_keys:
                value = extra_metrics[key]
                value_str = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                print(f"|    {key:<20} | {value_str:<24} |")
    
    # Rollout/episode metrics from DetailedMetricsCollector
    if metrics:
        if not extra_metrics or not any(k in extra_metrics for k in ["mrr_mean", "_mrr", "auc_pr"]):
            print(f"| {prefix + '/':<23} | {'':<24} |")
        
        # 1. Overall episode stats
        for key in ["ep_len", "ep_rew"]:
            if key in metrics:
                print(f"|    {key:<20} | {metrics[key]:<24} |")
        
        # 2. Aggregate by label
        for key in ["len_pos", "len_neg", "proven_pos", "proven_neg", "rwd_pos", "rwd_neg"]:
            if key in metrics:
                print(f"|    {key:<20} | {metrics[key]:<24} |")
        
        # 3. Detailed breakdown by depth (if present)
        # Group by metric type for cleaner display
        len_depth_keys = sorted(
            [k for k in metrics.keys() if k.startswith("len_d_")],
            key=_sort_metric_key
        )
        proven_depth_keys = sorted(
            [k for k in metrics.keys() if k.startswith("proven_d_")],
            key=_sort_metric_key
        )
        rwd_depth_keys = sorted(
            [k for k in metrics.keys() if k.startswith("rwd_d_")],
            key=_sort_metric_key
        )
        
        for key in len_depth_keys:
            print(f"|    {key:<20} | {metrics[key]:<24} |")
        for key in proven_depth_keys:
            print(f"|    {key:<20} | {metrics[key]:<24} |")
        for key in rwd_depth_keys:
            print(f"|    {key:<20} | {metrics[key]:<24} |")
    
    # Additional metrics (time, train losses, etc.)
    if extra_metrics:
        # Time metrics
        time_keys = [k for k in extra_metrics.keys() if k in ["fps", "iterations", "total_timesteps"]]
        if time_keys:
            print(f"| {'time/':<23} | {'':<24} |")
            for key in time_keys:
                value = extra_metrics[key]
                value_str = str(value)
                print(f"|    {key:<20} | {value_str:<24} |")
        
        # Other metrics (losses, entropy, etc. for training; hits, episode breakdown for evaluation)
        # Only show under "train/" prefix if we're actually in training context (prefix != "eval")
        other_keys = [k for k in extra_metrics.keys() 
                     if k not in time_keys and k not in ["mrr_mean", "_mrr", "auc_pr", "auc_pr_mean", "success_rate"]]
        if other_keys:
            # Don't add "train/" prefix during evaluation
            if prefix == "eval":
                print(f"| {prefix + '/':<23} | {'':<24} |")
            else:
                print(f"| {'train/':<23} | {'':<24} |")
            for key in other_keys:
                value = extra_metrics[key]
                value_str = str(value)
                print(f"|    {key:<20} | {value_str:<24} |")
    
    # Global step if provided
    if global_step is not None:
        if not extra_metrics or "total_timesteps" not in extra_metrics:
            print(f"|    {'total_timesteps':<20} | {global_step:<24} |")
    
    print("-" * 52)
    print()


class DetailedMetricsCollector:
    """
    Common module for collecting detailed episode metrics by label and depth.
    
    This collector tracks episode-level statistics (reward, length, success)
    broken down by:
    - Label (positive/negative)
    - Depth (0, 1, 2, ..., unknown)
    - Combinations (e.g., len_d_0_pos, proven_d_1_neg, rwd_d_2_pos)
    
    Can be used by both training and evaluation callbacks to provide consistent
    detailed information when enabled.
    
    Example metrics generated:
        - len_pos, len_neg
        - len_d_0_pos, len_d_1_pos, ..., len_unknown_pos
        - len_d_0_neg, len_d_1_neg, ..., len_unknown_neg
        - proven_pos, proven_neg
        - proven_d_0_pos, proven_d_1_pos, ..., proven_unknown_pos
        - proven_d_0_neg, proven_d_1_neg, ..., proven_unknown_neg
        - rwd_pos, rwd_neg (aggregate)
        - rwd_d_0_pos, rwd_d_1_pos, ..., rwd_unknown_pos
        - rwd_d_0_neg, rwd_d_1_neg, ..., rwd_unknown_neg
    """
    
    def __init__(self, collect_detailed: bool = True):
        """
        Args:
            collect_detailed: If True, collect detailed breakdown by depth.
                            If False, only collect aggregate stats by label.
        """
        self.collect_detailed = collect_detailed
        self.reset()
    
    def reset(self):
        """Clear all accumulated statistics."""
        # Store raw episode data by (label, depth_key) for detailed tracking
        self._episode_stats: defaultdict[Tuple[int, str], List[Dict[str, float]]] = defaultdict(list)
        # Track last episode ID to avoid duplicates
        self._last_episode_id: Dict[int, int] = {}
    
    def accumulate(self, infos: List[Dict[str, Any]]):
        """
        Accumulate episode statistics from environment info dicts.
        
        Args:
            infos: List of info dictionaries from environment step.
                   Each info dict should contain:
                   - "episode": dict with "r" (reward) and "l" (length)
                   - "label": int (1 for positive, 0 for negative)
                   - "query_depth": optional int or None
                   - "is_success": optional bool
                   - "episode_idx": optional int for duplicate detection
        """
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue
            
            label = info.get("label")
            if label is None:
                continue
            
            episode_data = info.get("episode")
            if not isinstance(episode_data, dict):
                continue
            
            # Check for duplicate episodes (same episode reported multiple times)
            episode_idx = info.get("episode_idx")
            if episode_idx is not None:
                if self._last_episode_id.get(env_idx) == episode_idx:
                    continue
                self._last_episode_id[env_idx] = episode_idx
            else:
                # Fallback: use object id if episode_idx not available
                episode_id = id(episode_data)
                if self._last_episode_id.get(env_idx) == episode_id:
                    continue
                self._last_episode_id[env_idx] = episode_id
            
            # Extract episode statistics
            reward = episode_data.get("r")
            length = episode_data.get("l")
            depth_value = info.get("query_depth")
            success_flag = bool(info.get("is_success", False))
            
            try:
                label_value = int(label)
            except (TypeError, ValueError):
                continue
            
            # Format depth for consistent naming
            depth_key = _format_depth_key(depth_value)
            
            # Store episode stats
            stats = {
                "reward": float(reward) if reward is not None else None,
                "length": float(length) if length is not None else None,
                "success": 1.0 if success_flag else 0.0,
                "depth_raw": depth_value,
            }
            self._episode_stats[(label_value, depth_key)].append(stats)
    
    def compute_metrics(self) -> Dict[str, str]:
        """
        Compute formatted metrics from accumulated statistics.
        
        Returns:
            Dictionary of metric name -> formatted string "mean +/- std (count)"
            
        Metrics include:
            - Overall: ep_len, ep_rew (if any episodes collected)
            - By label: len_pos, len_neg, proven_pos, proven_neg, rwd_pos, rwd_neg
            - By depth (if detailed): len_d_X_pos/neg, proven_d_X_pos/neg, rwd_d_X_pos/neg
        """
        metrics = {}
        
        if not self._episode_stats:
            return metrics
        
        # ----------------------------------------------------------------
        # 1. Overall episode statistics (aggregate across all labels/depths)
        # ----------------------------------------------------------------
        all_rewards = []
        all_lengths = []
        for episodes in self._episode_stats.values():
            all_rewards.extend([ep["reward"] for ep in episodes if ep.get("reward") is not None])
            all_lengths.extend([ep["length"] for ep in episodes if ep.get("length") is not None])
        
        if all_rewards:
            metrics["ep_rew"] = f"{np.mean(all_rewards):.2f}"
        if all_lengths:
            metrics["ep_len"] = f"{np.mean(all_lengths):.1f}"
        
        # ----------------------------------------------------------------
        # 2. Aggregate by label (positive/negative)
        # ----------------------------------------------------------------
        for label in (1, 0):
            label_str = "pos" if label == 1 else "neg"
            
            rewards = []
            lengths = []
            successes = []
            
            # Aggregate across all depths for this label
            for (lbl, depth_key), episodes in self._episode_stats.items():
                if lbl == label:
                    for ep in episodes:
                        if ep.get("reward") is not None:
                            rewards.append(ep["reward"])
                        if ep.get("length") is not None:
                            lengths.append(ep["length"])
                        successes.append(ep.get("success", 0.0))
            
            # Length by label
            if lengths:
                metrics[f"len_{label_str}"] = _format_stat_string(
                    np.mean(lengths), np.std(lengths), len(lengths)
                )
            
            # Success (proven) by label
            if successes:
                metrics[f"proven_{label_str}"] = _format_stat_string(
                    np.mean(successes), np.std(successes), len(successes)
                )
            
            # Reward by label
            if rewards:
                metrics[f"rwd_{label_str}"] = _format_stat_string(
                    np.mean(rewards), np.std(rewards), len(rewards)
                )
        
        # ----------------------------------------------------------------
        # 3. Detailed breakdown by depth (if enabled)
        # ----------------------------------------------------------------
        if self.collect_detailed:
            # Sort by (label_order, depth_order) for consistent display
            def _order_key(item: Tuple[Tuple[int, str], List[Dict]]) -> Tuple[int, Union[int, float]]:
                (label, depth_key), _ = item
                label_order = 0 if label == 1 else 1 if label == 0 else 2
                if depth_key == "unknown":
                    depth_order: Union[int, float] = float("inf")
                else:
                    try:
                        depth_order = int(depth_key)
                    except (TypeError, ValueError):
                        depth_order = float("inf")
                return (label_order, depth_order)
            
            for (label, depth_key), episodes in sorted(self._episode_stats.items(), key=_order_key):
                label_str = "pos" if label == 1 else "neg" if label == 0 else f"label{label}"
                
                # Length by depth and label
                lengths = [ep["length"] for ep in episodes if ep.get("length") is not None]
                if lengths:
                    metrics[f"len_d_{depth_key}_{label_str}"] = _format_stat_string(
                        np.mean(lengths), np.std(lengths), len(lengths)
                    )
                
                # Success (proven) by depth and label
                successes = [ep["success"] for ep in episodes if "success" in ep]
                if successes:
                    metrics[f"proven_d_{depth_key}_{label_str}"] = _format_stat_string(
                        np.mean(successes), np.std(successes), len(successes)
                    )
                
                # Reward by depth and label
                rewards = [ep["reward"] for ep in episodes if ep.get("reward") is not None]
                if rewards:
                    metrics[f"rwd_d_{depth_key}_{label_str}"] = _format_stat_string(
                        np.mean(rewards), np.std(rewards), len(rewards)
                    )
        
        return metrics
