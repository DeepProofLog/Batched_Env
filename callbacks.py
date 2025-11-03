"""
TorchRL Callbacks for Neural-guided Grounding.

Provides comprehensive training callbacks similar to Stable-Baselines3:
- Rollout progress tracking
- Evaluation metrics with depth/label breakdowns
- Training metrics formatting and logging
- Timing information
- Detailed metrics collection module for both training and evaluation
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

def _format_depth_key(depth_value: Any) -> str:
    """
    Normalize depth IDs so metrics share consistent naming.
    
    Args:
        depth_value: Raw depth value (int, None, -1, etc.)
        
    Returns:
        Normalized depth string ("0", "1", "2", ..., "unknown", or "none")
    """
    if depth_value == -1:
        return "unknown"
    if depth_value is None:
        return "none"
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
                try:
                    num_val = float(value)
                    if num_val.is_integer():
                        value_str = str(int(num_val))
                    else:
                        value_str = f"{num_val:.3f}"
                except (ValueError, TypeError):
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
    
    def __init__(self, collect_detailed: bool = True, verbose: bool = False):
        """
        Args:
            collect_detailed: If True, collect detailed breakdown by depth.
                            If False, only collect aggregate stats by label.
            verbose: If True, print debug information during collection.
        """
        self.collect_detailed = collect_detailed
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        """Clear all accumulated statistics."""
        # Store raw episode data by (label, depth_key) for detailed tracking
        self._episode_stats: defaultdict[Tuple[int, str], List[Dict[str, float]]] = defaultdict(list)
        # Track last episode ID to avoid duplicates
        self._last_episode_id: Dict[int, int] = {}
        if self.verbose:
            print("[DetailedMetricsCollector] Stats reset")
    
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
        if self.verbose:
            print(f"[DetailedMetricsCollector] Accumulating stats from {len(infos)} infos")
        
        episodes_added = 0
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                if self.verbose and info:
                    print(f"[DetailedMetricsCollector] Info {env_idx}: No episode data")
                continue
            
            label = info.get("label")
            if label is None:
                if self.verbose:
                    print(f"[DetailedMetricsCollector] Info {env_idx}: No label")
                continue
            
            episode_data = info.get("episode")
            if not isinstance(episode_data, dict):
                if self.verbose:
                    print(f"[DetailedMetricsCollector] Info {env_idx}: Episode data not a dict")
                continue
            
            # Check for duplicate episodes (same episode reported multiple times)
            episode_idx = info.get("episode_idx")
            if episode_idx is not None:
                if self._last_episode_id.get(env_idx) == episode_idx:
                    if self.verbose:
                        print(f"[DetailedMetricsCollector] Info {env_idx}: Duplicate episode_idx {episode_idx}")
                    continue
                self._last_episode_id[env_idx] = episode_idx
            else:
                # Fallback: use object id if episode_idx not available
                episode_id = id(episode_data)
                if self._last_episode_id.get(env_idx) == episode_id:
                    if self.verbose:
                        print(f"[DetailedMetricsCollector] Info {env_idx}: Duplicate episode_id {episode_id}")
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
                if self.verbose:
                    print(f"[DetailedMetricsCollector] Info {env_idx}: Invalid label {label}")
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
            episodes_added += 1
            
            if self.verbose:
                print(f"[DetailedMetricsCollector] Info {env_idx}: Added episode - "
                      f"label={label_value}, depth={depth_key}, reward={reward}, length={length}, success={success_flag}")
        
        if self.verbose:
            print(f"[DetailedMetricsCollector] Accumulated {episodes_added} new episodes, "
                  f"total {sum(len(eps) for eps in self._episode_stats.values())} episodes")
    
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
        if self.verbose:
            total_episodes = sum(len(eps) for eps in self._episode_stats.values())
            print(f"[DetailedMetricsCollector] Computing metrics from {total_episodes} episodes")
        
        metrics = {}
        
        if not self._episode_stats:
            if self.verbose:
                print("[DetailedMetricsCollector] No episode stats to compute")
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



# ============================================================================
# Evaluation Callback
# ============================================================================


class EvaluationCallback:
    """
    Callback for comprehensive evaluation with depth and label breakdowns.
    
    Provides detailed metrics similar to SB3's CustomEvalCallbackMRR.
    Uses DetailedMetricsCollector for optional detailed breakdown by depth.
    """
    
    def __init__(
        self,
        eval_env,
        sampler,
        eval_data: List[Any],
        eval_data_depths: Optional[List[int]] = None,
        n_corruptions: int = 50,
        eval_freq: int = 1,  # Evaluate every N iterations
        best_metric: str = "mrr_mean",
        save_path: Optional[Path] = None,
        verbose: bool = True,
        collect_detailed: bool = True,  # NEW: Enable detailed depth breakdown
        verbose_cb: bool = False,  # NEW: Verbose callback debugging
    ):
        """
        Args:
            eval_env: Environment for evaluation
            sampler: Negative sampler for corruption-based evaluation
            eval_data: List of evaluation queries
            eval_data_depths: Optional depth information for queries
            n_corruptions: Number of corruptions per query
            eval_freq: Evaluate every N iterations
            best_metric: Metric to track for best model (e.g., 'mrr_mean', 'auc_pr')
            save_path: Path to save best model
            verbose: Whether to print detailed evaluation info
            collect_detailed: If True, collect detailed breakdown by depth.
                            If False, only collect aggregate stats.
            verbose_cb: If True, print debug information during callback collection.
        """
        self.eval_env = eval_env
        self.sampler = sampler
        self.eval_data = eval_data
        self.eval_data_depths = eval_data_depths
        self.n_corruptions = n_corruptions
        self.eval_freq = eval_freq
        self.best_metric = best_metric
        self.save_path = Path(save_path) if save_path else None
        self.verbose = verbose
        self.verbose_cb = verbose_cb
        
        # Use the common detailed metrics collector
        self.metrics_collector = DetailedMetricsCollector(
            collect_detailed=collect_detailed,
            verbose=verbose_cb
        )
        
        # Tracking
        self.best_metric_value = float('-inf')
        self.current_iteration = 0
    
    def should_evaluate(self, iteration: int) -> bool:
        """Check if we should run evaluation at this iteration."""
        return (iteration % self.eval_freq == 0) or (iteration == 0)
    
    def on_evaluation_start(self, iteration: int, global_step: int):
        """Called at the start of evaluation."""
        self.current_iteration = iteration
        if self.verbose_cb:
            print(f"[EvaluationCallback] Starting evaluation at iteration {iteration}, step {global_step}")
        self.metrics_collector.reset()
        self.eval_start_time = time.time()
    
    def accumulate_episode_stats(self, infos: List[Dict[str, Any]]):
        """Accumulate episode stats by label and depth using the common collector."""
        if self.verbose_cb:
            print(f"[EvaluationCallback] Accumulating episode stats from {len(infos)} infos")
        self.metrics_collector.accumulate(infos)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics from collected stats."""
        if self.verbose_cb:
            print("[EvaluationCallback] Computing metrics")
        return self.metrics_collector.compute_metrics()
    
    def on_evaluation_end(self, iteration: int, global_step: int, eval_metrics: Dict[str, Any]):
        """Called when evaluation is complete."""        
        # Get global statistics from DetailedMetricsCollector
        # These include: ep_len, ep_rew, len_pos, len_neg, proven_pos, proven_neg, rwd_pos, rwd_neg
        rollout_metrics = self.compute_metrics()
        
        # Convert episode_len_* metrics from eval_metrics (the breakdown stats)
        # from mean/std pairs to formatted strings
        self._convert_episode_len_metrics(eval_metrics)
        
        # Merge: rollout_metrics take priority for display, then eval_metrics
        # This ensures we show both global stats AND the detailed breakdown
        merged_metrics = {}
        merged_metrics.update(eval_metrics)  # Start with eval metrics
        merged_metrics.update(rollout_metrics)  # Override with global stats
        
        # Check if this is a new best
        metric_value = merged_metrics.get(self.best_metric, float('-inf'))
        if isinstance(metric_value, str):
            # Try to extract numerical value if it's a formatted string
            try:
                metric_value = float(metric_value.split()[0])
            except:
                metric_value = float('-inf')
        
        is_new_best = metric_value > self.best_metric_value
        if is_new_best:
            self.best_metric_value = metric_value
            metric_display = self.best_metric.replace('_', ' ').upper()
            if self.verbose:
                print(f"New best {metric_display} in eval: {metric_value:.4f}!")
        
        # Print formatted metrics using the common function
        if self.verbose:
            # Only include priority evaluation metrics (MRR, hits, AUC, etc.)
            # Exclude the detailed breakdown metrics that are already captured in rollout_metrics
            priority_eval_keys = [
                "mrr_mean", "hits1_mean", "hits3_mean", "hits10_mean", 
                "average_precision", "auc_pr", "auc_pr_mean", "success_rate",
                "head_mrr_mean", "tail_mrr_mean"
            ]
            extra_metrics = {k: v for k, v in merged_metrics.items() 
                           if k not in rollout_metrics and k in priority_eval_keys}
            
            # Use the common print function
            print_formatted_metrics(
                metrics=rollout_metrics,
                prefix="eval",
                extra_metrics=extra_metrics,
                global_step=global_step,
            )
        return is_new_best
    
    def _convert_episode_len_metrics(self, metrics: Dict[str, Any]):
        """Convert episode_len_* mean/std pairs to formatted strings."""
        # Convert episode_len metrics to formatted strings
        # We don't always have count information from _episode_stats (e.g., when info_callback
        # doesn't populate it), so we'll show metrics without counts
        
        metric_prefixes = [
            'episode_len_pos',
            'episode_len_neg',
            'episode_len_pos_true',
            'episode_len_pos_false',
            'episode_len_neg_true',
            'episode_len_neg_false',
        ]
        
        for metric_base in metric_prefixes:
            mean_key = f'{metric_base}_mean'
            std_key = f'{metric_base}_std'
            
            if mean_key in metrics and std_key in metrics:
                mean_val = metrics[mean_key]
                std_val = metrics[std_key]
                # Format as "mean +/- std" without count
                formatted = f"{mean_val:.3f} +/- {std_val:.2f}"
                metrics[f'ep_len_{metric_base.replace("episode_len_", "")}'] = formatted
                # Remove the old _mean and _std keys
                del metrics[mean_key]
                del metrics[std_key]


# ============================================================================
# Training Metrics Callback
# ============================================================================


class TrainingMetricsCallback:
    """
    Callback to format and display training metrics.
    
    Provides formatted output similar to SB3's logger.dump().
    Uses DetailedMetricsCollector for optional detailed breakdown by depth.
    """
    
    def __init__(
        self,
        log_interval: int = 1,
        verbose: bool = True,
        collect_detailed: bool = True,
        verbose_cb: bool = False,  # NEW: Verbose callback debugging
    ):
        """
        Args:
            log_interval: Print metrics every N iterations
            verbose: Whether to print metrics
            collect_detailed: If True, collect detailed breakdown by depth.
                            If False, only collect aggregate stats.
            verbose_cb: If True, print debug information during callback collection.
        """
        self.log_interval = log_interval
        self.verbose = verbose
        self.verbose_cb = verbose_cb
        
        # Use the common detailed metrics collector
        self.metrics_collector = DetailedMetricsCollector(
            collect_detailed=collect_detailed,
            verbose=verbose_cb
        )
        
        self.train_start_time = None
        self.training_epoch_losses = []  # Track losses per epoch
    
    def on_training_start(self):
        """Called at the start of training."""
        if self.verbose_cb:
            print("[TrainingMetricsCallback] Training started")
        self.train_start_time = time.time()
    
    def on_training_epoch(self, epoch: int, n_epochs: int, policy_loss: float, value_loss: float, entropy: float):
        """
        Called during each training epoch.
        
        Args:
            epoch: Current epoch number (1-indexed)
            n_epochs: Total number of epochs
            policy_loss: Policy gradient loss for this epoch
            value_loss: Value function loss for this epoch
            entropy: Entropy for this epoch
        """
        self.training_epoch_losses.append({
            'epoch': epoch,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
        })
        
        if self.verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs):
            print(f"  Training epoch {epoch}/{n_epochs}: "
                  f"pg_loss={policy_loss:.4f}, v_loss={value_loss:.4f}, entropy={entropy:.4f}")
    
    def accumulate_episode_stats(self, infos: List[Dict[str, Any]]):
        """Accumulate episode stats during training using the common collector."""
        if self.verbose_cb:
            print(f"[TrainingMetricsCallback] Accumulating episode stats from {len(infos)} infos")
        self.metrics_collector.accumulate(infos)
    
    def on_iteration_end(
        self,
        iteration: int,
        global_step: int,
        train_metrics: Dict[str, float],
        n_envs: int = 1,
    ):
        """Called at the end of a training iteration."""
        if self.verbose_cb:
            print(f"[TrainingMetricsCallback] Iteration {iteration} ended at step {global_step}")
        
        if not self.verbose or iteration % self.log_interval != 0:
            self.metrics_collector.reset()
            return
        
        if self.verbose_cb:
            print("[TrainingMetricsCallback] Computing and printing metrics")
        
        # Compute rollout metrics using the common collector
        rollout_metrics = self.metrics_collector.compute_metrics()
        
        # Compute timing metrics
        time_metrics = self._compute_time_metrics(iteration, global_step, n_envs)
        
        # Combine time and train metrics for extra_metrics
        extra_metrics = {**time_metrics, **train_metrics}
        
        # Use the common print function
        print_formatted_metrics(
            metrics=rollout_metrics,
            prefix="rollout",
            extra_metrics=extra_metrics,
            global_step=None,  # Already in extra_metrics as total_timesteps
        )
        
        # Reset stats for next iteration
        self.metrics_collector.reset()
    
    def _compute_time_metrics(self, iteration: int, global_step: int, n_envs: int) -> Dict[str, str]:
        """Compute timing metrics."""
        metrics = {}
        
        if self.train_start_time is not None:
            elapsed = time.time() - self.train_start_time
            if elapsed > 0:
                fps = int(global_step / elapsed)
                metrics["fps"] = str(fps)
        
        metrics["iterations"] = str(iteration)
        metrics["total_timesteps"] = str(global_step)
        
        return metrics


# ============================================================================
# Rollout Progress Callback
# ============================================================================


class RolloutProgressCallback:
    """
    Callback to track and display rollout progress during training.
    
    Prints progress updates during rollout collection, similar to SB3's output.
    """
    
    def __init__(self, total_steps: int, n_envs: int = 1, update_interval: int = 25, verbose: bool = True):
        """
        Args:
            total_steps: Total number of steps to collect per rollout (n_steps * n_envs)
            n_envs: Number of parallel environments
            update_interval: Print progress every N total steps
            verbose: Whether to print progress
        """
        self.total_steps = total_steps
        self.n_envs = n_envs
        self.update_interval = update_interval
        self.verbose = verbose
        self.current_steps = 0
        self.start_time = None
    
    def on_rollout_start(self):
        """Called at the start of rollout collection."""
        self.current_steps = 0
        # self.start_time = time.time()
        if self.verbose:
            print("Collecting rollouts")
    
    def on_step(self, step: int):
        """
        Called after each step during rollout.
        
        Args:
            step: Current step index (per environment, 0 to n_steps-1)
        """
        # Calculate total steps collected so far
        total_collected = (step + 1) * self.n_envs
        
        if self.verbose and total_collected % self.update_interval == 0:
            print(f"Collecting rollouts: {total_collected}/{self.total_steps} steps")
        
        self.current_steps = total_collected
    
    def on_rollout_end(self):
        """Called when rollout collection is complete."""
        pass
        # if self.start_time is not None:
        #     elapsed = time.time() - self.start_time
        #     if self.verbose:
        #         print(f"Time to collect_rollouts {elapsed:.2f}")


# ============================================================================
# Callback Manager
# ============================================================================


class TorchRLCallbackManager:
    """
    Main callback manager that coordinates all callbacks.
    
    This class integrates all callbacks and provides a unified interface
    for the training loop.
    """
    
    def __init__(
        self,
        rollout_callback: Optional[RolloutProgressCallback] = None,
        eval_callback: Optional[EvaluationCallback] = None,
        train_callback: Optional[TrainingMetricsCallback] = None,
    ):
        """
        Args:
            rollout_callback: Callback for rollout progress
            eval_callback: Callback for evaluation
            train_callback: Callback for training metrics
        """
        self.rollout_callback = rollout_callback
        self.eval_callback = eval_callback
        self.train_callback = train_callback  # Expose publicly for direct access
    
    def on_training_start(self):
        """Called at the start of training."""
        if self.train_callback:
            self.train_callback.on_training_start()
    
    def on_rollout_start(self):
        """Called at the start of rollout collection."""
        if self.rollout_callback:
            self.rollout_callback.on_rollout_start()
    
    def on_rollout_step(self, step: int):
        """Called after each step during rollout."""
        if self.rollout_callback:
            self.rollout_callback.on_step(step)
    
    def on_rollout_end(self):
        """Called when rollout collection is complete."""
        if self.rollout_callback:
            self.rollout_callback.on_rollout_end()
    
    def accumulate_episode_stats(self, infos: List[Dict[str, Any]], mode: str = "train"):
        """
        Accumulate episode statistics.
        
        Args:
            infos: List of info dicts from environment
            mode: 'train' or 'eval'
        """
        if mode == "eval" and self.eval_callback:
            if hasattr(self.eval_callback, 'verbose_cb') and self.eval_callback.verbose_cb:
                print(f"[CallbackManager] Forwarding {len(infos)} infos to eval callback")
            self.eval_callback.accumulate_episode_stats(infos)
        elif mode == "train" and self.train_callback:
            if hasattr(self.train_callback, 'verbose_cb') and self.train_callback.verbose_cb:
                print(f"[CallbackManager] Forwarding {len(infos)} infos to train callback")
            self.train_callback.accumulate_episode_stats(infos)
    
    def should_evaluate(self, iteration: int) -> bool:
        """Check if evaluation should be run."""
        if self.eval_callback:
            return self.eval_callback.should_evaluate(iteration)
        return False
    
    def on_evaluation_start(self, iteration: int, global_step: int):
        """Called at the start of evaluation."""
        if self.eval_callback:
            self.eval_callback.on_evaluation_start(iteration, global_step)
    
    def on_evaluation_end(
        self,
        iteration: int,
        global_step: int,
        eval_metrics: Dict[str, Any],
    ) -> bool:
        """
        Called when evaluation is complete.
        
        Returns:
            True if this is a new best model
        """
        if self.eval_callback:
            # Compute depth-based metrics from accumulated episode stats
            depth_metrics = self.eval_callback.compute_metrics()
            # Merge depth metrics into eval_metrics
            merged_metrics = {**eval_metrics, **depth_metrics}
            return self.eval_callback.on_evaluation_end(iteration, global_step, merged_metrics)
        return False
    
    def on_iteration_end(
        self,
        iteration: int,
        global_step: int,
        train_metrics: Dict[str, float],
        n_envs: int = 1,
    ):
        """Called at the end of a training iteration."""
        if self.train_callback:
            self.train_callback.on_iteration_end(iteration, global_step, train_metrics, n_envs)
