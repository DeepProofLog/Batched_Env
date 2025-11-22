"""
TorchRL Callbacks for Neural-guided Grounding.

Provides comprehensive training callbacks similar to Stable-Baselines3:
- Rollout progress tracking
- Evaluation metrics with depth/label breakdowns
- Training metrics formatting and logging
- Timing information
- Detailed metrics collection module for both training and evaluation
- MRR evaluation callbacks with corruption-based metrics
- Model checkpoint management
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
        return "unknown"
    return str(depth_value)

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
    
    # Collect all metrics into a dictionary for sorting
    final_output = {}
    
    # Add extra metrics if present
    if extra_metrics:
        # Other metrics (losses, entropy, etc. for training; hits, episode breakdown for evaluation)
        # Only show under "train/" prefix if we're actually in training context (prefix != "eval")
        time_keys = ["fps", "iterations", "total_timesteps"]
        other_keys = sorted([k for k in extra_metrics.keys() if k not in time_keys])
        
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
                raise ValueError("DetailedMetricsCollector: 'label' is missing from info; cannot compute metrics")
            
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
            
            try:
                label_value = int(label)
            except (TypeError, ValueError):
                if self.verbose:
                    print(f"[DetailedMetricsCollector] Info {env_idx}: Invalid label {label}")
                continue

            # Extract episode statistics
            reward = episode_data.get("r")
            length = episode_data.get("l")
            depth_value = info.get("query_depth")
            # For negatives, bucket into "none" like SB3 (no depth for neg samples)
            if label_value == 0:
                depth_value = -1
            success_flag = bool(info.get("is_success", False))
            
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
        all_successes = []
        for episodes in self._episode_stats.values():
            all_rewards.extend([ep["reward"] for ep in episodes if ep.get("reward") is not None])
            all_lengths.extend([ep["length"] for ep in episodes if ep.get("length") is not None])
            all_successes.extend([ep.get("success", 0.0) for ep in episodes])
        
        # Overall statistics with std and means (align to SB3-style rollout)
        if all_rewards:
            metrics["ep_rew_mean"] = float(np.mean(all_rewards))
        if all_rewards:
            metrics["reward_overall"] = _format_stat_string(
                np.mean(all_rewards), np.std(all_rewards), len(all_rewards)
            )
        if all_lengths:
            metrics["ep_len_mean"] = float(np.mean(all_lengths))
            metrics["length mean +/- std"] = _format_stat_string(
                np.mean(all_lengths), np.std(all_lengths), len(all_lengths)
            )
        if all_successes:
            metrics["success_rate"] = f"{np.mean(all_successes):.3f}"
        
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
            
            # Reward by label (with both naming conventions)
            if rewards:
                metrics[f"rwd_{label_str}"] = _format_stat_string(
                    np.mean(rewards), np.std(rewards), len(rewards)
                )
                metrics[f"reward_label_{label_str}"] = _format_stat_string(
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
                
                # Reward by depth and label (with both naming conventions)
                rewards = [ep["reward"] for ep in episodes if ep.get("reward") is not None]
                if rewards:
                    # metrics[f"rwd_d_{depth_key}_{label_str}"] = _format_stat_string(
                    #     np.mean(rewards), np.std(rewards), len(rewards)
                    # )
                    metrics[f"reward_d_{depth_key}_{label_str}"] = _format_stat_string(
                        np.mean(rewards), np.std(rewards), len(rewards)
                    )
        
        return metrics



# ============================================================================
# Evaluation Callback
# ============================================================================


class EvaluationCallback:
    """
    Callback for comprehensive evaluation with depth and label breakdowns.
    
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
        rollout_metrics = self.compute_metrics()
        
        # Merge metrics for best model tracking
        merged_metrics = {**rollout_metrics, **eval_metrics}
        
        # Check if this is a new best
        metric_value = merged_metrics.get(self.best_metric, float('-inf'))
        if isinstance(metric_value, str):
            try:
                metric_value = float(metric_value.split()[0])
            except (ValueError, TypeError):
                metric_value = float('-inf')
        
        is_new_best = metric_value > self.best_metric_value
        if is_new_best:
            self.best_metric_value = metric_value
            metric_display = self.best_metric.replace('_', ' ').upper()
            if self.verbose:
                print(f"New best {metric_display} in eval: {metric_value:.4f}!")
        
        # Print formatted metrics using the common function
        if self.verbose:
            print_formatted_metrics(
                metrics=rollout_metrics,
                prefix="eval",
                extra_metrics=eval_metrics,
                global_step=global_step,
            )
        return is_new_best
    

    



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
    
    # def on_training_epoch(self, epoch: int, n_epochs: int, 
    #                       policy_loss: float, 
    #                       value_loss: float, 
    #                       entropy: float,
    #                       approx_kl: float,
    #                       clip_frac: float,
    #                       grad_norm: Optional[float] = None):
    #     """
    #     Called during each training epoch.
        
    #     Args:
    #         epoch: Current epoch number (1-indexed)
    #         n_epochs: Total number of epochs
    #         policy_loss: Policy gradient loss for this epoch
    #         value_loss: Value function loss for this epoch
    #         entropy: Entropy for this epoch
    #     """
    #     self.training_epoch_losses.append({
    #         'epoch': epoch,
    #         'policy_loss': policy_loss,
    #         'value_loss': value_loss,
    #         'entropy': entropy,
    #     })
    #     shown_in_learn_loop = True
    #     if not shown_in_learn_loop:
    #         if self.verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs):
    #             if grad_norm is not None:
    #                 print(f"  Training epoch {epoch}/{n_epochs}: "
    #                     f"pg_loss={policy_loss:.4f}, v_loss={value_loss:.4f}, entropy={entropy:.4f}, kl={approx_kl:.4f}, clip_frac={clip_frac:.4f}, grad_norm={grad_norm:.4f}")
    #             else:
    #                 print(f"  Training epoch {epoch}/{n_epochs}: "
    #                     f"pg_loss={policy_loss:.4f}, v_loss={value_loss:.4f}, entropy={entropy:.4f}, kl={approx_kl:.4f}, clip_frac={clip_frac:.4f}")

    def accumulate_episode_stats(self, infos: List[Dict[str, Any]]):
        """Accumulate episode stats during training using the common collector."""
        if self.verbose_cb:
            print(f"[TrainingMetricsCallback] Accumulating episode stats from {len(infos)} infos")
        self.metrics_collector.accumulate(infos)
    
    def on_iteration_end(
        self,
        iteration: int,
        global_step: int,
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
        
        # Use only timing metrics for additional context
        extra_metrics = time_metrics
        
        # Use the common print function
        print_formatted_metrics(
            metrics=rollout_metrics,
            prefix="rollout",
            extra_metrics=extra_metrics,
            global_step=None,  # Already in extra_metrics as total_timesteps
        )
        
        # Reset metrics after printing to avoid accumulation across iterations
        self.metrics_collector.reset()
        
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
        eval_callback: Optional[Union[EvaluationCallback, "MRREvaluationCallback"]] = None,
        train_callback: Optional[TrainingMetricsCallback] = None,
        checkpoint_callback: Optional["TrainingCheckpointCallback"] = None,
    ):
        """
        Args:
            rollout_callback: Callback for rollout progress
            eval_callback: Callback for evaluation (standard or MRR-based)
            train_callback: Callback for training metrics
            checkpoint_callback: Callback for saving model checkpoints
        """
        self.rollout_callback = rollout_callback
        self.eval_callback = eval_callback
        self.train_callback = train_callback  # Expose publicly for direct access
        self.checkpoint_callback = checkpoint_callback
    
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
    
    def accumulate_episode_stats(self, infos: Union[List[Dict[str, Any]], Dict[str, torch.Tensor]], mode: str = "train"):
        """
        Accumulate episode statistics.
        
        Args:
            infos: List of info dicts from environment
            mode: 'train' or 'eval'
        """
        def _tensor_batch_to_list(batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
            # Handle both old keys (r, l, s) and new keys (reward, length, is_success)
            reward_tensor = batch.get("r") if "r" in batch else batch.get("reward")
            length_tensor = batch.get("l") if "l" in batch else batch.get("length")
            success_tensor = batch.get("s") if "s" in batch else batch.get("is_success")
            
            if reward_tensor is None or length_tensor is None:
                return []

            count = reward_tensor.shape[0]
            if count == 0:
                return []

            def _to_list(t: Optional[torch.Tensor]) -> Optional[List[Any]]:
                if t is None:
                    return None
                return t.detach().cpu().tolist()

            rewards = _to_list(reward_tensor)
            lengths = _to_list(length_tensor)
            labels = _to_list(batch.get("label"))
            depths = _to_list(batch.get("query_depth"))
            successes = _to_list(success_tensor)

            result: List[Dict[str, Any]] = []
            for idx in range(count):
                entry = {
                    "episode": {
                        "r": float(rewards[idx]),
                        "l": int(lengths[idx]),
                    }
                }
                if labels is not None:
                    entry["label"] = int(labels[idx])
                if depths is not None:
                    entry["query_depth"] = int(depths[idx])
                if successes is not None:
                    entry["is_success"] = bool(successes[idx])
                result.append(entry)
            return result

        processed_infos: List[Dict[str, Any]]
        if isinstance(infos, dict):
            processed_infos = _tensor_batch_to_list(infos)
        else:
            processed_infos = infos
        if not processed_infos:
            return

        if mode == "train" and self.train_callback:
            if hasattr(self.train_callback, "verbose_cb") and self.train_callback.verbose_cb:
                print(f"[CallbackManager] Forwarding {len(processed_infos)} infos to train callback")
            self.train_callback.accumulate_episode_stats(processed_infos)
        elif mode == "eval" and self.eval_callback:
            if hasattr(self.eval_callback, 'verbose_cb') and self.eval_callback.verbose_cb:
                print(f"[CallbackManager] Forwarding {len(processed_infos)} infos to eval callback")
            self.eval_callback.accumulate_episode_stats(processed_infos)
    
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
            return self.eval_callback.on_evaluation_end(iteration, global_step, eval_metrics)
        return False
    
    def on_iteration_end(
        self,
        iteration: int,
        global_step: int,
        n_envs: int = 1,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Called at the end of a training iteration."""
        if self.train_callback:
            self.train_callback.on_iteration_end(iteration, global_step, n_envs)
        
        # Handle checkpoint callback
        if self.checkpoint_callback and metrics:
            self.checkpoint_callback.on_iteration_end(iteration, global_step, metrics)


# ============================================================================
# MRR Evaluation Callback (adapted from SB3)
# ============================================================================


class MRREvaluationCallback(EvaluationCallback):
    """
    Enhanced evaluation callback with MRR metrics using corruption-based evaluation.
    
    This callback extends the standard EvaluationCallback by computing MRR, Hits@K
    metrics through evaluate_ranking_metrics. It's adapted for TorchRL environments.
    
    Key features:
    - Computes MRR, Hits@1/3/10 metrics via evaluate_ranking_metrics
    - Tracks best model based on configurable metric (MRR or Hits@K)
    - Integrates with DetailedMetricsCollector for depth breakdowns
    - Compatible with TorchRL batched environments
    
    NOTE: This callback requires the policy to be passed during evaluation.
          It will call evaluate_ranking_metrics from model_eval.py.
    """
    
    def __init__(
        self,
        eval_env,
        sampler,
        eval_data: torch.Tensor,  # Should be [N, 3] tensor of triples for MRR eval
        eval_data_depths: Optional[List[int]] = None,
        n_corruptions: Optional[int] = 10,
        eval_freq: int = 1,
        best_metric: str = "mrr_mean",
        save_path: Optional[Path] = None,
        model_name: str = "model",
        verbose: bool = True,
        collect_detailed: bool = True,
        verbose_cb: bool = False,
        corruption_scheme: Optional[List[str]] = None,
    ):
        """
        Args:
            eval_env: BatchedEnv for evaluation
            sampler: Negative sampler for corruption-based evaluation
            eval_data: Tensor of evaluation query triples [N, 3] or [N, 1, 3]
            eval_data_depths: Optional depth information for queries
            n_corruptions: Number of corruptions per query (None = all)
            eval_freq: Evaluate every N iterations
            best_metric: Metric to track for best model ('mrr_mean' or 'hits1_mean', 'hits3_mean', 'hits10_mean')
            save_path: Path to save best model
            model_name: Name prefix for saved models
            verbose: Whether to print detailed evaluation info
            collect_detailed: If True, collect detailed breakdown by depth
            verbose_cb: If True, print debug information during callback collection
            corruption_scheme: List of corruption modes (e.g., ['head', 'tail'])
        """
        # Convert eval_data to list format if needed for parent class compatibility
        if isinstance(eval_data, torch.Tensor):
            # Parent class expects a list but we'll store the tensor separately
            eval_data_list = [None] * eval_data.shape[0]  # Dummy list for parent
        else:
            eval_data_list = eval_data
        
        super().__init__(
            eval_env=eval_env,
            sampler=sampler,
            eval_data=eval_data_list,
            eval_data_depths=eval_data_depths,
            n_corruptions=0,  # Parent doesn't use this for MRR
            eval_freq=eval_freq,
            best_metric=best_metric,
            save_path=save_path,
            verbose=verbose,
            collect_detailed=collect_detailed,
            verbose_cb=verbose_cb,
        )
        
        # Store the actual tensor for MRR evaluation
        self.eval_queries_tensor = eval_data if isinstance(eval_data, torch.Tensor) else None
        self.n_corruptions = n_corruptions
        self.model_name = model_name
        self.corruption_scheme = corruption_scheme or ["head", "tail"]
        self.best_epoch_step = None
        
        # Metric configuration
        metric_options = {
            "mrr_mean": {"display_name": "MRR", "log_key": "mrr_mean"},
            "hits1_mean": {"display_name": "Hits@1", "log_key": "hits1_mean"},
            "hits3_mean": {"display_name": "Hits@3", "log_key": "hits3_mean"},
            "hits10_mean": {"display_name": "Hits@10", "log_key": "hits10_mean"},
        }
        
        metric_key = best_metric.lower()
        if metric_key not in metric_options:
            allowed = ", ".join(sorted(metric_options.keys()))
            raise ValueError(f"Unsupported best_metric '{best_metric}'. Allowed: {allowed}.")
        
        self._metric_options = metric_options
        self.best_metric = metric_key
        self._metric_display_name = metric_options[metric_key]["display_name"]
        self._metric_log_key = metric_options[metric_key]["log_key"]
        
        # For collecting eval episode info
        self._eval_episode_buffer = []
    
    def _collect_eval_episode_info(self, step_td):
        """Callback to collect episode info during evaluation."""
        
        def _get_val(td, key, default=None):
            if key in td.keys():
                return td.get(key)
            if "next" in td.keys():
                nxt = td.get("next")
                if key in nxt.keys():
                    return nxt.get(key)
            return default

        # Extract episode info if done
        done = _get_val(step_td, "done")
        
        if done is not None and done.any():
            # print(f"DEBUG: done detected! reward={_get_val(step_td, 'reward')}")
            reward = _get_val(step_td, "reward")
            if reward is not None:
                # Get additional info
                label = _get_val(step_td, "label")
                query_depth = _get_val(step_td, "query_depth")
                is_success = _get_val(step_td, "is_success")
                length = _get_val(step_td, "length")
                
                # Store for each done environment
                batch_size = done.shape[0]
                for i in range(batch_size):
                    if done[i]:
                        info = {
                            "episode": {
                                "r": float(reward[i].item()) if reward is not None else 0.0,
                                "l": int(length[i].item()) if length is not None else 1,
                            }
                        }
                        if label is not None:
                            info["label"] = int(label[i].item())
                        if query_depth is not None:
                            info["query_depth"] = int(query_depth[i].item())
                        if is_success is not None:
                            info["is_success"] = bool(is_success[i].item())
                        self._eval_episode_buffer.append(info)
    
    def on_evaluation_start(self, iteration: int, global_step: int):
        """Called at start of evaluation - reset eval episode buffer."""
        if self.verbose:
            print(f"[MRREvaluationCallback] Starting evaluation at iter {iteration}, step {global_step}. Resetting collector.")
        self._eval_episode_buffer = []
        # Reset metrics collector for eval mode
        self.metrics_collector.reset()
    
    def on_evaluation_end(self, iteration: int, global_step: int, eval_metrics: Dict[str, Any]) -> bool:
        """Called when MRR evaluation is complete."""
        # Process any collected eval episode info
        if self._eval_episode_buffer:
            self.accumulate_episode_stats(self._eval_episode_buffer)
        
        # Get rollout metrics from DetailedMetricsCollector
        rollout_metrics = self.compute_metrics()
        
        # Merge metrics for best model tracking
        merged_metrics = {**rollout_metrics, **eval_metrics}
        
        # Extract metric value
        metric_value = merged_metrics.get(self._metric_log_key, float('-inf'))
        if isinstance(metric_value, str):
            try:
                metric_value = float(metric_value.split()[0])
            except (ValueError, TypeError):
                metric_value = float('-inf')
        
        # Check if this is a new best
        is_new_best = metric_value > self.best_metric_value
        if is_new_best:
            self.best_metric_value = metric_value
            self.best_epoch_step = global_step
            if self.verbose:
                print(f"✓ New best {self._metric_display_name}: {metric_value:.4f} at step {global_step}")
            
            # Save model if path is provided
            if self.save_path is not None:
                self._save_best_model(iteration, global_step, metric_value)
        
        # Print formatted metrics
        if self.verbose:
            print_formatted_metrics(
                metrics=rollout_metrics,
                prefix="eval",
                extra_metrics=eval_metrics,
                global_step=global_step,
            )
        
        return is_new_best
    
    def evaluate_mrr(self, policy: nn.Module) -> Dict[str, Any]:
        """
        Perform MRR evaluation using evaluate_ranking_metrics.
        
        Args:
            policy: The full ActorCriticPolicy to evaluate
        
        Returns:
            Dictionary of MRR metrics
        """
        if self.eval_queries_tensor is None or self.eval_queries_tensor.shape[0] == 0:
            return {}
        
        # Import here to avoid circular imports
        from model_eval import evaluate_ranking_metrics, evaluate_policy
        
        try:
            # Use the policy's _predict_actions method which handles embeddings correctly
            # Create a wrapper that matches the expected interface
            class PolicyWrapper:
                def __init__(self, policy):
                    self.policy = policy
                    self.training = policy.training
                
                def eval(self):
                    self.policy.eval()
                    return self
                
                def train(self):
                    self.policy.train()
                    return self
                
                def __call__(self, obs_td, deterministic=True):
                    # obs_td is a TensorDict from evaluate_policy
                    # Call policy's forward which returns (actions, values, log_probs)
                    with torch.no_grad():
                        actions, _, log_probs = self.policy(obs_td, deterministic=deterministic)
                    # Return TensorDict with action key as evaluate_policy expects
                    from tensordict import TensorDict
                    return TensorDict({
                        "action": actions,
                        "sample_log_prob": log_probs
                    }, batch_size=obs_td.batch_size)
            
            wrapped_policy = PolicyWrapper(policy)
            
            # Convert depths to tensor if available
            query_depths_tensor = None
            if self.eval_data_depths is not None:
                # Handle None values in list (replace with -1)
                depths_clean = [d if d is not None else -1 for d in self.eval_data_depths]
                # Ensure device matches queries
                device = self.eval_queries_tensor.device if self.eval_queries_tensor is not None else torch.device('cpu')
                query_depths_tensor = torch.tensor(depths_clean, dtype=torch.long, device=device)

            # Do MRR evaluation (internally calls evaluate_policy which collects episode stats)
            metrics = evaluate_ranking_metrics(
                actor=wrapped_policy,
                env=self.eval_env,
                queries=self.eval_queries_tensor,
                sampler=self.sampler,
                query_depths=query_depths_tensor,
                n_corruptions=self.n_corruptions,
                corruption_modes=self.corruption_scheme,
                deterministic=True,
                verbose=False,
                info_callback=self._collect_eval_episode_info,  # Pass callback to collect stats
            )
            
            # Transform keys to match expected format
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
        except Exception as e:
            if self.verbose:
                print(f"Warning: MRR evaluation failed: {e}")
            return {}
    
    def _save_best_model(self, iteration: int, global_step: int, metric_value: float):
        """Save the best model checkpoint."""
        if self.save_path is None:
            return
        
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint info
        info = {
            'metric': self._metric_log_key,
            'metric_name': self._metric_display_name,
            'best_value': float(metric_value),
            'timesteps': int(global_step),
            'iteration': int(iteration),
        }
        info_path = save_dir / f'info_best_eval_{self.model_name}.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        if self.verbose:
            print(f"  Saved checkpoint info to {info_path}")


# ============================================================================
# Training Checkpoint Callback
# ============================================================================


class TrainingCheckpointCallback:
    """
    Callback to save model checkpoints during training.
    
    Tracks a monitored metric (e.g., 'rollout/ep_rew_mean') and saves
    checkpoints at regular intervals. Adapted from SB3's checkpoint callback.
    """
    
    def __init__(
        self,
        save_path: Optional[Path] = None,
        model_name: str = "model",
        monitor_metric: str = "ep_rew",
        save_freq: int = 1,
        maximize: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            save_path: Path to save model checkpoints
            model_name: Name prefix for saved models
            monitor_metric: Metric name to monitor (without 'rollout/' prefix)
            save_freq: Save checkpoint every N iterations
            maximize: If True, higher values are better
            verbose: Print messages when saving
        """
        self.save_path = Path(save_path) if save_path else None
        self.model_name = model_name
        self.monitor_metric = monitor_metric
        self.save_freq = save_freq
        self.maximize = maximize
        self.verbose = verbose
        
        self.best_value = -float('inf') if maximize else float('inf')
        self.best_step = None
        self.current_value = None
    
    def should_save(self, iteration: int) -> bool:
        """Check if we should save at this iteration."""
        return (iteration % self.save_freq == 0) or (iteration == 0)
    
    def on_iteration_end(
        self,
        iteration: int,
        global_step: int,
        metrics: Dict[str, Any],
    ):
        """Called at the end of each training iteration."""
        # Extract monitored metric
        if self.monitor_metric not in metrics:
            if self.verbose and iteration == 0:
                print(f"Warning: Monitored metric '{self.monitor_metric}' not found in metrics.")
            return
        
        self.current_value = metrics[self.monitor_metric]
        
        # Check for improvement
        improved = (
            (self.maximize and self.current_value > self.best_value) or
            (not self.maximize and self.current_value < self.best_value)
        )
        
        if improved:
            self.best_value = self.current_value
            self.best_step = global_step
            if self.verbose:
                print(f"  Training metric '{self.monitor_metric}' improved to {self.current_value:.4f}")
        
        # Save checkpoint
        if self.save_path is not None and self.should_save(iteration):
            self._save_checkpoint(iteration, global_step)
    
    def _save_checkpoint(self, iteration: int, global_step: int):
        """Save a training checkpoint."""
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint info
        info = {
            'metric': self.monitor_metric,
            'current_value': float(self.current_value) if self.current_value is not None else None,
            'best_value': float(self.best_value),
            'timesteps': int(global_step),
            'iteration': int(iteration),
        }
        info_path = save_dir / f'info_last_epoch_{self.model_name}.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        if self.verbose and iteration % (self.save_freq * 5) == 0:  # Print less frequently
            print(f"  Saved checkpoint info to {info_path}")


# ============================================================================
# Depth-based Proof Statistics Tracker
# ============================================================================


class _EvalDepthRewardTracker:
    """
    Utility class to track reward and success metrics by depth and label.
    
    This is a standalone tracker that can be used by evaluation callbacks
    to accumulate depth-based statistics independently.
    """
    
    def __init__(self) -> None:
        self._rewards_by_depth: Dict[str, List[float]] = defaultdict(list)
        self._success_values_by_depth: defaultdict[Tuple[int, str], List[float]] = defaultdict(list)
        self._success_values_by_label: defaultdict[int, List[float]] = defaultdict(list)
        self._last_episode_id: Dict[int, int] = {}
    
    def __call__(self, infos: List[Dict[str, Any]]) -> None:
        """Accumulate statistics from info dicts."""
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue
            
            label = info.get("label")
            if label is None:
                continue
            
            try:
                label_value = int(label)
            except (TypeError, ValueError):
                continue
            
            depth_key = _format_depth_key(info.get("query_depth"))
            success_flag = bool(info.get("is_success", False))
            episode_data = info.get("episode")
            
            if not isinstance(episode_data, dict):
                continue
            
            # Check for duplicate episodes
            episode_idx = info.get("episode_idx")
            if episode_idx is not None:
                if self._last_episode_id.get(env_idx) == episode_idx:
                    continue
                self._last_episode_id[env_idx] = episode_idx
            else:
                episode_id = id(episode_data)
                if self._last_episode_id.get(env_idx) == episode_id:
                    continue
                self._last_episode_id[env_idx] = episode_id
            
            # Accumulate success values
            self._success_values_by_depth[(label_value, depth_key)].append(1.0 if success_flag else 0.0)
            self._success_values_by_label[label_value].append(1.0 if success_flag else 0.0)
            
            # Only track rewards for positive labels
            if label_value != 1:
                continue
            
            reward = episode_data.get("r")
            if reward is None:
                continue
            self._rewards_by_depth[depth_key].append(float(reward))
    
    def metrics(self) -> Dict[str, Union[float, int]]:
        """Compute and return accumulated metrics."""
        def reward_sort_key(item: Tuple[str, List[float]]) -> Tuple[float, str]:
            depth_str = item[0]
            if depth_str == "unknown":
                return (float("inf"), depth_str)
            try:
                return (float(int(depth_str)), depth_str)
            except ValueError:
                return (float("inf"), depth_str)
        
        def success_sort_key(item: Tuple[Tuple[int, str], List[float]]) -> Tuple[int, Union[int, float]]:
            (label, depth_str), _ = item
            label_order = 0 if label == 1 else 1 if label == 0 else 2
            if depth_str == "unknown":
                depth_order: Union[int, float] = float("inf")
            else:
                try:
                    depth_order = int(depth_str)
                except (TypeError, ValueError):
                    depth_order = float("inf")
            return (label_order, depth_order)
        
        metrics: Dict[str, Union[float, int]] = {}
        
        # Reward metrics by depth
        for depth_key, rewards in sorted(self._rewards_by_depth.items(), key=reward_sort_key):
            if not rewards:
                continue
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            count = rewards_arr.size
            mean_reward = float(rewards_arr.mean())
            std_reward = float(rewards_arr.std()) if count > 1 else 0.0
            base = f"reward_d_{depth_key}_pos"
            metrics[f"{base}_mean"] = mean_reward
            metrics[f"{base}_std"] = std_reward
            metrics[f"{base}_count"] = int(count)
        
        # Success metrics by depth and label
        for (label, depth_key), values in sorted(self._success_values_by_depth.items(), key=success_sort_key):
            if not values:
                continue
            values_arr = np.asarray(values, dtype=np.float32)
            count = values_arr.size
            mean_success = float(values_arr.mean())
            std_success = float(values_arr.std()) if count > 1 else 0.0
            label_str = "pos" if label == 1 else "neg" if label == 0 else f"label{label}"
            base = f"proven_d_{depth_key}_{label_str}"
            metrics[f"{base}_mean"] = mean_success
            metrics[f"{base}_std"] = std_success
            metrics[f"{base}_count"] = int(count)
        
        # Success metrics by label (aggregate)
        for label in (1, 0):
            values = self._success_values_by_label.get(label, [])
            if not values:
                continue
            values_arr = np.asarray(values, dtype=np.float32)
            count = values_arr.size
            mean_success = float(values_arr.mean())
            std_success = float(values_arr.std()) if count > 1 else 0.0
            label_str = "pos" if label == 1 else "neg"
            base = f"proven_{label_str}"
            metrics[f"{base}_mean"] = mean_success
            metrics[f"{base}_std"] = std_success
            metrics[f"{base}_count"] = int(count)
        
        return metrics
