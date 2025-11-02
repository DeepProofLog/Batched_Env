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

# Import shared utilities
from callback_utils import DetailedMetricsCollector, _format_depth_key, _format_stat_string


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
        self.start_time = time.time()
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
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if self.verbose:
                print(f"Time to collect_rollouts {elapsed:.2f}")


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
        
        # Use the common detailed metrics collector
        self.metrics_collector = DetailedMetricsCollector(collect_detailed=collect_detailed)
        
        # Tracking
        self.best_metric_value = float('-inf')
        self.current_iteration = 0
    
    def should_evaluate(self, iteration: int) -> bool:
        """Check if we should run evaluation at this iteration."""
        return (iteration % self.eval_freq == 0) or (iteration == 0)
    
    def on_evaluation_start(self, iteration: int, global_step: int):
        """Called at the start of evaluation."""
        self.current_iteration = iteration
        if self.verbose:
            print('---------------evaluation started---------------')
        self.metrics_collector.reset()
        self.eval_start_time = time.time()
    
    def accumulate_episode_stats(self, infos: List[Dict[str, Any]]):
        """Accumulate episode stats by label and depth using the common collector."""
        self.metrics_collector.accumulate(infos)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics from collected stats."""
        return self.metrics_collector.compute_metrics()
    
    def on_evaluation_end(self, iteration: int, global_step: int, eval_metrics: Dict[str, Any]):
        """Called when evaluation is complete."""
        elapsed = time.time() - self.eval_start_time
        
        # Convert episode_len_* metrics from mean/std pairs to formatted strings
        self._convert_episode_len_metrics(eval_metrics)
        
        # Check if this is a new best
        metric_value = eval_metrics.get(self.best_metric, float('-inf'))
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
        
        # Print formatted metrics
        if self.verbose:
            self._print_formatted_metrics(eval_metrics, global_step)
            print(f'---------------evaluation finished---------------  took {elapsed:.2f} seconds')
        
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
    
    def _print_formatted_metrics(self, metrics: Dict[str, Any], global_step: int):
        """Print metrics in SB3-style formatted table."""
        # Build ordered list of metrics to display
        # First, metrics that should appear at the top
        priority_metrics = []
        
        # Check for different metric name patterns
        if "_mrr" in metrics:
            priority_metrics.append("_mrr")
        elif "mrr_mean" in metrics:
            priority_metrics.append("mrr_mean")
        
        if "auc_pr" in metrics:
            priority_metrics.append("auc_pr")
        elif "auc_pr_mean" in metrics:
            priority_metrics.append("auc_pr_mean")
            
        if "success_rate" in metrics:
            priority_metrics.append("success_rate")
        
        # Episode length and reward stats (in specific order)
        ep_stats = []
        for key in ["ep_len", "ep_rew", "ep_len_pos", "ep_len_neg", 
                    "ep_len_pos_true", "ep_len_pos_false", 
                    "ep_len_neg_true", "ep_len_neg_false"]:
            if key in metrics:
                ep_stats.append(key)
        
        # Length and reward metrics by label (aggregate)
        label_metrics = []
        for k in ["len_pos", "len_neg", "proven_pos", "proven_neg", "rwd_pos", "rwd_neg"]:
            if k in metrics:
                label_metrics.append(k)
        
        # Depth-based metrics (sorted by depth then metric type)
        depth_len_keys = sorted(
            [k for k in metrics.keys() if k.startswith("len_d_")],
            key=lambda x: self._sort_depth_key(x)
        )
        depth_proven_keys = sorted(
            [k for k in metrics.keys() if k.startswith("proven_d_")],
            key=lambda x: self._sort_depth_key(x)
        )
        depth_rwd_keys = sorted(
            [k for k in metrics.keys() if k.startswith("rwd_d_")],
            key=lambda x: self._sort_depth_key(x)
        )
        
        # Legacy reward metrics (keep for backward compatibility)
        legacy_reward = [k for k in metrics.keys() if "reward_label" in k or "reward_depth" in k]
        
        # Other length metrics
        other_length = [k for k in metrics.keys() if k == "length mean +/- std"]
        
        # Combine all in desired order
        display_order = (
            priority_metrics + 
            ep_stats +
            label_metrics +
            depth_len_keys +
            depth_proven_keys +
            depth_rwd_keys +
            legacy_reward +
            other_length
        )
        
        # Remove duplicates while preserving order
        seen = set()
        display_order = [x for x in display_order if not (x in seen or seen.add(x))]
        
        # Print table
        print("-" * 52)
        print(f"| {'eval/':<23} | {'':<24} |")
        
        for key in display_order:
            if key in metrics:
                value = metrics[key]
                # Format the value
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        value_str = f"{value:.3f}"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)
                
                print(f"|    {key:<20} | {value_str:<24} |")
        
        # Add timesteps
        print(f"|    {'total_timesteps':<20} | {global_step:<24} |")
        print("-" * 52)
        print()
    
    def _sort_depth_key(self, key: str) -> Tuple[int, Union[int, float]]:
        """Sort key for depth-based metrics."""
        # Extract depth number and label
        parts = key.split('_')
        label = parts[-1]  # 'pos' or 'neg'
        label_order = 0 if label == 'pos' else 1
        
        # Extract depth - should be parts[2] for format like "len_d_0_pos"
        depth_str = parts[2] if len(parts) > 2 else "unknown"
        if depth_str == "unknown":
            depth_order = float("inf")
        else:
            try:
                depth_order = int(depth_str)
            except (TypeError, ValueError):
                depth_order = float("inf")
        
        return (label_order, depth_order)


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
        collect_detailed: bool = True,  # NEW: Enable detailed depth breakdown
    ):
        """
        Args:
            log_interval: Print metrics every N iterations
            verbose: Whether to print metrics
            collect_detailed: If True, collect detailed breakdown by depth.
                            If False, only collect aggregate stats.
        """
        self.log_interval = log_interval
        self.verbose = verbose
        
        # Use the common detailed metrics collector
        self.metrics_collector = DetailedMetricsCollector(collect_detailed=collect_detailed)
        
        self.train_start_time = None
        self.training_epoch_losses = []  # Track losses per epoch
    
    def on_training_start(self):
        """Called at the start of training."""
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
        self.metrics_collector.accumulate(infos)
    
    def on_iteration_end(
        self,
        iteration: int,
        global_step: int,
        train_metrics: Dict[str, float],
        n_envs: int = 1,
    ):
        """Called at the end of a training iteration."""
        if not self.verbose or iteration % self.log_interval != 0:
            self.metrics_collector.reset()
            return
        
        # Compute rollout metrics using the common collector
        rollout_metrics = self.metrics_collector.compute_metrics()
        
        # Compute timing metrics
        time_metrics = self._compute_time_metrics(iteration, global_step, n_envs)
        
        # Print formatted output
        self._print_training_metrics(rollout_metrics, train_metrics, time_metrics)
        
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
    
    def _print_training_metrics(
        self,
        rollout_metrics: Dict[str, str],
        train_metrics: Dict[str, str],
        time_metrics: Dict[str, str],
    ):
        """Print formatted training metrics table."""
        print("-" * 52)
        
        # Rollout metrics - organize by priority
        if rollout_metrics:
            print(f"| {'rollout/':<23} | {'':<24} |")
            
            # First: overall episode stats
            for key in ["ep_len", "ep_rew"]:
                if key in rollout_metrics:
                    print(f"|    {key:<20} | {rollout_metrics[key]:<24} |")
            
            # Second: aggregate by label
            for key in ["len_pos", "len_neg", "proven_pos", "proven_neg", "rwd_pos", "rwd_neg"]:
                if key in rollout_metrics:
                    print(f"|    {key:<20} | {rollout_metrics[key]:<24} |")
            
            # Third: detailed breakdown by depth (if collected)
            # Group by metric type for cleaner display
            len_depth_keys = sorted(
                [k for k in rollout_metrics.keys() if k.startswith("len_d_")],
                key=lambda x: self._sort_metric_key(x)
            )
            proven_depth_keys = sorted(
                [k for k in rollout_metrics.keys() if k.startswith("proven_d_")],
                key=lambda x: self._sort_metric_key(x)
            )
            rwd_depth_keys = sorted(
                [k for k in rollout_metrics.keys() if k.startswith("rwd_d_")],
                key=lambda x: self._sort_metric_key(x)
            )
            
            for key in len_depth_keys:
                print(f"|    {key:<20} | {rollout_metrics[key]:<24} |")
            for key in proven_depth_keys:
                print(f"|    {key:<20} | {rollout_metrics[key]:<24} |")
            for key in rwd_depth_keys:
                print(f"|    {key:<20} | {rollout_metrics[key]:<24} |")
        
        # Time metrics
        if time_metrics:
            print(f"| {'time/':<23} | {'':<24} |")
            for key, value in time_metrics.items():
                print(f"|    {key:<20} | {value:<24} |")
        
        # Training metrics
        if train_metrics:
            print(f"| {'train/':<23} | {'':<24} |")
            for key, value in train_metrics.items():
                print(f"|    {key:<20} | {value:<24} |")
        
        print("-" * 52)
    
    def _sort_metric_key(self, key: str) -> Tuple[int, Union[int, float]]:
        """Sort key for depth-based metrics."""
        # Extract label (pos/neg) and depth
        parts = key.split('_')
        label = parts[-1]  # 'pos' or 'neg'
        label_order = 0 if label == 'pos' else 1
        
        # Extract depth number - should be parts[2] for format like "len_d_0_pos"
        depth_str = parts[2] if len(parts) > 2 else "unknown"
        if depth_str == "unknown":
            depth_order = float("inf")
        else:
            try:
                depth_order = int(depth_str)
            except (TypeError, ValueError):
                depth_order = float("inf")
        
        return (label_order, depth_order)


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
        metrics_callback: Optional[TrainingMetricsCallback] = None,
    ):
        """
        Args:
            rollout_callback: Callback for rollout progress
            eval_callback: Callback for evaluation
            metrics_callback: Callback for training metrics
        """
        self.rollout_callback = rollout_callback
        self.eval_callback = eval_callback
        self.metrics_callback = metrics_callback  # Expose publicly for direct access
    
    def on_training_start(self):
        """Called at the start of training."""
        if self.metrics_callback:
            self.metrics_callback.on_training_start()
    
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
            self.eval_callback.accumulate_episode_stats(infos)
        elif mode == "train" and self.metrics_callback:
            self.metrics_callback.accumulate_episode_stats(infos)
    
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
        if self.metrics_callback:
            self.metrics_callback.on_iteration_end(iteration, global_step, train_metrics, n_envs)
