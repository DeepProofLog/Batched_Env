"""
TorchRL Callbacks for Neural-guided Grounding.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict

# ============================================================================
# Display Helper
# ============================================================================

class Display:
    """Helper for formatting and displaying metrics."""
    
    @staticmethod
    def _format_stat_string(mean: Optional[float], std: Optional[float], count: int) -> str:
        """Format statistics as a display string."""
        if mean is None or std is None or count == 0:
            return "N/A"
        return f"{mean:.3f} +/- {std:.2f} ({count})"

    @staticmethod
    def print_formatted_metrics(
        metrics: Dict[str, Any],
        prefix: str = "rollout",
        extra_metrics: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """Print metrics in a formatted table."""
        print("-" * 52)
        
        final_output = {}
        
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (float, int, np.number)):
                    final_output[k] = f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                else:
                    final_output[k] = str(v)
        
        if extra_metrics:
            for k, v in extra_metrics.items():
                if isinstance(v, (float, int, np.number)):
                    final_output[k] = f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                else:
                    final_output[k] = str(v)
        
        if global_step is not None and "total_timesteps" not in final_output:
            final_output["total_timesteps"] = str(global_step)
        
        if final_output:
            print(f"| {prefix + '/':<23} | {'':<24} |")
            for key in sorted(final_output.keys()):
                print(f"|    {key:<20} | {final_output[key]:<24} |")
        
        print("-" * 52)
        print()

    @staticmethod
    def _format_depth_key(depth_value: Any) -> str:
        """Normalize depth IDs."""
        if depth_value == -1:
            return "unknown"
        if depth_value is None:
            return "unknown"
        return str(depth_value)


# ============================================================================
# Trackers
# ============================================================================

class MRRTracker:
    """Tracks MRR during training."""
    
    def __init__(self, patience: int = 10):
        self.history: list = []
        self.best_mrr: float = 0.0
        self.best_iteration: int = 0
        self.patience = patience
        self.no_improvement_count: int = 0
        
    def update(self, mrr: float, iteration: int) -> Dict[str, Any]:
        self.history.append({'iteration': iteration, 'mrr': mrr})
        is_best = mrr > self.best_mrr
        if is_best:
            self.best_mrr = mrr
            self.best_iteration = iteration
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        return {
            'current_mrr': mrr,
            'best_mrr': self.best_mrr,
            'best_iteration': self.best_iteration,
            'is_best': is_best,
            'no_improvement_count': self.no_improvement_count,
        }
    
    def get_summary(self) -> str:
        if not self.history:
            return "No MRR data recorded"
        current = self.history[-1]['mrr']
        return f"MRR: current={current:.3f}, best={self.best_mrr:.3f} (iter {self.best_iteration})"


class RewardTracker:
    """Tracks training rewards."""
    
    def __init__(self, patience: int = 10):
        self.history: list = []
        self.best_reward: float = float('-inf')
        self.best_iteration: int = 0
        self.patience = patience
        self.no_improvement_count: int = 0
        
    def update(self, reward: float, iteration: int) -> Dict[str, Any]:
        self.history.append({'iteration': iteration, 'reward': reward})
        is_best = reward > self.best_reward
        if is_best:
            self.best_reward = reward
            self.best_iteration = iteration
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        return {
            'current_reward': reward,
            'best_reward': self.best_reward,
            'best_iteration': self.best_iteration,
            'is_best': is_best,
        }
    
    def get_summary(self) -> str:
        if not self.history:
            return "No reward data recorded"
        current = self.history[-1]['reward']
        return f"Reward (train): current={current:.3f}, best={self.best_reward:.3f} (iter {self.best_iteration})"


# ============================================================================
# Detailed Metrics Collector
# ============================================================================

class DetailedMetricsCollector:
    """
    Common module for collecting detailed episode metrics by label and depth.
    """
    
    def __init__(self, collect_detailed: bool = True, verbose: bool = False):
        self.collect_detailed = collect_detailed
        self.verbose = verbose
        self.reset()
    
    def reset(self) -> None:
        self._episode_stats: defaultdict[Tuple[int, str], List[Dict[str, float]]] = defaultdict(list)
        self._last_episode_id: Dict[int, int] = {}
    
    def accumulate(self, infos: List[Dict[str, Any]]) -> None:
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue
            
            label = info.get("label")
            if label is None:
                continue
            
            episode_data = info.get("episode")
            if not isinstance(episode_data, dict):
                continue
            
            # Check duplicate
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
            
            try:
                label_value = int(label)
            except (TypeError, ValueError):
                continue

            # Extract stats
            reward = episode_data.get("r")
            length = episode_data.get("l")
            depth_value = info.get("query_depth")
            if label_value == 0:
                depth_value = -1 # Bucket negatives to unknown/none
            success_flag = bool(info.get("is_success", False))
            
            depth_key = Display._format_depth_key(depth_value)
            
            stats = {
                "reward": float(reward) if reward is not None else None,
                "length": float(length) if length is not None else None,
                "success": 1.0 if success_flag else 0.0,
                "depth_raw": depth_value,
            }
            self._episode_stats[(label_value, depth_key)].append(stats)
    
    def compute_metrics(self) -> Dict[str, str]:
        metrics = {}
        if not self._episode_stats:
            return metrics
        
        # 1. Overall
        all_rewards = []
        all_lengths = []
        all_successes = []
        for episodes in self._episode_stats.values():
            all_rewards.extend([ep["reward"] for ep in episodes if ep.get("reward") is not None])
            all_lengths.extend([ep["length"] for ep in episodes if ep.get("length") is not None])
            all_successes.extend([ep.get("success", 0.0) for ep in episodes])
        
        if all_rewards:
            metrics["ep_rew_mean"] = float(np.mean(all_rewards))
            metrics["reward"] = Display._format_stat_string(
                np.mean(all_rewards), np.std(all_rewards), len(all_rewards)
            )
        if all_lengths:
            metrics["ep_len_mean"] = float(np.mean(all_lengths))
            metrics["len"] = Display._format_stat_string(
                np.mean(all_lengths), np.std(all_lengths), len(all_lengths)
            )
        if all_successes:
            metrics["success_rate"] = f"{np.mean(all_successes):.3f}"
        
        # 2. By Label
        for label in (1, 0):
            label_str = "pos" if label == 1 else "neg"
            rewards, lengths, successes = [], [], []
            for (lbl, depth_key), episodes in self._episode_stats.items():
                if lbl == label:
                    for ep in episodes:
                        if ep.get("reward") is not None: rewards.append(ep["reward"])
                        if ep.get("length") is not None: lengths.append(ep["length"])
                        successes.append(ep.get("success", 0.0))
            
            if lengths:
                metrics[f"len_{label_str}"] = Display._format_stat_string(
                    np.mean(lengths), np.std(lengths), len(lengths)
                )
            if successes:
                metrics[f"proven_{label_str}"] = Display._format_stat_string(
                    np.mean(successes), np.std(successes), len(successes)
                )
            if rewards:
                 metrics[f"reward_{label_str}"] = Display._format_stat_string(
                    np.mean(rewards), np.std(rewards), len(rewards)
                )

        # 3. Detailed by Depth
        if self.collect_detailed:
            def _order_key(item):
                (label, depth_key), _ = item
                label_order = 0 if label == 1 else 1 if label == 0 else 2
                try: 
                    depth_order = int(depth_key) if depth_key != "unknown" else float('inf')
                except ValueError: 
                    depth_order = float('inf')
                return (label_order, depth_order)

            for (label, depth_key), episodes in sorted(self._episode_stats.items(), key=_order_key):
                label_str = "pos" if label == 1 else "neg" if label == 0 else f"label{label}"
                
                lengths = [ep["length"] for ep in episodes if ep.get("length") is not None]
                if lengths:
                    metrics[f"len_d_{depth_key}_{label_str}"] = Display._format_stat_string(
                        np.mean(lengths), np.std(lengths), len(lengths)
                    )
                
                successes = [ep["success"] for ep in episodes if "success" in ep]
                if successes:
                    metrics[f"proven_d_{depth_key}_{label_str}"] = Display._format_stat_string(
                        np.mean(successes), np.std(successes), len(successes)
                    )
                
                rewards = [ep["reward"] for ep in episodes if ep.get("reward") is not None]
                if rewards:
                    metrics[f"reward_d_{depth_key}_{label_str}"] = Display._format_stat_string(
                        np.mean(rewards), np.std(rewards), len(rewards)
                    )

        return metrics


# ============================================================================
# TorchRLCallbackManager
# ============================================================================

class TorchRLCallbackManager:
    """Manages a list of callbacks."""
    
    def __init__(self, callbacks: Optional[List[Any]] = None):
        self.callbacks = callbacks or []
        self.last_metrics = {}
    
    def add_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)
    
    def on_training_start(self, total_timesteps: Optional[int] = None) -> None:
        for cb in self.callbacks:
            if hasattr(cb, 'on_training_start'):
                import inspect
                sig = inspect.signature(cb.on_training_start)
                if 'total_timesteps' in sig.parameters:
                    cb.on_training_start(total_timesteps=total_timesteps)
                else:
                    cb.on_training_start()
                
    def on_iteration_start(self, iteration: int, global_step: int) -> None:
        for cb in self.callbacks:
             if hasattr(cb, 'on_iteration_start'):
                cb.on_iteration_start(iteration, global_step)

    def on_training_end(self) -> None:
        """Called at the end of training."""
        for cb in self.callbacks:
            if hasattr(cb, 'on_training_end'):
                cb.on_training_end()

    def on_step(self, infos: List[Dict[str, Any]]) -> None:
        for cb in self.callbacks:
            if hasattr(cb, 'on_step'):
                cb.on_step(infos)

    def prepare_batch_infos(
        self,
        rewards: Any,
        lengths: Any,
        done_idx_cpu: Any,
        current_query_indices: Any,
        query_labels: Any,
        query_depths: Any,
        successes: Optional[Any] = None,
        step_labels: Optional[Any] = None,  # Actual labels with negative sampling
    ) -> None:
        """Helper to construct rich infos from raw rollout stats and call on_step."""
        num_dones = len(rewards)
        if num_dones == 0:
            return

        # Use provided successes if available, else default to False
        if successes is not None:
            batch_succ = list(successes)
        else:
            batch_succ = [False] * num_dones

        # Fetch meta info if available
        batch_q_idxs = current_query_indices[done_idx_cpu] if current_query_indices is not None else None

        batch_lbls = None
        batch_depths = None

        # Use step_labels directly if provided (actual labels with negative sampling)
        if step_labels is not None:
            batch_lbls = step_labels
        elif batch_q_idxs is not None and query_labels is not None:
            # Fallback to indexing original labels (legacy behavior)
            n_labels = query_labels.shape[0]
            if n_labels > 0:
                safe_idx = torch.as_tensor(batch_q_idxs, dtype=torch.long) % n_labels
                batch_lbls = query_labels[safe_idx].numpy()

        # Get depths from query pool
        if batch_q_idxs is not None and query_depths is not None:
            n_depths = query_depths.shape[0]
            if n_depths > 0:
                safe_idx = torch.as_tensor(batch_q_idxs, dtype=torch.long) % n_depths
                batch_depths = query_depths[safe_idx].numpy()
        
        # Construct infos using efficient zipping
        iterators = [
            rewards, lengths, batch_succ,
            batch_q_idxs if batch_q_idxs is not None else [None] * num_dones,
            batch_lbls if batch_lbls is not None else [None] * num_dones,
            batch_depths if batch_depths is not None else [None] * num_dones
        ]
        
        batch_infos = [
            {
                "episode": {"r": float(r), "l": int(l)},
                "is_success": bool(s),
                **({ "episode_idx": int(q) } if q is not None else {}),
                **({ "label": int(lbl) } if lbl is not None else {}),
                **({ "query_depth": int(d) } if d is not None else {})
            }
            for r, l, s, q, lbl, d in zip(*iterators)
        ]
        
        self.on_step(batch_infos)

    def on_iteration_end(self, iteration: int, global_step: int, train_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metrics = {}
        for cb in self.callbacks:
            if hasattr(cb, 'on_iteration_end'):
                # Check if callback accepts train_metrics
                import inspect
                sig = inspect.signature(cb.on_iteration_end)
                if 'train_metrics' in sig.parameters:
                    m = cb.on_iteration_end(iteration, global_step, train_metrics=train_metrics)
                else:
                    m = cb.on_iteration_end(iteration, global_step)
                
                if m and isinstance(m, dict):
                    metrics.update(m)
        return metrics

    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """
        PPO callback wrapper.
        """
        iteration = locals_.get('iteration', 0)
        total_steps = locals_.get('total_steps_done', 0)
        
        # Merge PPO training metrics if available
        train_metrics = locals_.get('train_metrics', {})
        
        # Call on_iteration_end which aggregates metrics
        metrics = self.on_iteration_end(iteration, total_steps, train_metrics=train_metrics)
        self.last_metrics = metrics # Store for PPO to retrieve
        
        if train_metrics:
            metrics.update(train_metrics)
            self.last_metrics.update(train_metrics)
            
        # Trigger CheckpointCallback check_and_save manually with full metrics
        # This is a bridge between PPO iteration end and CheckpointCallback
        for cb in self.callbacks:
            if hasattr(cb, 'check_and_save'):
                cb.check_and_save(metrics, iteration)
                
        return True

# ============================================================================
# MetricsCallback
# ============================================================================

class MetricsCallback:
    """Collects and displays training metrics."""
    
    def __init__(self, log_interval: int = 1, verbose: bool = True, collect_detailed: bool = True):
        self.log_interval = log_interval
        self.verbose = verbose
        self.collector = DetailedMetricsCollector(collect_detailed=collect_detailed)
        self.reward_tracker = RewardTracker(patience=20)
        self.train_start_time = None
        self.last_time = None
        self.last_step = 0
    
    def on_training_start(self) -> None:
        self.train_start_time = time.time()
        self.last_time = time.time()
    
    def on_step(self, infos: List[Dict[str, Any]]) -> None:
        self.collector.accumulate(infos)
    
    def on_iteration_end(self, iteration: int, global_step: int, train_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if iteration % self.log_interval != 0:
            self.collector.reset()
            return {}
        
        # Compute metrics
        rollout_metrics = self.collector.compute_metrics()
        
        # Compute timing
        current_time = time.time()
        elapsed = current_time - self.last_time
        steps_done = global_step - self.last_step
        fps = int(steps_done / elapsed) if elapsed > 0 else 0
        
        self.last_time = current_time
        self.last_step = global_step
        
        timing = {
            "time/fps": fps,
            "time/elapsed": int(current_time - self.train_start_time),
            "total_timesteps": global_step,
        }
        
        if train_metrics and "explained_var" in train_metrics:
            timing["explained_variance"] = train_metrics["explained_var"]
        
        # Update reward tracker if 'ep_rew_mean' exists
        if "ep_rew_mean" in rollout_metrics:
            self.reward_tracker.update(rollout_metrics["ep_rew_mean"], iteration)
            if self.verbose:
                print(f"[Metrics] {self.reward_tracker.get_summary()}")

        if self.verbose:
            Display.print_formatted_metrics(
                metrics=rollout_metrics,
                prefix="rollout",
                extra_metrics=timing,
                global_step=global_step
            )
            
        # Return merged metrics for CheckpointCallback
        return {**rollout_metrics, **timing}


# ============================================================================
# CheckpointCallback
# ============================================================================

class CheckpointCallback:
    """
    Saves checkpoints based on best training reward and best evaluation metric.
    """
    
    def __init__(
        self, 
        save_path: str,
        policy: Any,
        best_model_name_train: str = "best_model_train.pt",
        best_model_name_eval: str = "best_model_eval.pt",
        train_metric: str = "ep_rew_mean",
        eval_metric: str = "mrr_mean",  # Can be 'mrr_mean', 'auc_pr', etc.
        verbose: bool = True,
        date: str = None,
        restore_best: bool = False,
        load_best_metric: str = 'eval',
        load_model: Any = False
    ):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.best_model_name_train = best_model_name_train
        self.best_model_name_eval = best_model_name_eval
        
        self.train_metric = train_metric
        self.eval_metric = eval_metric
        
        self.best_train_value = float('-inf')
        self.best_eval_value = float('-inf')
        self.verbose = verbose
        self.date = date
        self.restore_best = restore_best
        self.load_best_metric = load_best_metric
        self.load_model = load_model
        
        self._total_timesteps_config = 0 # Track for end-of-training behavior
        
    def on_training_start(self, total_timesteps: Optional[int] = None) -> None:
        """Called at the start of training to load an existing model if specified."""
        self._total_timesteps_config = total_timesteps or 0
        if self.load_model:
            if self.verbose:
                print(f"\n[Checkpoint] Loading existing model at start of training (load_model={self.load_model})...")
            
            # If load_model is a path string, load it directly
            if isinstance(self.load_model, str) and self.load_model not in ["True", "eval", "train"]:
                path = Path(self.load_model)
                if path.exists():
                    if self.verbose:
                        print(f"Loading model from path: {path}")
                    self.policy.load_state_dict(torch.load(path, map_location=next(self.policy.parameters()).device))
                    return
            
            # Otherwise use the standard load_best_model logic
            load_metric = 'eval'
            if isinstance(self.load_model, str) and self.load_model in ["eval", "train"]:
                load_metric = self.load_model
            
            self.load_best_model(load_metric=load_metric)
        
    def on_iteration_end(self, iteration: int, global_step: int) -> None:
        pass

    def on_training_end(self) -> None:
        """Called at the end of training to optionally restore best model."""
        if self.restore_best and self._total_timesteps_config > 0:
            if self.verbose:
                print("\n[Checkpoint] Restoring best model at end of training...")
            
            # Determine device from policy
            device = None
            try:
                device = next(self.policy.parameters()).device
            except:
                pass
                
            self.load_best_model(load_metric=self.load_best_metric, device=device)

    def check_and_save(self, metrics: Dict[str, Any], iteration: int) -> None:
        """Called manually or by Manager if we enhance the API."""
        
        # 1. Check Eval Metric (RankingCallback output)
        if self.eval_metric in metrics:
            val = metrics[self.eval_metric]
            # Handle formatted string/list if necessary, though typical Eval callbacks return floats for main metrics
            if isinstance(val, (list, tuple, np.ndarray)):
                 val = np.mean(val) # Should not happen usually
            if isinstance(val, str):
                try: val = float(val.split()[0])
                except: val = float('-inf')

            if val > self.best_eval_value:
                self.best_eval_value = val
                
                if self.date:
                    stem = self.best_model_name_eval.replace('.pt', '')
                    filename = f"{stem}_{self.date}.pt"
                else:
                    filename = self.best_model_name_eval

                path = self.save_path / filename
                torch.save(self.policy.state_dict(), path)
                
                # SAVE JSON INFO
                if self.date:
                    json_filename = f"info_best_eval_{self.date}.json"
                    json_path = self.save_path / json_filename
                    
                    # Try to get simplified types for json
                    explained_var = metrics.get("explained_var", None)
                    if isinstance(explained_var, torch.Tensor):
                        explained_var = explained_var.item()
                    elif isinstance(explained_var, (np.float32, np.float64)):
                        explained_var = float(explained_var)
                        
                    info = {
                        "metric": self.eval_metric,
                        "best_value": float(val),
                        "timesteps": iteration, 
                        "explained_variance": explained_var
                    }
                    try:
                        with open(json_path, 'w') as f:
                            json.dump(info, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Failed to save best model info json: {e}")

                if self.verbose:
                    print(f"[Checkpoint] New best eval model saved to {path} ({self.eval_metric}={val:.4f})")
        
        # Always save "last" train model whenever checkpoint is called
        stem = "last_model_train"
        if self.date:
            filename = f"{stem}_{self.date}.pt"
        else:
            filename = f"{stem}.pt"
        
        path = self.save_path / filename
        torch.save(self.policy.state_dict(), path)
        if self.verbose:
            print(f"[Checkpoint] Saved last train model to {path}")

    def load_best_model(self, load_metric: str = 'eval', device: Any = None) -> bool:
        """
        Loads the best model based on the specified metric ('eval' or 'train').
        If the preferred model is not found, falls back to the other one if available.
        """
        path_to_load = None
        if self.date:
             stem_train = self.best_model_name_train.replace('.pt', '')
             name_train = f"{stem_train}_{self.date}.pt"
             stem_eval = self.best_model_name_eval.replace('.pt', '')
             name_eval = f"{stem_eval}_{self.date}.pt"
        else:
             name_train = self.best_model_name_train
             name_eval = self.best_model_name_eval

        best_model_path_train = self.save_path / name_train
        best_model_path_eval = self.save_path / name_eval
        
        if load_metric == 'train':
            path_to_load = best_model_path_train
            if not path_to_load.exists():
                 # Fallback to non-dated file
                 path_to_load = self.save_path / self.best_model_name_train
            
            if not path_to_load.exists():
                # Search for latest dated train model
                stem = self.best_model_name_train.replace('.pt', '')
                import glob
                files = sorted(glob.glob(str(self.save_path / f"{stem}_*.pt")))
                if files:
                    path_to_load = Path(files[-1])
            
            if path_to_load.exists():
                print(f"Restoring best TRAIN model (reward) from {path_to_load}")
            else:
                 print(f"Warning: Best train model not found at {path_to_load} or dated version.")
                 path_to_load = None
        else: # eval
            path_to_load = best_model_path_eval
            if not path_to_load.exists():
                 # Fallback to non-dated file
                 path_to_load = self.save_path / self.best_model_name_eval
            
            if not path_to_load.exists():
                # Search for latest dated eval model
                stem = self.best_model_name_eval.replace('.pt', '')
                import glob
                files = sorted(glob.glob(str(self.save_path / f"{stem}_*.pt")))
                if files:
                    path_to_load = Path(files[-1])

            if path_to_load.exists():
                 print(f"Restoring best EVAL model (MRR) from {path_to_load}")
            else:
                 print(f"Warning: Best eval model not found at {path_to_load}. Falling back to train model?")
                 path_to_load = best_model_path_train
                 if not path_to_load.exists():
                      path_to_load = self.save_path / self.best_model_name_train
                 
                 if not path_to_load.exists():
                     # Search for latest dated train model as last resort
                     stem = self.best_model_name_train.replace('.pt', '')
                     import glob
                     files = sorted(glob.glob(str(self.save_path / f"{stem}_*.pt")))
                     if files:
                         path_to_load = Path(files[-1])

                 if path_to_load.exists():
                      print(f"Restoring best TRAIN model instead from {path_to_load}")
                 else:
                      path_to_load = None
                     
        if path_to_load:
            if device is None:
                # Try to infer device from policy
                try:
                    device = next(self.policy.parameters()).device
                except:
                    device = 'cpu'
            
            self.policy.load_state_dict(torch.load(path_to_load, map_location=device))
            
            # Load and print JSON info if available
            if self.date and load_metric == 'eval':
                json_filename = f"info_best_eval_{self.date}.json"
                json_path = self.save_path / json_filename
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            info = json.load(f)
                        print(f"Loaded best model info: {info}")
                    except Exception as e:
                        print(f"Warning: Failed to load info json: {e}")
            
            return True
        else:
            print("Warning: No best model found to restore.")
            return False


# ============================================================================
# EvaluationCallback
# ============================================================================

class EvaluationCallback:
    """Standard evaluation loop (reward-based)."""
    
    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 1,
        verbose: bool = True
    ):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.collector = DetailedMetricsCollector(collect_detailed=True)
    
    def should_evaluate(self, iteration: int) -> bool:
        return (iteration % self.eval_freq == 0)

    def on_iteration_end(self, iteration: int, global_step: int, train_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.should_evaluate(iteration):
            return {}
        
        # Run Evaluation
        # NOTE: This assumes eval_env is a vector env compatible with similar loop to train
        # Here we just implement the callback structure. The actual eval running usually happens 
        # by switching policy to eval mode and running episodes.
        
        # For this refactor, we stick to the user's focus which is RankingCallback for KG.
        # But if standard eval is needed, it would collect rewards here.
        return {}


# ============================================================================
# RankingCallback (MRR/Hits)
# ============================================================================

class RankingCallback:
    """
    Performs corruption-based evaluation (MRR, Hits@K).
    Replaces MRREvaluationCallback.
    """
    
    def __init__(
        self,
        eval_env: Any,
        policy: Any,
        sampler: Any,
        eval_data: Any, # tensor queries
        eval_data_depths: Optional[Any] = None, # tensor depths
        eval_freq: int = 1,
        n_corruptions: int = 50,
        corruption_scheme: Tuple[str] = ('head', 'tail'),
        verbose: bool = True,
        save_path: Optional[Path] = None, # Optional: if user wants to save best from here (legacy)
        ppo_agent: Any = None, # Optional: PPOOptimized instance for optimized evaluation
    ):
        self.eval_env = eval_env
        self.policy = policy
        self.sampler = sampler
        self.eval_data = eval_data
        self.eval_data_depths = eval_data_depths
        self.eval_freq = eval_freq
        self.n_corruptions = n_corruptions
        self.corruption_scheme = corruption_scheme
        self.verbose = verbose
        self.ppo_agent = ppo_agent
        
        self.mrr_tracker = MRRTracker(patience=20)
        self.last_eval_step = 0
        
    def on_training_start(self, total_timesteps: Optional[int] = None) -> None:
        """Run initial evaluation before training starts."""
        if total_timesteps == 0:
            if self.verbose:
                print("[Ranking] Skipping initial evaluation as total_timesteps=0")
            return
            
        print("\n" + "="*60 + "\nInitial evaluation (untrained model)\n" + "="*60)
        self.on_iteration_end(iteration=0, global_step=0)
        # Reset last_eval_step so we count from 0 correctly (if we want to exclude initial)
        self.last_eval_step = 0
        print("="*60 + "\n")

    def should_evaluate(self, iteration: int) -> bool:
        # NOTE: iteration passed here is roughly (timesteps / batch_size)
        # However, users typically specify eval_freq in steps.
        # Since we don't pass global_step to should_evaluate in manager (only iteration),
        # we need to be careful. The manager's on_iteration_end receives global_step.
        return True # Handled in on_iteration_end with global_step check

    def on_iteration_end(self, iteration: int, global_step: int, train_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # eval_freq is now measured in iterations (rollouts), not steps
        if iteration > 0 and iteration % self.eval_freq != 0:
            return {}

        self.policy.eval()
        
        start_time = time.time()
        
        # Optimized evaluation path
        results = self.ppo_agent.evaluate(
            queries=self.eval_data,
            sampler=self.sampler,
            n_corruptions=self.n_corruptions,
            corruption_modes=self.corruption_scheme,
            verbose=False,
            query_depths=self.eval_data_depths, 
        )

        print(f"[Eval] Took {time.time() - start_time:.2f} seconds\n")
        self.policy.train()
        
        # Parse metrics for clean dictionary
        eval_metrics = {}
        for k, v in results.items():
            # Include stats that are strings (formatted outputs) or numbers
            if isinstance(v, (int, float, str)):
                eval_metrics[k] = v
                
        # Main key for mrr
        mrr_mean = eval_metrics.get("MRR", eval_metrics.get("mrr_mean", 0.0))
        eval_metrics["mrr_mean"] = mrr_mean
        
        # Track
        self.mrr_tracker.update(mrr_mean, iteration)
        if self.verbose:
             print(f"[Ranking] {self.mrr_tracker.get_summary()}")
             Display.print_formatted_metrics(
                 metrics=eval_metrics,
                 prefix="eval",
             )
        
        return eval_metrics


# ============================================================================
# ScalarAnnealingCallback
# ============================================================================

@dataclass
class AnnealingTarget:
    name: str
    setter: Callable[[float], None]
    initial: float
    final: float
    start_point: float = 0.0          
    end_point: float = 1.0            
    transform: str = 'linear'         
    value_type: str = 'float' 

class ScalarAnnealingCallback:
    """Anneals scalars (lr, ent_coef) over training."""
    
    def __init__(self, total_timesteps: int, targets: List[AnnealingTarget], verbose: int = 0):
        self.total_timesteps = total_timesteps
        self.targets = targets
        self.verbose = verbose
        
    def on_iteration_start(self, iteration: int, global_step: int) -> None:
        progress = min(1.0, max(0.0, global_step / self.total_timesteps))
        
        for tgt in self.targets:
            # Determine effective progress for this target
            if progress < tgt.start_point:
                current_val = tgt.initial
            elif progress > tgt.end_point:
                current_val = tgt.final
            else:
                # Local progress between start_point and end_point
                p = (progress - tgt.start_point) / (tgt.end_point - tgt.start_point + 1e-9)
                p = max(0.0, min(1.0, p))
                
                if tgt.transform == 'linear':
                    current_val = tgt.initial + p * (tgt.final - tgt.initial)
                elif tgt.transform == 'exp':
                     current_val = tgt.initial * (tgt.final / tgt.initial)**p
                elif tgt.transform == 'cos':
                    current_val = tgt.final + (tgt.initial - tgt.final) * 0.5 * (1 + math.cos(math.pi * p))
                else:
                    current_val = tgt.initial + p * (tgt.final - tgt.initial)

            if tgt.value_type == 'int':
                current_val = int(current_val)
            
            # Apply
            tgt.setter(current_val)
            
            if self.verbose > 1:
                print(f"[Anneal] {tgt.name} = {current_val:.6f} (progress={progress:.2f})")

