"""
Callbacks Parity Tests.

Tests verifying that the TorchRL callbacks produce the same metrics
and behavior as the SB3 callbacks.
"""
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))


# ============================================================================
# DetailedMetricsCollector Tests
# ============================================================================

def test_detailed_metrics_collector_basic():
    """Test DetailedMetricsCollector basic accumulation."""
    from callbacks import DetailedMetricsCollector
    
    collector = DetailedMetricsCollector(collect_detailed=True, verbose=False)
    
    # Create some test infos
    infos = [
        {
            "episode": {"r": 1.0, "l": 5},
            "label": 1,
            "query_depth": 0,
            "is_success": True,
        },
        {
            "episode": {"r": 0.5, "l": 3},
            "label": 0,
            "query_depth": None,
            "is_success": False,
        },
    ]
    
    collector.accumulate(infos)
    metrics = collector.compute_metrics()
    
    # Should have metrics for both labels
    assert "len_pos" in metrics or any("pos" in k for k in metrics)
    assert "len_neg" in metrics or any("neg" in k for k in metrics)


def test_detailed_metrics_collector_depth_breakdown():
    """Test that depth breakdown is computed correctly."""
    from callbacks import DetailedMetricsCollector
    
    collector = DetailedMetricsCollector(collect_detailed=True, verbose=False)
    
    # Add multiple episodes at different depths
    infos = [
        {"episode": {"r": 1.0, "l": 5}, "label": 1, "query_depth": 0, "is_success": True},
        {"episode": {"r": 0.8, "l": 4}, "label": 1, "query_depth": 1, "is_success": True},
        {"episode": {"r": 0.6, "l": 3}, "label": 1, "query_depth": 2, "is_success": False},
    ]
    
    # Add each info separately to avoid duplicate detection
    for i, info in enumerate(infos):
        info["episode_idx"] = i
        collector.accumulate([info])
    
    metrics = collector.compute_metrics()
    
    # Should have depth-specific metrics
    depth_keys = [k for k in metrics if "d_0" in k or "d_1" in k or "d_2" in k]
    assert len(depth_keys) > 0, "Should have depth-specific metrics"


def test_detailed_metrics_collector_duplicate_rejection():
    """Test that duplicate episodes are rejected."""
    from callbacks import DetailedMetricsCollector
    
    collector = DetailedMetricsCollector(collect_detailed=True, verbose=False)
    
    # Same episode_idx should be rejected
    info = {
        "episode": {"r": 1.0, "l": 5},
        "label": 1,
        "query_depth": 0,
        "is_success": True,
        "episode_idx": 0,
    }
    
    collector.accumulate([info])
    collector.accumulate([info])  # Duplicate
    collector.accumulate([info])  # Duplicate
    
    metrics = collector.compute_metrics()
    
    # Should only have 1 episode counted
    # Check count in formatted strings
    for k, v in metrics.items():
        if "pos" in k and isinstance(v, str) and "(" in v:
            count = int(v.split("(")[-1].rstrip(")"))
            assert count == 1, f"Should have 1 episode, got {count} for {k}"
            break


def test_detailed_metrics_collector_reset():
    """Test that reset clears accumulated data."""
    from callbacks import DetailedMetricsCollector
    
    collector = DetailedMetricsCollector(collect_detailed=True, verbose=False)
    
    infos = [
        {"episode": {"r": 1.0, "l": 5}, "label": 1, "query_depth": 0, "is_success": True, "episode_idx": 0},
    ]
    
    collector.accumulate(infos)
    assert len(collector.compute_metrics()) > 0
    
    collector.reset()
    assert len(collector.compute_metrics()) == 0


# ============================================================================
# Format Functions Tests
# ============================================================================

def test_format_depth_key():
    """Test depth key formatting."""
    from callbacks import _format_depth_key
    
    # Normal depths
    assert _format_depth_key(0) == "0"
    assert _format_depth_key(1) == "1"
    assert _format_depth_key(10) == "10"
    
    # Special cases
    assert _format_depth_key(-1) == "unknown"
    assert _format_depth_key(None) == "unknown"


def test_format_stat_string():
    """Test stat string formatting."""
    from callbacks import _format_stat_string
    
    # Normal case
    result = _format_stat_string(0.5, 0.1, 10)
    assert "0.500" in result
    assert "0.10" in result
    assert "10" in result
    
    # Edge cases
    assert _format_stat_string(None, 0.1, 10) == "N/A"
    assert _format_stat_string(0.5, None, 10) == "N/A"
    assert _format_stat_string(0.5, 0.1, 0) == "N/A"


# ============================================================================
# TrainingMetricsCallback Tests
# ============================================================================

def test_training_metrics_callback_accumulation():
    """Test TrainingMetricsCallback episode stats accumulation."""
    from callbacks import TrainingMetricsCallback
    
    callback = TrainingMetricsCallback(log_interval=1, verbose=False, collect_detailed=True)
    callback.on_training_start()
    
    infos = [
        {"episode": {"r": 1.0, "l": 5}, "label": 1, "query_depth": 0, "is_success": True, "episode_idx": 0},
        {"episode": {"r": 0.5, "l": 3}, "label": 0, "query_depth": None, "is_success": False, "episode_idx": 1},
    ]
    
    callback.accumulate_episode_stats(infos)
    
    # Check metrics via the collector
    metrics = callback.metrics_collector.compute_metrics()
    assert len(metrics) > 0


def test_training_metrics_callback_timing():
    """Test TrainingMetricsCallback timing metrics."""
    from callbacks import TrainingMetricsCallback
    import time
    
    callback = TrainingMetricsCallback(log_interval=1, verbose=False)
    callback.on_training_start()
    
    time.sleep(0.1)  # Brief pause
    
    # Call internal method to compute timing
    time_metrics = callback._compute_time_metrics(iteration=1, global_step=100, n_envs=4)
    
    assert "iterations" in time_metrics
    assert time_metrics["iterations"] == "1"
    assert "total_timesteps" in time_metrics
    assert time_metrics["total_timesteps"] == "100"


# ============================================================================
# RolloutProgressCallback Tests
# ============================================================================

def test_rollout_progress_callback():
    """Test RolloutProgressCallback step tracking."""
    from callbacks import RolloutProgressCallback
    
    callback = RolloutProgressCallback(
        total_steps=100,
        n_envs=4,
        update_interval=25,
        verbose=False
    )
    
    callback.on_rollout_start()
    assert callback.current_steps == 0
    
    callback.on_step(0)
    assert callback.current_steps == 4  # 1 * n_envs
    
    callback.on_step(9)
    assert callback.current_steps == 40  # 10 * n_envs
    
    callback.on_rollout_end()


# ============================================================================
# EvaluationCallback Tests
# ============================================================================

def test_evaluation_callback_should_evaluate():
    """Test EvaluationCallback evaluation frequency."""
    from callbacks import EvaluationCallback
    
    callback = EvaluationCallback(
        eval_env=None,
        sampler=None,
        eval_data=[],
        eval_freq=5,
        verbose=False,
    )
    
    # Should evaluate at iteration 0, 5, 10, etc.
    assert callback.should_evaluate(0)
    assert not callback.should_evaluate(1)
    assert not callback.should_evaluate(4)
    assert callback.should_evaluate(5)
    assert callback.should_evaluate(10)


def test_evaluation_callback_metrics_collection():
    """Test EvaluationCallback metrics collection."""
    from callbacks import EvaluationCallback
    
    callback = EvaluationCallback(
        eval_env=None,
        sampler=None,
        eval_data=[],
        eval_freq=1,
        verbose=False,
        collect_detailed=True,
    )
    
    callback.on_evaluation_start(iteration=1, global_step=100)
    
    infos = [
        {"episode": {"r": 1.0, "l": 5}, "label": 1, "query_depth": 0, "is_success": True, "episode_idx": 0},
    ]
    callback.accumulate_episode_stats(infos)
    
    metrics = callback.compute_metrics()
    assert len(metrics) > 0


# ============================================================================
# TorchRLCallbackManager Tests
# ============================================================================

def test_callback_manager_integration():
    """Test TorchRLCallbackManager coordinates callbacks correctly."""
    from callbacks import TorchRLCallbackManager, RolloutProgressCallback, TrainingMetricsCallback
    
    rollout_cb = RolloutProgressCallback(total_steps=100, n_envs=4, verbose=False)
    train_cb = TrainingMetricsCallback(log_interval=1, verbose=False)
    
    manager = TorchRLCallbackManager(
        rollout_callback=rollout_cb,
        train_callback=train_cb,
    )
    
    # Test callback methods
    manager.on_training_start()
    manager.on_rollout_start()
    manager.on_rollout_step(0)
    manager.on_rollout_end()
    
    # Test episode stats accumulation
    infos = [
        {"episode": {"r": 1.0, "l": 5}, "label": 1, "query_depth": 0, "is_success": True, "episode_idx": 0},
    ]
    manager.accumulate_episode_stats(infos, mode="train")


def test_callback_manager_tensor_batch_conversion():
    """Test that TorchRLCallbackManager converts tensor batches correctly."""
    from callbacks import TorchRLCallbackManager, TrainingMetricsCallback
    
    train_cb = TrainingMetricsCallback(log_interval=1, verbose=False)
    manager = TorchRLCallbackManager(train_callback=train_cb)
    manager.on_training_start()
    
    # Pass tensor batch format
    tensor_batch = {
        "r": torch.tensor([1.0, 0.5]),
        "l": torch.tensor([5, 3]),
        "label": torch.tensor([1, 0]),
        "query_depth": torch.tensor([0, -1]),
        "is_success": torch.tensor([True, False]),
    }
    
    manager.accumulate_episode_stats(tensor_batch, mode="train")
    
    # Should have accumulated stats
    metrics = train_cb.metrics_collector.compute_metrics()
    assert len(metrics) > 0


# ============================================================================
# MRREvaluationCallback Tests
# ============================================================================

def test_mrr_callback_metric_options():
    """Test MRREvaluationCallback supports different best metrics."""
    from callbacks import MRREvaluationCallback
    
    # Test valid metrics
    for metric in ["mrr_mean", "hits1_mean", "hits3_mean", "hits10_mean"]:
        callback = MRREvaluationCallback(
            eval_env=None,
            sampler=None,
            eval_data=torch.randn(10, 3),
            best_metric=metric,
            verbose=False,
        )
        assert callback.best_metric == metric


def test_mrr_callback_invalid_metric():
    """Test MRREvaluationCallback rejects invalid metrics."""
    from callbacks import MRREvaluationCallback
    
    with pytest.raises(ValueError):
        MRREvaluationCallback(
            eval_env=None,
            sampler=None,
            eval_data=torch.randn(10, 3),
            best_metric="invalid_metric",
        )


# ============================================================================
# TrainingCheckpointCallback Tests
# ============================================================================

def test_training_checkpoint_should_save():
    """Test TrainingCheckpointCallback save frequency."""
    from callbacks import TrainingCheckpointCallback
    
    callback = TrainingCheckpointCallback(
        save_path=None,
        save_freq=5,
        verbose=False,
    )
    
    assert callback.should_save(0)
    assert not callback.should_save(1)
    assert callback.should_save(5)
    assert callback.should_save(10)


def test_training_checkpoint_improvement_tracking():
    """Test TrainingCheckpointCallback tracks improvement correctly."""
    from callbacks import TrainingCheckpointCallback
    
    callback = TrainingCheckpointCallback(
        save_path=None,
        monitor_metric="ep_rew",
        maximize=True,
        verbose=False,
    )
    
    # Initial state
    assert callback.best_value == -float('inf')
    
    # First call with improvement
    callback.on_iteration_end(
        iteration=0,
        global_step=100,
        metrics={"ep_rew": 1.0}
    )
    assert callback.best_value == 1.0
    
    # Better value
    callback.on_iteration_end(
        iteration=1,
        global_step=200,
        metrics={"ep_rew": 2.0}
    )
    assert callback.best_value == 2.0
    
    # Worse value - should not update
    callback.on_iteration_end(
        iteration=2,
        global_step=300,
        metrics={"ep_rew": 1.5}
    )
    assert callback.best_value == 2.0


# ============================================================================
# _EvalDepthRewardTracker Tests
# ============================================================================

def test_eval_depth_reward_tracker():
    """Test _EvalDepthRewardTracker accumulation."""
    from callbacks import _EvalDepthRewardTracker
    
    tracker = _EvalDepthRewardTracker()
    
    infos = [
        {"episode": {"r": 1.0}, "label": 1, "query_depth": 0, "is_success": True, "episode_idx": 0},
        {"episode": {"r": 0.8}, "label": 1, "query_depth": 1, "is_success": True, "episode_idx": 1},
        {"episode": {"r": 0.0}, "label": 0, "query_depth": None, "is_success": False, "episode_idx": 2},
    ]
    
    tracker(infos)
    metrics = tracker.metrics()
    
    # Should have metrics
    assert len(metrics) > 0
    
    # Should have depth-specific success metrics
    success_keys = [k for k in metrics if "proven_d_" in k]
    assert len(success_keys) > 0


# ============================================================================
# Parity Tests: TorchRL vs SB3
# ============================================================================

def test_format_functions_parity():
    """Test that format functions produce identical output."""
    from callbacks import _format_depth_key as trl_format_depth
    from callbacks import _format_stat_string as trl_format_stat
    from sb3_callbacks import _format_depth_key as sb3_format_depth
    from sb3_callbacks import _format_stat_string as sb3_format_stat
    
    # Test depth key formatting
    test_depths = [0, 1, 2, 10, -1, None]
    for depth in test_depths:
        assert trl_format_depth(depth) == sb3_format_depth(depth), f"Depth {depth}"
    
    # Test stat string formatting
    test_cases = [
        (0.5, 0.1, 10),
        (1.0, 0.0, 1),
        (None, 0.1, 10),
        (0.5, None, 10),
        (0.5, 0.1, 0),
    ]
    for mean, std, count in test_cases:
        assert trl_format_stat(mean, std, count) == sb3_format_stat(mean, std, count), \
            f"Stats ({mean}, {std}, {count})"


def test_depth_reward_tracker_parity():
    """Test that both _EvalDepthRewardTracker implementations produce same metrics."""
    from callbacks import _EvalDepthRewardTracker as TRLTracker
    from sb3_callbacks import _EvalDepthRewardTracker as SB3Tracker
    
    # Create test data
    infos = [
        {"episode": {"r": 1.0}, "label": 1, "query_depth": 0, "is_success": True, "episode_idx": 0},
        {"episode": {"r": 0.8}, "label": 1, "query_depth": 1, "is_success": True, "episode_idx": 1},
        {"episode": {"r": 0.6}, "label": 1, "query_depth": 2, "is_success": False, "episode_idx": 2},
        {"episode": {"r": 0.0}, "label": 0, "query_depth": None, "is_success": False, "episode_idx": 3},
    ]
    
    trl_tracker = TRLTracker()
    sb3_tracker = SB3Tracker()
    
    trl_tracker(infos)
    sb3_tracker(infos)
    
    trl_metrics = trl_tracker.metrics()
    sb3_metrics = sb3_tracker.metrics()
    
    # Both should have same keys
    assert set(trl_metrics.keys()) == set(sb3_metrics.keys()), \
        f"TRL keys: {trl_metrics.keys()}, SB3 keys: {sb3_metrics.keys()}"
    
    # Values should match
    for key in trl_metrics:
        trl_val = trl_metrics[key]
        sb3_val = sb3_metrics[key]
        if isinstance(trl_val, float):
            assert abs(trl_val - sb3_val) < 1e-6, f"Mismatch for {key}: TRL={trl_val}, SB3={sb3_val}"
        else:
            assert trl_val == sb3_val, f"Mismatch for {key}: TRL={trl_val}, SB3={sb3_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
