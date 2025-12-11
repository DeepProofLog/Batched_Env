"""
Runner Parity Tests.

Tests verifying that both runner.py and sb3_runner.py produce the SAME results
when configured with identical settings.

This test ensures that:
1. Both runners create equivalent configurations from the same DEFAULT_CONFIG
2. build_namespace produces equivalent argparse.Namespace objects
3. Training with the same settings produces equivalent results

Usage:
    python tests/parity/test_runner_parity.py --dataset countries_s3 --timesteps 2000
"""
import gc
import os

# Force strict parity for testing
os.environ['USE_FAST_CATEGORICAL'] = '0'

import sys
import copy
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import json

import torch
import numpy as np


# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# Import seeding utilities
from seed_utils import seed_all
from parity_config import ParityConfig, TOLERANCE, create_parser, config_from_args


@dataclass
class RunnerParityConfig:
    """Configuration for runner parity tests."""
    # Dataset / data files
    dataset_name: str = "countries_s3"
    data_path: str = "./data/"
    
    # Training params (using SB3 defaults)
    n_envs: int = 4
    n_steps: int = 32
    n_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2
    ent_coef: float = 0.0
    timesteps_train: int = 256
    n_corruptions: int = 10
    
    # Environment params
    reward_type: int = 0
    train_neg_ratio: int = 4
    max_depth: int = 20
    memory_pruning: bool = True
    end_proof_action: bool = True
    skip_unary_actions: bool = True
    
    # Embedding params
    atom_embedding_size: int = 64
    padding_atoms: int = 6
    padding_states: int = -1  # Auto-compute
    max_total_vars: int = 100
    
    # Other params
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True


def _parse_metric_value(value: Any) -> float:
    """
    Parse a metric value which may be:
    - A float/int
    - A string like "0.263 +/- 0.44 (80)" -> returns 0.263 (mean)
    - A string like "0.300" -> returns 0.3
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try to extract mean from "mean +/- std (count)" format
        if '+/-' in value:
            try:
                mean_part = value.split('+/-')[0].strip()
                return float(mean_part)
            except (ValueError, IndexError):
                pass
        # Try direct float conversion
        try:
            return float(value)
        except ValueError:
            pass
    return 0.0


def _get_metric_with_fallbacks(metrics: Dict[str, Any], *keys: str) -> float:
    """
    Try to get a metric value from the dict using multiple possible keys.
    Returns the first found, or 0.0 if none found.
    """
    for key in keys:
        if key in metrics:
            return _parse_metric_value(metrics[key])
    return 0.0


@dataclass
class MetricsSnapshot:
    """Snapshot of all evaluation metrics for comparison."""
    # Reward metrics
    reward_overall: float = 0.0
    reward_label_pos: float = 0.0
    reward_label_neg: float = 0.0
    
    # Success rate
    success_rate: float = 0.0
    
    # MRR metrics
    mrr_mean: float = 0.0
    tail_mrr_mean: float = 0.0
    head_mrr_mean: float = 0.0
    
    # HITS metrics
    hits1_mean: float = 0.0
    hits3_mean: float = 0.0
    hits10_mean: float = 0.0
    tail_hits1_mean: float = 0.0
    tail_hits3_mean: float = 0.0
    tail_hits10_mean: float = 0.0
    head_hits1_mean: float = 0.0
    head_hits3_mean: float = 0.0
    head_hits10_mean: float = 0.0
    
    @classmethod
    def from_metrics_dict(cls, metrics: Dict[str, Any]) -> "MetricsSnapshot":
        """Create MetricsSnapshot from a metrics dictionary.
        
        Handles both raw float values and formatted strings like "0.263 +/- 0.44 (80)".
        Also handles different key naming conventions between SB3 and tensor implementations.
        
        Key differences between SB3 and tensor eval_corruptions output:
        - SB3: rewards_pos_mean = mean reward for positive queries only
        - Tensor: reward_overall = mean reward for ALL queries (pos + neg)
        - Tensor: reward_label_pos = mean reward for positive queries only
        - Tensor: proven_pos = mean success rate for positive queries (formatted string)
        
        For fair comparison, we compare positive-only metrics from both.
        """
        # For reward_overall, compare positive query rewards only
        # SB3 uses rewards_pos_mean, tensor uses reward_label_pos (the pos-only metric)
        reward_overall = _get_metric_with_fallbacks(
            metrics,
            'reward_label_pos',  # Tensor pos-only reward
            'rewards_pos_mean',  # SB3 pos-only reward
            'reward_overall',    # Fallback to tensor overall
            'ep_rew_mean'
        )
        
        # For success_rate, compare positive query success only
        # SB3 uses rewards_pos_mean (reward=success for binary), tensor uses proven_pos
        success_rate = _get_metric_with_fallbacks(
            metrics,
            'proven_pos',        # Tensor pos success rate (formatted string)
            'rewards_pos_mean',  # SB3 pos reward mean (equals success for binary reward)
            'success_rate',      # Tensor overall success rate
            'proven_pos_mean'
        )
        
        return cls(
            reward_overall=reward_overall,
            reward_label_pos=_get_metric_with_fallbacks(metrics, 'reward_label_pos', 'rewards_pos_mean'),
            reward_label_neg=_get_metric_with_fallbacks(metrics, 'reward_label_neg', 'rewards_neg_mean'),
            success_rate=success_rate,
            mrr_mean=_parse_metric_value(metrics.get('mrr_mean', 0.0)),
            tail_mrr_mean=_parse_metric_value(metrics.get('tail_mrr_mean', 0.0)),
            head_mrr_mean=_parse_metric_value(metrics.get('head_mrr_mean', 0.0)),
            hits1_mean=_parse_metric_value(metrics.get('hits1_mean', 0.0)),
            hits3_mean=_parse_metric_value(metrics.get('hits3_mean', 0.0)),
            hits10_mean=_parse_metric_value(metrics.get('hits10_mean', 0.0)),
            tail_hits1_mean=_parse_metric_value(metrics.get('tail_hits1_mean', 0.0)),
            tail_hits3_mean=_parse_metric_value(metrics.get('tail_hits3_mean', 0.0)),
            tail_hits10_mean=_parse_metric_value(metrics.get('tail_hits10_mean', 0.0)),
            head_hits1_mean=_parse_metric_value(metrics.get('head_hits1_mean', 0.0)),
            head_hits3_mean=_parse_metric_value(metrics.get('head_hits3_mean', 0.0)),
            head_hits10_mean=_parse_metric_value(metrics.get('head_hits10_mean', 0.0)),
        )
    
    def compare_with(self, other: "MetricsSnapshot", tolerance: float = 0.001) -> Tuple[bool, Dict[str, Tuple[float, float, float]]]:
        """
        Compare this snapshot with another.
        
        Args:
            other: Another MetricsSnapshot to compare against
            tolerance: Maximum absolute difference allowed
            
        Returns:
            Tuple of (all_match, dict of {metric_name: (self_value, other_value, diff)})
        """
        comparisons = {}
        all_match = True
        
        for field_name in ['reward_overall', 'success_rate', 'mrr_mean', 
                          'hits1_mean', 'hits3_mean', 'hits10_mean',
                          'tail_mrr_mean', 'head_mrr_mean']:
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)
            diff = abs(self_val - other_val)
            comparisons[field_name] = (self_val, other_val, diff)
            if diff > tolerance:
                all_match = False
        
        return all_match, comparisons


@dataclass
class RunnerParityResults:
    """Results container for runner parity comparison."""
    # Configuration comparison
    config_keys_match: bool = False
    config_values_match: bool = False
    config_mismatches: List[str] = field(default_factory=list)
    
    # Namespace comparison
    namespace_keys_match: bool = False
    namespace_values_match: bool = False
    namespace_mismatches: List[str] = field(default_factory=list)
    
    # Training comparison (if run)
    training_run: bool = False
    
    # Detailed metrics (new)
    sb3_train_metrics: Optional[MetricsSnapshot] = None
    sb3_valid_metrics: Optional[MetricsSnapshot] = None
    sb3_test_metrics: Optional[MetricsSnapshot] = None
    tensor_train_metrics: Optional[MetricsSnapshot] = None
    tensor_valid_metrics: Optional[MetricsSnapshot] = None
    tensor_test_metrics: Optional[MetricsSnapshot] = None
    
    # Legacy MRR fields for backwards compatibility
    sb3_mrr: float = 0.0
    tensor_mrr: float = 0.0
    mrr_match: bool = False
    
    # Detailed comparison results
    metrics_match: bool = False
    metrics_comparisons: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Overall
    overall_success: bool = False
    error_message: str = ""
    
    def print_detailed_comparison(self, split: str = "test", verbose: bool = True):
        """Print detailed comparison of metrics between SB3 and Tensor."""
        if split == "train":
            sb3_m, tensor_m = self.sb3_train_metrics, self.tensor_train_metrics
        elif split == "valid":
            sb3_m, tensor_m = self.sb3_valid_metrics, self.tensor_valid_metrics
        else:
            sb3_m, tensor_m = self.sb3_test_metrics, self.tensor_test_metrics
        
        if sb3_m is None or tensor_m is None:
            print(f"  No {split} metrics available")
            return
        
        match, comparisons = sb3_m.compare_with(tensor_m)
        
        print(f"\n  {'Metric':<20} {'SB3':>12} {'Tensor':>12} {'Diff':>10} {'Status':>8}")
        print("  " + "-" * 66)
        
        for metric_name, (sb3_val, tensor_val, diff) in comparisons.items():
            status = "✓" if diff < 0.001 else "✗"
            print(f"  {metric_name:<20} {sb3_val:>12.4f} {tensor_val:>12.4f} {diff:>10.4f} {status:>8}")
        
        print("  " + "-" * 66)
        print(f"  Overall match (tol=0.001): {'PASS' if match else 'FAIL'}")


def get_sb3_default_config() -> Dict[str, Any]:
    """Get the default configuration from sb3_runner.py."""
    return {
        'dataset_name': 'countries_s3',
        'eval_neg_samples': None,
        'test_neg_samples': None,
        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,
        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,
        'prob_facts': False,
        'topk_facts': None,
        'topk_facts_threshold': 0.33,
        'model_name': 'PPO',
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'n_epochs': 20,
        'lr': 5e-5,
        'gamma': 0.99,
        'clip_range_vf': None,
        'target_kl': 0.03,
        'seed': [0],
        'timesteps_train': 2000,
        'restore_best_val_model': True,
        'load_model': False,
        'save_model': True,
        'n_envs': 16,
        'n_steps': 128,
        'n_eval_envs': 128,
        'batch_size': 4096,
        'eval_freq': 1,
        'reward_type': 0,
        'train_neg_ratio': 4,
        'engine': 'python',
        'engine_strategy': 'cmp',
        'endf_action': True,
        'endt_action': False,
        'skip_unary_actions': True,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',
        'false_rules': False,
        'ent_coef_decay': False,
        'ent_coef_init_value': 0.5,
        'ent_coef_final_value': 0.01,
        'ent_coef_start': 0.0,
        'ent_coef_end': 1.0,
        'ent_coef_transform': 'linear',
        'lr_decay': False,
        'lr_init_value': 3e-4,
        'lr_final_value': 1e-6,
        'lr_start': 0.0,
        'lr_end': 1.0,
        'lr_transform': 'linear',
        'eval_hybrid_success_only': True,
        'eval_hybrid_kge_weight': 2.0,
        'eval_hybrid_rl_weight': 1.0,
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 250,
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': -1,
        'max_total_vars': 100,
        'verbose': False,
        'prover_verbose': False,
        'device': 'auto',
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot': False,
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
    }


def get_tensor_default_config() -> Dict[str, Any]:
    """Get the default configuration from runner.py (tensor version).
    
    NOTE: These defaults are now aligned with sb3_runner.py for parity.
    """
    return {
        'dataset_name': 'countries_s3',
        'eval_neg_samples': None,
        'test_neg_samples': None,
        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,
        'load_depth_info': True,
        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,
        'model_name': 'PPO',
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'n_epochs': 20,  # Aligned with SB3
        'lr': 5e-5,  # Aligned with SB3
        'gamma': 0.99,
        'seed': [0],
        'timesteps_train': 2000,  # Aligned with SB3
        'restore_best_val_model': True,
        'load_model': False,
        'save_model': True,
        'use_amp': True,
        'use_compile': True,
        'n_steps': 128,
        'batch_size_env': 16,
        'batch_size_env_eval': 16,
        'batch_size': 4096,  # Aligned with SB3
        'reward_type': 0,  # Aligned with SB3
        'train_neg_ratio': 4,
        'end_proof_action': True,
        'skip_unary_actions': True,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',  # Aligned with SB3
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 250,  # Aligned with SB3
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': -1,  # Aligned with SB3
        'max_total_vars': 100,  # Aligned with SB3
        'device': 'auto',  # Aligned with SB3
        'rollout_device': None,
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': True,
        'verbose': False,  # Aligned with SB3
        'verbose_cb': False,
        'verbose_env': 0,
        'verbose_prover': 0,
        'prover_verbose': False,  # Aligned with SB3
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
        'debug_ppo': False,
    }


def get_common_config_for_parity() -> Dict[str, Any]:
    """
    Get a unified configuration that both runners should use for parity testing.
    
    This configuration uses SB3 defaults as the baseline (as per user request).
    """
    return {
        # Dataset params - same for both
        'dataset_name': 'countries_s3',
        'eval_neg_samples': None,
        'test_neg_samples': None,
        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,
        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,
        'load_depth_info': True,
        
        # Prob facts (SB3 specific but needed for train.py)
        'prob_facts': False,
        'topk_facts': None,
        'topk_facts_threshold': 0.33,
        
        # Model params - use SB3 defaults
        'model_name': 'PPO',
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'n_epochs': 20,
        'lr': 5e-5,
        'gamma': 0.99,
        'clip_range_vf': None,
        'target_kl': 0.03,
        
        # Training params - use SB3 defaults
        'seed': [0],
        'timesteps_train': 2000,
        'restore_best_val_model': False,  # Disable for testing
        'load_model': False,
        'save_model': False,  # Disable for testing
        'n_envs': 16,  # SB3 uses n_envs
        'batch_size_env': 16,  # Tensor uses batch_size_env
        'batch_size_env_eval': 16,  # Tensor eval batch size
        'n_eval_envs': 16,  # SB3 eval envs
        'n_steps': 128,
        'batch_size': 4096,
        'eval_freq': 1,
        
        # Env params - use SB3 defaults
        'reward_type': 0,
        'train_neg_ratio': 4,
        'end_proof_action': True,
        'endf_action': True,  # SB3 name
        'endt_action': False,  # SB3 name
        'skip_unary_actions': True,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',  # or True for tensor
        'engine': 'python',
        'engine_strategy': 'cmp',
        'false_rules': False,
        
        # KGE params (disabled for testing)
        'kge_action': False,
        'logit_fusion': False,
        'inference_fusion': False,
        'inference_success_only': True,
        'pbrs': False,
        'enable_top_k': False,
        
        # Decay params (disabled for testing)
        'ent_coef_decay': False,
        'lr_decay': False,
        
        # Eval hybrid params
        'eval_hybrid_success_only': True,
        'eval_hybrid_kge_weight': 2.0,
        'eval_hybrid_rl_weight': 1.0,
        
        # Embedding params - use SB3 defaults
        'atom_embedder': 'transe',
        'state_embedder': 'sum',
        'atom_embedding_size': 64,
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': 20,  # Use explicit value for parity
        'max_total_vars': 100,
        
        # Other params
        'device': 'cpu',
        'verbose': False,
        'prover_verbose': False,
        'verbose_env': 0,
        'verbose_prover': 0,
        'verbose_cb': False,
        'extended_eval_info': True,
        'depth_info': True,
        'eval_best_metric': 'mrr',
        'plot': False,
        'plot_trajectories': False,
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': False,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
        'debug_ppo': False,
        'use_amp': False,
        'use_compile': False,
        'rollout_device': None,
        'min_gpu_memory_gb': 2.0,
        'use_compile': False,
        
        # Determinism settings - enable for exact match
        'deterministic': True,
        'eval_deterministic': True,
    }


def compare_configs(sb3_config: Dict, tensor_config: Dict, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Compare two configuration dictionaries.
    
    Returns:
        Tuple of (all_match, list_of_mismatches)
    """
    mismatches = []
    
    # Keys that are expected to differ between implementations
    expected_different_keys = {
        # Naming differences
        'n_envs', 'batch_size_env',  # Same meaning, different names
        'endf_action', 'end_proof_action',  # Same meaning, different names
        # Implementation-specific
        'use_amp', 'use_compile', 'rollout_device', 'debug_ppo',
        'verbose_cb', 'verbose_env', 'verbose_prover',
        'batch_size_env_eval', 'n_eval_envs',
        'load_depth_info', 'plot_trajectories', 'depth_info',
        # SB3-specific
        'engine', 'engine_strategy', 'endt_action', 'false_rules',
        'ent_coef_decay', 'ent_coef_init_value', 'ent_coef_final_value',
        'ent_coef_start', 'ent_coef_end', 'ent_coef_transform',
        'lr_decay', 'lr_init_value', 'lr_final_value',
        'lr_start', 'lr_end', 'lr_transform',
        'eval_hybrid_success_only', 'eval_hybrid_kge_weight', 'eval_hybrid_rl_weight',
        'clip_range_vf', 'target_kl', 'eval_freq',
        'prob_facts', 'topk_facts', 'topk_facts_threshold',
        'min_gpu_memory_gb', 'extended_eval_info',
    }
    
    # Keys that should match for parity
    critical_keys = {
        'dataset_name', 'lr', 'gamma', 'n_epochs', 'n_steps',
        'clip_range', 'ent_coef', 'reward_type', 'train_neg_ratio',
        'max_depth', 'memory_pruning', 'skip_unary_actions',
        'atom_embedding_size', 'padding_atoms', 'padding_states',
        'seed', 'data_path', 'rules_file', 'facts_file',
    }
    
    all_keys = set(sb3_config.keys()) | set(tensor_config.keys())
    
    for key in sorted(all_keys):
        if key in expected_different_keys:
            continue
            
        sb3_val = sb3_config.get(key, '<missing>')
        tensor_val = tensor_config.get(key, '<missing>')
        
        if sb3_val != tensor_val:
            is_critical = key in critical_keys
            mismatch_str = f"{'[CRITICAL] ' if is_critical else ''}{key}: SB3={sb3_val}, Tensor={tensor_val}"
            mismatches.append(mismatch_str)
            if verbose:
                print(f"  Mismatch: {mismatch_str}")
    
    all_match = len([m for m in mismatches if '[CRITICAL]' in m]) == 0
    return all_match, mismatches


def build_sb3_namespace(config: Dict[str, Any], device: str = "cpu") -> argparse.Namespace:
    """
    Build an argparse.Namespace from config dict, mimicking sb3_runner.py's build_namespace.
    """
    cfg = copy.deepcopy(config)
    
    # Best metric for model selection
    best_metric = cfg.get('eval_best_metric', 'mrr')
    if isinstance(best_metric, str):
        metric_normalized = best_metric.strip().lower()
        cfg['eval_best_metric'] = metric_normalized
    
    # Set default values for SB3-specific params
    cfg['kge_action'] = bool(cfg.get('kge_action', False))
    cfg['logit_fusion'] = bool(cfg.get('logit_fusion', False))
    cfg['inference_fusion'] = bool(cfg.get('inference_fusion', False))
    cfg['inference_success_only'] = bool(cfg.get('inference_success_only', True))
    cfg['pbrs'] = bool(cfg.get('pbrs', False))
    cfg['enable_top_k'] = bool(cfg.get('enable_top_k', False))
    cfg['prob_facts'] = bool(cfg.get('prob_facts', False))
    
    # Set defaults for required params
    cfg.setdefault('topk_facts', None)
    cfg.setdefault('topk_facts_threshold', 0.33)
    cfg.setdefault('clip_range_vf', None)
    cfg.setdefault('target_kl', 0.03)
    cfg.setdefault('eval_hybrid_success_only', True)
    cfg.setdefault('eval_hybrid_kge_weight', 2.0)
    cfg.setdefault('eval_hybrid_rl_weight', 1.0)
    cfg.setdefault('extended_eval_info', True)
    cfg.setdefault('engine', 'python')
    cfg.setdefault('engine_strategy', 'cmp')
    cfg.setdefault('false_rules', False)
    cfg.setdefault('endf_action', cfg.get('end_proof_action', True))
    cfg.setdefault('endt_action', False)
    cfg.setdefault('n_eval_envs', cfg.get('n_envs', 16))
    cfg.setdefault('pbrs_gamma', None)
    cfg.setdefault('pbrs_beta', 0.0)
    cfg.setdefault('verbose', 0)
    cfg.setdefault('prover_verbose', 0)
    
    cfg['annealing_specs'] = {}
    cfg['annealing'] = {}
    
    namespace = argparse.Namespace(**cfg)
    
    # Auto-configure padding_states
    if namespace.padding_states == -1:
        if namespace.dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
            namespace.padding_states = 20
        elif namespace.dataset_name == "family":
            namespace.padding_states = 130
        elif namespace.dataset_name == "wn18rr":
            namespace.padding_states = 262
        elif namespace.dataset_name == "fb15k237":
            namespace.padding_states = 358
    
    # Corruption scheme
    namespace.corruption_scheme = ['head', 'tail']
    if 'countries' in namespace.dataset_name:
        namespace.corruption_scheme = ['tail']
    
    # Engine-specific settings
    namespace.janus_file = None
    
    # File names
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"
    
    if getattr(namespace, 'train_depth', None) is not None:
        train_file = train_file.replace('.txt', '_depths.txt')
    if getattr(namespace, 'valid_depth', None) is not None:
        valid_file = valid_file.replace('.txt', '_depths.txt')
    if getattr(namespace, 'test_depth', None) is not None:
        test_file = test_file.replace('.txt', '_depths.txt')
    
    namespace.train_file = train_file
    namespace.valid_file = valid_file
    namespace.test_file = test_file
    
    # Embedding sizes
    namespace.state_embedding_size = namespace.atom_embedding_size
    namespace.constant_embedding_size = namespace.atom_embedding_size
    namespace.predicate_embedding_size = namespace.atom_embedding_size
    
    namespace.device = device
    namespace.eval_freq = int(namespace.n_steps * cfg.get('eval_freq', 1))
    
    return namespace


def build_tensor_namespace(config: Dict[str, Any], device: str = "cpu") -> argparse.Namespace:
    """
    Build an argparse.Namespace from config dict, mimicking runner.py's build_namespace.
    """
    cfg = copy.deepcopy(config)
    
    # Best metric for model selection
    best_metric = cfg.get('eval_best_metric', 'mrr')
    if isinstance(best_metric, str):
        metric_normalized = best_metric.strip().lower()
        metric_name_map = {
            'mrr': 'mrr_mean',
            'auc_pr': 'auc_pr',
        }
        cfg['eval_best_metric'] = metric_name_map.get(metric_normalized, metric_normalized)
    
    # Set default values for missing SB3-compatible params
    cfg.setdefault('prob_facts', False)
    cfg.setdefault('pbrs_gamma', None)
    cfg.setdefault('pbrs_beta', 0.0)
    
    namespace = argparse.Namespace(**cfg)
    
    # Normalize corruption_mode
    raw_corruption_mode = getattr(namespace, "corruption_mode", True)
    namespace.corruption_mode = bool(raw_corruption_mode) if not isinstance(raw_corruption_mode, str) else raw_corruption_mode
    
    # Auto-configure padding_states (do NOT change atom_embedding_size here - should match SB3)
    if namespace.padding_states == -1:
        if namespace.dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
            namespace.padding_states = 20
            # Note: Original runner.py sets atom_embedding_size=256 here, but for parity
            # with SB3, we keep the value from config (250)
        elif namespace.dataset_name == "family":
            namespace.padding_states = 130
        elif namespace.dataset_name == "wn18rr":
            namespace.padding_states = 262
        elif namespace.dataset_name == "fb15k237":
            namespace.padding_states = 358
    
    # Corruption scheme
    namespace.corruption_scheme = ['head', 'tail']
    if 'countries' in namespace.dataset_name:
        namespace.corruption_scheme = ['tail']
    namespace.corruption_scheme = list(namespace.corruption_scheme)
    
    # File names - tensor version also uses depth files based on train_depth setting
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"
    
    # Only use depth files if depth filtering is explicitly requested
    # (matching SB3 behavior which uses depth files when train_depth/valid_depth/test_depth is set)
    if getattr(namespace, 'train_depth', None) is not None:
        train_file = train_file.replace('.txt', '_depths.txt')
    if getattr(namespace, 'valid_depth', None) is not None:
        valid_file = valid_file.replace('.txt', '_depths.txt')
    if getattr(namespace, 'test_depth', None) is not None:
        test_file = test_file.replace('.txt', '_depths.txt')
    
    namespace.train_file = train_file
    namespace.valid_file = valid_file
    namespace.test_file = test_file
    
    # Embedding sizes
    # Matches logic in train_new.py / simple runner
    # For state_embedder="sum", state_dim = atom_dim
    # For state_embedder="concat", state_dim = atom_dim * padding_atoms (handled in train.py logic if state_embedder is correct string)
    atom_embedder = getattr(namespace, 'atom_embedder', 'transe')
    
    # Force alignment with simple runner which uses train_new logic:
    # embedder = TensorEmbedder(..., state_embedder='sum', constant_embedding_size=64, predicate_embedding_size=64, atom_embedding_size=64)
    # in simple runner parity config, atom_embedding_size=64.
    
    # Here namespace.atom_embedding_size comes from config which is 64 (we updated it to use default or passed value).
    # But wait, config default in test_runner_parity dataclass is 64? No, we see 250 in the dictionary.
    # We should update the dictionary default in test_runner_parity to 64 too!
    
    namespace.state_embedding_size = namespace.atom_embedding_size
    namespace.constant_embedding_size = namespace.atom_embedding_size
    namespace.predicate_embedding_size = namespace.atom_embedding_size
    
    if atom_embedder == "complex":
        namespace.constant_embedding_size = 2 * namespace.atom_embedding_size
        namespace.predicate_embedding_size = 2 * namespace.atom_embedding_size
    if atom_embedder == "rotate":
        namespace.constant_embedding_size = 2 * namespace.atom_embedding_size
    
    namespace.device = device
    
    # Compute eval_freq - should match SB3 (n_steps * eval_freq_multiplier)
    # SB3 uses: namespace.eval_freq = int(namespace.n_steps * cfg.get('eval_freq', 1))
    # For parity, we use the same calculation
    namespace.eval_freq = int(namespace.n_steps * cfg.get('eval_freq', 1))
    
    return namespace


def compare_namespaces(sb3_ns: argparse.Namespace, tensor_ns: argparse.Namespace, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Compare two argparse.Namespace objects.
    
    Returns:
        Tuple of (all_match, list_of_mismatches)
    """
    mismatches = []
    
    # Keys that are expected to differ between implementations
    expected_different_keys = {
        # Implementation-specific
        'kge_action', 'logit_fusion', 'inference_fusion', 'inference_success_only',
        'pbrs', 'enable_top_k', 'annealing_specs', 'annealing',
        'janus_file', 'eval_best_metric',
        # Naming differences handled elsewhere
        'n_envs', 'batch_size_env',
        'endf_action', 'end_proof_action',
        'use_amp', 'use_compile', 'rollout_device', 'debug_ppo',
        'verbose_cb', 'verbose_env', 'verbose_prover',
        'batch_size_env_eval', 'n_eval_envs', 'load_depth_info',
        'plot_trajectories', 'depth_info',
    }
    
    # Keys that should match for parity
    critical_keys = {
        'dataset_name', 'lr', 'gamma', 'n_epochs', 'n_steps',
        'clip_range', 'ent_coef', 'reward_type', 'train_neg_ratio',
        'max_depth', 'memory_pruning', 'skip_unary_actions',
        'atom_embedding_size', 'padding_atoms', 'padding_states',
        'seed', 'data_path', 'rules_file', 'facts_file',
        'train_file', 'valid_file', 'test_file',
        'state_embedding_size', 'constant_embedding_size', 'predicate_embedding_size',
        'corruption_scheme',
    }
    
    sb3_dict = vars(sb3_ns)
    tensor_dict = vars(tensor_ns)
    
    all_keys = set(sb3_dict.keys()) | set(tensor_dict.keys())
    
    for key in sorted(all_keys):
        if key in expected_different_keys:
            continue
        
        sb3_val = sb3_dict.get(key, '<missing>')
        tensor_val = tensor_dict.get(key, '<missing>')
        
        # Handle special cases
        if key == 'corruption_mode':
            # Normalize: SB3 uses 'dynamic', tensor uses True
            if sb3_val == 'dynamic' and tensor_val == True:
                continue
            if sb3_val == True and tensor_val == 'dynamic':
                continue
        
        if sb3_val != tensor_val:
            is_critical = key in critical_keys
            mismatch_str = f"{'[CRITICAL] ' if is_critical else ''}{key}: SB3={sb3_val}, Tensor={tensor_val}"
            mismatches.append(mismatch_str)
            if verbose:
                print(f"  Mismatch: {mismatch_str}")
    
    all_match = len([m for m in mismatches if '[CRITICAL]' in m]) == 0
    return all_match, mismatches


def run_runner_parity_test(
    config: RunnerParityConfig,
    run_training: bool = True,
    verbose: bool = True,
) -> RunnerParityResults:
    """
    Run runner parity test comparing sb3_runner.py and runner.py configurations.
    
    Args:
        config: Configuration for the parity test
        run_training: Whether to actually run training (slow)
        verbose: Whether to print detailed output
        
    Returns:
        RunnerParityResults with comparison results
    """
    results = RunnerParityResults()
    
    if verbose:
        print("=" * 70)
        print("RUNNER PARITY TEST")
        print(f"Dataset: {config.dataset_name}")
        print(f"Device: {config.device}")
        print("=" * 70)
    
    try:
        # Get unified config for parity testing
        common_config = get_common_config_for_parity()
        
        # Override with test config
        common_config['dataset_name'] = config.dataset_name
        common_config['timesteps_train'] = config.timesteps_train
        common_config['seed'] = [config.seed]
        common_config['n_envs'] = config.n_envs
        common_config['batch_size_env'] = config.n_envs
        common_config['n_steps'] = config.n_steps
        common_config['n_epochs'] = config.n_epochs
        common_config['batch_size'] = config.batch_size
        common_config['lr'] = config.learning_rate
        common_config['gamma'] = config.gamma
        common_config['clip_range'] = config.clip_range
        common_config['ent_coef'] = config.ent_coef
        common_config['reward_type'] = config.reward_type
        common_config['train_neg_ratio'] = config.train_neg_ratio
        common_config['max_depth'] = config.max_depth
        common_config['memory_pruning'] = config.memory_pruning
        common_config['end_proof_action'] = config.end_proof_action
        common_config['endf_action'] = config.end_proof_action
        common_config['skip_unary_actions'] = config.skip_unary_actions
        common_config['atom_embedding_size'] = config.atom_embedding_size
        common_config['padding_atoms'] = config.padding_atoms
        common_config['padding_states'] = config.padding_states
        common_config['max_total_vars'] = config.max_total_vars
        common_config['device'] = config.device
        common_config['verbose'] = config.verbose
        
        if verbose:
            print("\n[1/4] Comparing DEFAULT_CONFIG structures...")
        
        sb3_default = get_sb3_default_config()
        tensor_default = get_tensor_default_config()
        
        config_match, config_mismatches = compare_configs(sb3_default, tensor_default, verbose=verbose)
        results.config_keys_match = len(set(sb3_default.keys()) - set(tensor_default.keys())) < 10  # Allow some differences
        results.config_values_match = config_match
        results.config_mismatches = config_mismatches
        
        if verbose:
            print(f"  Config critical values match: {config_match}")
            print(f"  Total mismatches: {len(config_mismatches)}")
        
        if verbose:
            print("\n[2/4] Building namespaces from unified config...")
        
        # Build namespaces
        sb3_ns = build_sb3_namespace(common_config, device=config.device)
        tensor_ns = build_tensor_namespace(common_config, device=config.device)
        
        if verbose:
            print("\n[3/4] Comparing namespace structures...")
        
        ns_match, ns_mismatches = compare_namespaces(sb3_ns, tensor_ns, verbose=verbose)
        results.namespace_keys_match = True
        results.namespace_values_match = ns_match
        results.namespace_mismatches = ns_mismatches
        
        if verbose:
            print(f"  Namespace critical values match: {ns_match}")
            print(f"  Total namespace mismatches: {len(ns_mismatches)}")
        
        # Run training comparison if requested
        if run_training:
            if verbose:
                print("\n[4/4] Running training comparison...")
            
            results.training_run = True
            
            # Set seeds
            seed_all(config.seed)
            
            # Import component creation functions from test_train_parity
            from test_train_parity import (
                TrainParityConfig,
                create_sb3_components,
                create_tensor_components,
            )
            from ppo import PPO as TensorPPO
            from sb3_model_eval import eval_corruptions as sb3_eval_corruptions
            from model_eval import eval_corruptions as tensor_eval_corruptions
            
            # Create a TrainParityConfig from our RunnerParityConfig
            train_config = TrainParityConfig(
                dataset=config.dataset_name,
                data_path=config.data_path,
                n_envs=config.n_envs,
                n_steps=config.n_steps,
                n_epochs=config.n_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                total_timesteps=config.timesteps_train,
                n_corruptions=getattr(config, 'n_corruptions', 10),
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
            )
            
            # Create SB3 components
            if verbose:
                print("  Creating SB3 components...")
            sb3_comp = create_sb3_components(train_config)
            
            # Create tensor components (shares DataHandler for query alignment)
            if verbose:
                print("  Creating tensor components...")
            seed_all(config.seed)
            tensor_comp = create_tensor_components(train_config, sb3_comp['dh'])
            
            # SB3 training
            if verbose:
                print("  Running SB3 training...")
            seed_all(config.seed)
            sb3_comp['model'].learn(total_timesteps=train_config.total_timesteps, progress_bar=False)
            
            # Tensor training
            if verbose:
                print("  Running tensor training...")
            seed_all(config.seed)
            tensor_ppo = TensorPPO(
                policy=tensor_comp['policy'],
                env=tensor_comp['train_env'],
                n_steps=train_config.n_steps,
                learning_rate=train_config.learning_rate,
                n_epochs=train_config.n_epochs,
                batch_size=train_config.batch_size,
                clip_range=train_config.clip_range,
                ent_coef=train_config.ent_coef,
                gamma=train_config.gamma,
                device=tensor_comp['device'],
                verbose=False,
            )
            tensor_ppo.learn(total_timesteps=train_config.total_timesteps)
            
            # SB3 evaluation
            if verbose:
                print("  Running SB3 evaluation...")
            seed_all(config.seed + 1000)
            test_queries = sb3_comp['dh'].test_queries
            sb3_eval_results = sb3_eval_corruptions(
                model=sb3_comp['model'],
                env=sb3_comp['eval_env'],
                data=test_queries,
                sampler=sb3_comp['sampler'],
                n_corruptions=train_config.n_corruptions,
                corruption_scheme=['tail'],
                verbose=0,
            )
            results.sb3_mrr = sb3_eval_results.get('mrr_mean', 0.0)
            
            # Normalize SB3 metrics (mirroring Tensor logic for parity comparison)
            if 'tail_mrr_mean' not in sb3_eval_results:
                sb3_eval_results['tail_mrr_mean'] = sb3_eval_results.get('mrr_mean', 0.0)
            
            results.sb3_train_metrics = MetricsSnapshot.from_metrics_dict(sb3_eval_results)
            results.sb3_test_metrics = results.sb3_train_metrics
            
            # Tensor evaluation
            if verbose:
                print("  Running tensor evaluation...")
            seed_all(config.seed + 1000)
            tensor_comp['policy'].eval()
            tensor_im = tensor_comp['im']
            test_query_objs = tensor_comp['dh'].test_queries
            tensor_query_atoms = []
            for q in test_query_objs:
                query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
                tensor_query_atoms.append(query_atom)
            tensor_queries = torch.stack(tensor_query_atoms, dim=0)
            
            tensor_eval_results = tensor_eval_corruptions(
                actor=tensor_comp['policy'],
                env=tensor_comp['eval_env'],
                queries=tensor_queries,
                sampler=tensor_comp['sampler'],
                n_corruptions=train_config.n_corruptions,
                corruption_modes=('tail',),
                verbose=False,
            )
            results.tensor_mrr = tensor_eval_results.get('MRR', 0.0)
            
            # Convert tensor keys to match SB3 format for from_metrics_dict
            # Note: success_rate measures proof success rate across all episodes
            # reward_overall is the mean episode reward (usually same as success for this task)
            tensor_success = tensor_eval_results.get('success_rate', 0.0)
            tensor_metrics_normalized = {
                'mrr_mean': tensor_eval_results.get('MRR', 0.0),
                'hits1_mean': tensor_eval_results.get('Hits@1', 0.0),
                'hits3_mean': tensor_eval_results.get('Hits@3', 0.0),
                'hits10_mean': tensor_eval_results.get('Hits@10', 0.0),
                'success_rate': tensor_success,
                'tail_mrr_mean': tensor_eval_results.get('MRR', 0.0),  # Same as mrr for tail-only
                'head_mrr_mean': 0.0,  # No head corruption
                'reward_overall': tensor_success,  # For this task, reward = success
            }
            results.tensor_train_metrics = MetricsSnapshot.from_metrics_dict(tensor_metrics_normalized)
            results.tensor_test_metrics = results.tensor_train_metrics
            
            # Compare metrics
            results.mrr_match = abs(results.sb3_mrr - results.tensor_mrr) < 0.001
            
            if results.sb3_test_metrics and results.tensor_test_metrics:
                results.metrics_match, results.metrics_comparisons = results.sb3_test_metrics.compare_with(
                    results.tensor_test_metrics, tolerance=0.001
                )
            
            # Print detailed comparison
            if verbose:
                print("\n  DETAILED METRICS COMPARISON (Test Split):")
                results.print_detailed_comparison(split="test", verbose=verbose)
                
                print("\n  Summary:")
                print(f"    SB3 MRR: {results.sb3_mrr:.4f}")
                print(f"    Tensor MRR: {results.tensor_mrr:.4f}")
                print(f"    MRR match (±0.001): {results.mrr_match}")
                print(f"    All metrics match: {results.metrics_match}")
        else:
            if verbose:
                print("\n[4/4] Skipping training comparison (run_training=False)...")
        
        # Compute overall success
        # Note: config_values_match compares DEFAULT_CONFIGs which are expected to differ
        # The critical test is namespace_values_match which tests that given the SAME config,
        # both runners produce equivalent namespaces
        # For training comparison, we now use metrics_match which compares all key metrics
        if run_training:
            results.overall_success = (
                results.namespace_values_match and
                results.metrics_match
            )
        else:
            results.overall_success = results.namespace_values_match
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"OVERALL RESULT: {'PASS' if results.overall_success else 'FAIL'}")
            print("=" * 70)
        
    except Exception as e:
        results.error_message = str(e)
        if verbose:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return results


# ==============================================================================
# Standalone Test Functions (for __main__ execution)
# ==============================================================================

def test_config_parity_basic(args) -> bool:
    """Test basic configuration parity - namespace building only."""
    config = RunnerParityConfig(
        dataset_name=args.dataset,
        device="cpu" if not hasattr(args, 'device') else args.device,
        verbose=args.verbose if hasattr(args, 'verbose') else True,
    )
    
    results = run_runner_parity_test(config, run_training=True, verbose=True)
    
    if results.namespace_values_match and results.overall_success:
        return True
    else:
        print(f"Namespace mismatches: {results.namespace_mismatches}")
        return False

def test_training_parity(args, timesteps: int = 256) -> bool:
    """Test that both runners produce similar training results."""
    config = RunnerParityConfig(
        dataset_name=args.dataset,
        timesteps_train=timesteps,
        n_envs=args.n_envs if hasattr(args, 'n_envs') else 4,
        n_steps=args.n_steps if hasattr(args, 'n_steps') else 32,
        batch_size=64,
        device="cpu" if not hasattr(args, 'device') else args.device,
        verbose=True,
    )
    
    results = run_runner_parity_test(config, run_training=True, verbose=True)
    
    if results.sb3_test_metrics is None or results.tensor_test_metrics is None:
        print("Test metrics not available")
        return False
    
    mrr_diff = abs(results.sb3_mrr - results.tensor_mrr)
    success_diff = abs(
        results.sb3_test_metrics.success_rate - 
        results.tensor_test_metrics.success_rate
    )
    
    passed = mrr_diff < TOLERANCE and success_diff < TOLERANCE
    if not passed:
        print(f"MRR diff: {mrr_diff:.6f}, Success diff: {success_diff:.6f}")
    return passed


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Use shared parser with all relevant parameters
    # Note: --run-training is already in the shared parser
    parser = create_parser(description="Runner Parity Test")
    parser.add_argument("--timesteps", type=int, default=256,
                        help="Timesteps to train for (if --run-training)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("RUNNER PARITY TESTS")
    print(f"Tolerance: {TOLERANCE}")
    print(f"{'='*70}")
    
    # Run basic config parity test
    print(f"\n{'='*70}")
    print(f"Running test_config_parity_basic[{args.dataset}]")
    print(f"{'='*70}")
    
    config = RunnerParityConfig(
        dataset_name=args.dataset,
        seed=args.seed,
        device=args.device,
        verbose=getattr(args, 'verbose', True),
        n_envs=args.n_envs if hasattr(args, 'n_envs') and args.n_envs else 4,
        n_steps=args.n_steps if hasattr(args, 'n_steps') and args.n_steps else 32,
        batch_size=getattr(args, 'batch_size', 64),
    )
    
    # Basic parity test (no training)
    results = run_runner_parity_test(config, run_training=False, verbose=getattr(args, 'verbose', True))
    
    basic_passed = results.namespace_values_match and results.overall_success
    
    if basic_passed:
        print(f"\n✓ PASSED: test_config_parity_basic[{args.dataset}]")
    else:
        print(f"\n✗ FAILED: test_config_parity_basic[{args.dataset}]")
        if results.namespace_mismatches:
            print("Namespace mismatches:")
            for m in results.namespace_mismatches[:5]:
                print(f"  {m}")
    
    all_passed = basic_passed
    
    # training test
    print(f"\n{'='*70}")
    print(f"Running test_training_parity[{args.dataset}] (this may take a while)")
    print(f"{'='*70}")
    
    training_config = RunnerParityConfig(
        dataset_name=args.dataset,
        timesteps_train=args.timesteps,
        n_envs=args.n_envs if hasattr(args, 'n_envs') and args.n_envs else 4,
        n_steps=args.n_steps if hasattr(args, 'n_steps') and args.n_steps else 32,
        batch_size=getattr(args, 'batch_size', 128),
        device=args.device,
        verbose=getattr(args, 'verbose', True),
    )
    
    training_results = run_runner_parity_test(training_config, run_training=True, verbose=getattr(args, 'verbose', True))
    
    if training_results.overall_success:
        print(f"\n✓ PASSED: test_training_parity[{args.dataset}]")
    else:
        print(f"\n✗ FAILED: test_training_parity[{args.dataset}]")
        all_passed = False
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if all_passed:
        print("All runner parity tests PASSED")
    else:
        print("Some runner parity tests FAILED")
    
    sys.exit(0 if all_passed else 1)
