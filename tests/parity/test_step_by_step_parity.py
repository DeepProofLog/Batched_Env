"""
Step-by-Step Parity Debugging Test.

This test compares SB3 and Tensor implementations at each step of execution,
logging detailed information about:
- Observations (sub_index, action_mask, derived_sub_indices)
- Policy forward pass (features, latent states, logits)
- Actions selected
- Environment state after stepping

This helps identify exactly where the two implementations diverge.

Usage:
    python tests/parity/test_step_by_step_parity.py --query-idx 16 --max-steps 10
    python tests/parity/test_step_by_step_parity.py --all-queries --max-steps 5
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
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

from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from tensordict import TensorDict

from tests.parity.test_eval_parity import (
    create_aligned_environments,
    create_sb3_eval_env,
    create_tensor_eval_env,
    create_sb3_sampler,
    create_tensor_sampler,
)


@dataclass
class StepComparison:
    """Stores comparison results for a single step."""
    step: int
    
    # Observations
    sub_index_match: bool = False
    action_mask_match: bool = False
    derived_sub_indices_match: bool = False
    
    # Forward pass
    features_match: bool = False
    latent_pi_match: bool = False
    logits_match: bool = False
    
    # Actions
    sb3_action: Optional[np.ndarray] = None
    tensor_action: Optional[np.ndarray] = None
    actions_match: bool = False
    
    # Log probs
    sb3_logprob: Optional[float] = None
    tensor_logprob: Optional[float] = None
    logprobs_match: bool = False
    
    # Done flags
    sb3_done: Optional[np.ndarray] = None
    tensor_done: Optional[np.ndarray] = None
    
    # Detailed differences (for debugging)
    obs_diff_details: Dict[str, Any] = field(default_factory=dict)
    forward_diff_details: Dict[str, Any] = field(default_factory=dict)
    
    def is_fully_matched(self) -> bool:
        return (self.sub_index_match and 
                self.action_mask_match and 
                self.actions_match)


@dataclass 
class QueryComparison:
    """Stores comparison results for a single query."""
    query_idx: int
    query_str: str
    n_slots: int  # 1 positive + N negatives
    
    steps: List[StepComparison] = field(default_factory=list)
    
    # Final outcomes
    sb3_success: Optional[List[bool]] = None
    tensor_success: Optional[List[bool]] = None
    
    # Per-slot results
    sb3_total_logprobs: Optional[List[float]] = None
    tensor_total_logprobs: Optional[List[float]] = None
    
    def first_divergence_step(self) -> Optional[int]:
        """Return the first step where actions diverge, or None if all match."""
        for s in self.steps:
            if not s.actions_match:
                return s.step
        return None
    
    def all_steps_match(self) -> bool:
        return all(s.is_fully_matched() for s in self.steps)


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, 
                    name: str, rtol: float = 1e-5, atol: float = 1e-5) -> Tuple[bool, Dict]:
    """Compare two tensors and return match status and difference details."""
    if t1.shape != t2.shape:
        return False, {"error": f"Shape mismatch: {t1.shape} vs {t2.shape}"}
    
    if t1.dtype != t2.dtype:
        # Convert to float for comparison
        t1 = t1.float()
        t2 = t2.float()
    
    if t1.dtype in (torch.float32, torch.float64):
        match = torch.allclose(t1, t2, rtol=rtol, atol=atol)
        if not match:
            diff = (t1 - t2).abs()
            return False, {
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
                "diff_locations": torch.where(diff > atol),
            }
    else:
        match = torch.equal(t1, t2)
        if not match:
            diff_mask = t1 != t2
            return False, {
                "n_diffs": diff_mask.sum().item(),
                "diff_locations": torch.where(diff_mask),
            }
    
    return match, {}


def extract_sb3_forward_pass(policy, obs_tensor: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract intermediate values from SB3 policy forward pass."""
    with torch.no_grad():
        features = policy.features_extractor(obs_tensor)
        latent_pi, latent_vf = policy.mlp_extractor(features)
        action_logits = policy.action_net(latent_pi)
        
        # Apply mask to get final logits
        mask = obs_tensor['action_mask']
        masked_logits = torch.where(mask.bool(), action_logits, torch.tensor(-1e8))
        
    return {
        'features': features,
        'latent_pi': latent_pi,
        'latent_vf': latent_vf,
        'action_logits': action_logits,
        'masked_logits': masked_logits,
    }


def extract_tensor_forward_pass(policy, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract intermediate values from tensor policy forward pass."""
    with torch.no_grad():
        # The tensor policy wraps an SB3-like ActorCriticPolicy
        if hasattr(policy, 'module'):
            inner = policy.module
        else:
            inner = policy
            
        if hasattr(inner, 'features_extractor'):
            features = inner.features_extractor(obs)
            latent_pi, latent_vf = inner.mlp_extractor(features)
            action_logits = inner.action_net(latent_pi)
            
            mask = obs['action_mask']
            masked_logits = torch.where(mask.bool(), action_logits, torch.tensor(-1e8))
            
            return {
                'features': features,
                'latent_pi': latent_pi,
                'latent_vf': latent_vf,
                'action_logits': action_logits,
                'masked_logits': masked_logits,
            }
        else:
            # Fallback - just return empty dict
            return {}


def run_step_comparison(
    sb3_ppo, sb3_env,
    tensor_ppo, tensor_env,
    query_goals: List[Any],
    tensor_im,  # Index manager for tensor env
    max_steps: int = 20,
    verbose: bool = True,
    slot_idx: int = 0,  # Which slot to focus detailed comparison on
) -> QueryComparison:
    """
    Run step-by-step comparison for a single query with its corruptions.
    
    Args:
        sb3_ppo: SB3 PPO model
        sb3_env: SB3 VecEnv
        tensor_ppo: Tensor PPO model  
        tensor_env: Tensor BatchedEnv
        query_goals: List of goals [positive, neg1, neg2, ...]
        max_steps: Maximum steps to run
        verbose: Print detailed output
        slot_idx: Which slot to focus on for detailed comparisons
    """
    device = torch.device('cpu')
    n_envs = len(query_goals)
    
    result = QueryComparison(
        query_idx=-1,  # Will be set by caller
        query_str=str(query_goals[0]),
        n_slots=n_envs,
    )
    
    # Configure SB3 environments with goals
    # Access underlying envs through the vec_env
    for i in range(n_envs):
        # Get the underlying env (unwrap Monitor)
        base_env = sb3_env.envs[i]
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        base_env.goal = query_goals[i]
    
    # Reset both
    reset_result = sb3_env.reset()
    if isinstance(reset_result, tuple):
        sb3_obs = reset_result[0]
    else:
        sb3_obs = reset_result
    
    # For tensor env, need to convert goals to tensor format and set eval dataset
    # Get the index manager from tensor_env to convert goals
    im = tensor_env.im  # Assuming tensor env has index manager
    
    # Convert goals to tensor format
    query_tensors = []
    for goal in query_goals:
        # goal is a Term object - convert to tensor using im
        query_atom = im.atom_to_tensor(goal.predicate, goal.args[0], goal.args[1])
        query_tensors.append(query_atom)
    flat_queries = torch.stack(query_tensors, dim=0).unsqueeze(1)  # shape (n_envs, 1, 3)
    
    # Labels: first is positive (1), rest are negative (0)
    flat_labels = torch.zeros(n_envs, dtype=torch.long)
    flat_labels[0] = 1
    
    # Depths: use -1 for unknown
    flat_depths = torch.full((n_envs,), -1, dtype=torch.long)
    
    # Per-slot lengths: each slot has 1 query
    per_slot_lengths = torch.ones(n_envs, dtype=torch.long)
    
    tensor_env.set_eval_dataset(
        queries=flat_queries,
        labels=flat_labels,
        query_depths=flat_depths,
        per_slot_lengths=per_slot_lengths,
    )
    
    tensor_td = tensor_env.reset()
    tensor_obs = tensor_td['observation']
    
    # Track log probs per slot
    sb3_total_logps = np.zeros(n_envs)
    tensor_total_logps = np.zeros(n_envs)
    
    # Track done status
    sb3_all_done = np.zeros(n_envs, dtype=bool)
    tensor_all_done = np.zeros(n_envs, dtype=bool)
    
    for step in range(max_steps):
        step_result = StepComparison(step=step)
        
        # Convert SB3 obs to tensor
        sb3_obs_t = obs_as_tensor(sb3_obs, device)
        
        # === Compare Observations ===
        for key in ['sub_index', 'action_mask']:
            if key in sb3_obs_t and key in tensor_obs:
                match, diff = compare_tensors(sb3_obs_t[key], tensor_obs[key], key)
                if key == 'sub_index':
                    step_result.sub_index_match = match
                elif key == 'action_mask':
                    step_result.action_mask_match = match
                if not match:
                    step_result.obs_diff_details[key] = diff
        
        # Compare derived_sub_indices if available
        if 'derived_sub_indices' in sb3_obs_t and 'derived_sub_indices' in tensor_obs:
            match, diff = compare_tensors(
                sb3_obs_t['derived_sub_indices'], 
                tensor_obs['derived_sub_indices'],
                'derived_sub_indices'
            )
            step_result.derived_sub_indices_match = match
            if not match:
                step_result.obs_diff_details['derived_sub_indices'] = diff
        
        # === Compare Forward Pass ===
        sb3_forward = extract_sb3_forward_pass(sb3_ppo.policy, sb3_obs_t)
        tensor_forward = extract_tensor_forward_pass(tensor_ppo.policy, tensor_obs)
        
        if sb3_forward and tensor_forward:
            for key in ['features', 'latent_pi', 'action_logits']:
                if key in sb3_forward and key in tensor_forward:
                    match, diff = compare_tensors(sb3_forward[key], tensor_forward[key], key)
                    if key == 'features':
                        step_result.features_match = match
                    elif key == 'latent_pi':
                        step_result.latent_pi_match = match
                    elif key == 'action_logits':
                        step_result.logits_match = match
                    if not match:
                        step_result.forward_diff_details[key] = diff
        
        # === Get Actions ===
        with torch.no_grad():
            sb3_action, sb3_vals, sb3_logp = sb3_ppo.policy(sb3_obs_t)
            sb3_action_np = sb3_action.cpu().numpy()
        
        tensor_td_in = TensorDict({
            'observation': tensor_obs,
            'is_init': torch.zeros(n_envs, dtype=torch.bool),
        }, batch_size=[n_envs])
        with torch.no_grad():
            tensor_out = tensor_ppo.policy(tensor_td_in)
        tensor_action_np = tensor_out['action'].numpy()
        
        step_result.sb3_action = sb3_action_np.copy()
        step_result.tensor_action = tensor_action_np.copy()
        step_result.actions_match = np.array_equal(sb3_action_np, tensor_action_np)
        
        # Log probs
        step_result.sb3_logprob = sb3_logp.cpu().numpy().tolist() if sb3_logp is not None else None
        if 'sample_log_prob' in tensor_out.keys():
            step_result.tensor_logprob = tensor_out['sample_log_prob'].numpy().tolist()
        
        # Update total log probs for non-done slots
        if sb3_logp is not None:
            sb3_total_logps[~sb3_all_done] += sb3_logp.cpu().numpy()[~sb3_all_done]
        if step_result.tensor_logprob is not None:
            tensor_total_logps[~tensor_all_done] += np.array(step_result.tensor_logprob)[~tensor_all_done]
        
        result.steps.append(step_result)
        
        # === Print if verbose ===
        if verbose:
            print(f"\n--- Step {step} ---")
            print(f"  Obs match: sub_index={step_result.sub_index_match}, "
                  f"action_mask={step_result.action_mask_match}, "
                  f"derived={step_result.derived_sub_indices_match}")
            print(f"  Forward match: features={step_result.features_match}, "
                  f"latent_pi={step_result.latent_pi_match}, "
                  f"logits={step_result.logits_match}")
            print(f"  Actions: SB3={sb3_action_np}, Tensor={tensor_action_np}, "
                  f"MATCH={step_result.actions_match}")
            
            if not step_result.actions_match:
                print(f"  *** ACTION DIVERGENCE at step {step}! ***")
                # Print detailed comparison for focused slot
                print(f"  Focused slot {slot_idx}:")
                print(f"    SB3 sub_index: {sb3_obs_t['sub_index'][slot_idx]}")
                print(f"    Tensor sub_index: {tensor_obs['sub_index'][slot_idx]}")
                print(f"    SB3 action_mask valid: {sb3_obs_t['action_mask'][slot_idx].sum().item()}")
                print(f"    Tensor action_mask valid: {tensor_obs['action_mask'][slot_idx].sum().item()}")
                
                if sb3_forward and tensor_forward:
                    print(f"    SB3 logits (first 10): {sb3_forward['masked_logits'][slot_idx, :10]}")
                    print(f"    Tensor logits (first 10): {tensor_forward['masked_logits'][slot_idx, :10]}")
                    
                    # Find max logit indices
                    sb3_max = sb3_forward['masked_logits'][slot_idx].argmax().item()
                    tensor_max = tensor_forward['masked_logits'][slot_idx].argmax().item()
                    print(f"    SB3 argmax: {sb3_max}, Tensor argmax: {tensor_max}")
        
        # === Step both environments ===
        sb3_obs, sb3_rew, sb3_done, sb3_trunc, sb3_info = sb3_env.step(sb3_action_np)
        
        step_action = TensorDict({'action': tensor_out['action']}, batch_size=[n_envs])
        tensor_td = tensor_env.step(step_action)
        tensor_done = tensor_td['next']['done'].numpy()
        tensor_obs = tensor_td['next']['observation']
        
        step_result.sb3_done = sb3_done.copy()
        step_result.tensor_done = tensor_done.copy()
        
        # Update done tracking
        sb3_all_done |= sb3_done
        tensor_all_done |= tensor_done
        
        if verbose:
            print(f"  Done: SB3={sb3_done}, Tensor={tensor_done}")
        
        # Stop if all done
        if all(sb3_all_done) and all(tensor_all_done):
            if verbose:
                print(f"\nAll slots done after step {step}")
            break
    
    # Final results
    result.sb3_total_logprobs = sb3_total_logps.tolist()
    result.tensor_total_logprobs = tensor_total_logps.tolist()
    
    return result


def run_full_comparison(
    dataset: str = 'countries_s3',
    n_envs: int = 5,
    query_indices: Optional[List[int]] = None,
    max_steps: int = 20,
    corruption_mode: str = 'tail',
    seed: int = 42,
    verbose: bool = True,
) -> List[QueryComparison]:
    """
    Run step-by-step comparison for multiple queries.
    
    Args:
        dataset: Dataset name
        n_envs: Number of environments (1 pos + N-1 negatives)
        query_indices: Which query indices to test (None = all)
        max_steps: Maximum steps per episode
        corruption_mode: 'head', 'tail', or 'both'
        seed: Random seed
        verbose: Print detailed output
    """
    device = torch.device('cpu')
    
    print("=" * 70)
    print(f"Step-by-Step Parity Comparison")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, corruption: {corruption_mode}")
    print("=" * 70)
    
    # Create aligned environments
    print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs, mode='valid')
    queries = env_data['queries_sb3']
    
    if query_indices is None:
        query_indices = list(range(len(queries)))
    
    # Create environments
    print("Creating SB3 environment...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_eval_env(
        env_data['sb3'], queries, n_envs, seed
    )
    
    print("Creating tensor environment...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_eval_env(
        env_data['tensor'], env_data['queries_tensor'], n_envs, seed
    )
    
    # Create samplers for corruptions
    print("Creating negative samplers...")
    corruption_scheme = [corruption_mode] if corruption_mode != 'both' else ['head', 'tail']
    sb3_sampler = create_sb3_sampler(
        env_data['sb3']['dh'], sb3_im, device, seed, 
        corruption_scheme=corruption_scheme
    )
    
    results = []
    divergent_queries = []
    
    for q_idx in query_indices:
        query = queries[q_idx]
        
        print(f"\n{'='*70}")
        print(f"Query {q_idx}: {query}")
        print(f"{'='*70}")
        
        # Get corruptions
        # Access the underlying sampler method
        if hasattr(sb3_sampler, 'get_negatives_from_states_separate'):
            head_corrs, tail_corrs = sb3_sampler.get_negatives_from_states_separate(
                [[query]], device, num_negs=None  # Get all
            )
            # get_negatives_from_states_separate returns flat list of Term objects
            if corruption_mode == 'head':
                corrs = head_corrs if head_corrs else []
            else:
                corrs = tail_corrs if tail_corrs else []
        else:
            # Fallback - use callable sampler
            corrs = list(sb3_sampler(query, corruption_mode))
        
        # Ensure corrs is a list
        if not isinstance(corrs, list):
            corrs = list(corrs)
        
        # Limit to n_envs - 1 negatives
        if len(corrs) > n_envs - 1:
            corrs = corrs[:n_envs - 1]
        
        # Build goals list: [positive, neg1, neg2, ...]
        all_goals = [query] + list(corrs)
        
        # Pad if needed
        while len(all_goals) < n_envs:
            all_goals.append(query)  # Repeat positive as padding
        
        print(f"Goals ({len(all_goals)} slots):")
        for i, g in enumerate(all_goals):
            label = "POS" if i == 0 else "NEG"
            print(f"  [{i}] {label}: {g}")
        
        # Run comparison
        comparison = run_step_comparison(
            sb3_ppo, sb3_env,
            tensor_ppo, tensor_env,
            all_goals,
            max_steps=max_steps,
            verbose=verbose,
        )
        comparison.query_idx = q_idx
        
        results.append(comparison)
        
        # Check for divergence
        first_div = comparison.first_divergence_step()
        if first_div is not None:
            divergent_queries.append((q_idx, first_div))
            print(f"\n*** DIVERGENCE at query {q_idx}, step {first_div} ***")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total queries tested: {len(results)}")
    print(f"Queries with divergence: {len(divergent_queries)}")
    
    if divergent_queries:
        print("\nDivergent queries:")
        for q_idx, step in divergent_queries:
            print(f"  Query {q_idx}: first divergence at step {step}")
    else:
        print("\n✓ All queries have matching behavior!")
    
    return results


def analyze_divergence(comparison: QueryComparison) -> Dict[str, Any]:
    """Analyze a divergent query comparison to identify root cause."""
    analysis = {
        'query_idx': comparison.query_idx,
        'query': comparison.query_str,
        'first_divergence_step': comparison.first_divergence_step(),
        'observations_match_at_divergence': True,
        'forward_pass_match_at_divergence': True,
        'possible_causes': [],
    }
    
    div_step = comparison.first_divergence_step()
    if div_step is None:
        analysis['conclusion'] = "No divergence found"
        return analysis
    
    step = comparison.steps[div_step]
    
    # Check observations
    if not step.sub_index_match:
        analysis['observations_match_at_divergence'] = False
        analysis['possible_causes'].append("sub_index mismatch")
    if not step.action_mask_match:
        analysis['observations_match_at_divergence'] = False
        analysis['possible_causes'].append("action_mask mismatch")
    if not step.derived_sub_indices_match:
        analysis['observations_match_at_divergence'] = False
        analysis['possible_causes'].append("derived_sub_indices mismatch")
    
    # Check forward pass
    if not step.features_match:
        analysis['forward_pass_match_at_divergence'] = False
        analysis['possible_causes'].append("features mismatch (embedding issue)")
    if not step.latent_pi_match:
        analysis['forward_pass_match_at_divergence'] = False
        analysis['possible_causes'].append("latent_pi mismatch (MLP issue)")
    if not step.logits_match:
        analysis['forward_pass_match_at_divergence'] = False
        analysis['possible_causes'].append("action_logits mismatch")
    
    # Determine conclusion
    if analysis['observations_match_at_divergence'] and analysis['forward_pass_match_at_divergence']:
        analysis['conclusion'] = "Observations and forward pass match but actions differ - likely action sampling issue"
        analysis['possible_causes'].append("Action sampling/argmax implementation difference")
    elif not analysis['observations_match_at_divergence']:
        analysis['conclusion'] = "Observation mismatch caused the divergence"
    else:
        analysis['conclusion'] = "Forward pass mismatch caused the divergence"
    
    analysis['step_details'] = {
        'sb3_action': step.sb3_action.tolist() if step.sb3_action is not None else None,
        'tensor_action': step.tensor_action.tolist() if step.tensor_action is not None else None,
        'obs_diff_details': step.obs_diff_details,
        'forward_diff_details': step.forward_diff_details,
    }
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step-by-Step Parity Debugging")
    parser.add_argument("--dataset", type=str, default="countries_s3",
                       help="Dataset name (default: countries_s3)")
    parser.add_argument("--n-envs", type=int, default=5,
                       help="Number of environments (default: 5)")
    parser.add_argument("--query-idx", type=int, default=None,
                       help="Specific query index to test")
    parser.add_argument("--all-queries", action="store_true",
                       help="Test all queries")
    parser.add_argument("--max-steps", type=int, default=20,
                       help="Maximum steps per episode (default: 20)")
    parser.add_argument("--corruption", type=str, default="tail",
                       choices=["head", "tail", "both"],
                       help="Corruption mode (default: tail)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity")
    parser.add_argument("--analyze", action="store_true",
                       help="Run detailed analysis on divergent queries")
    
    args = parser.parse_args()
    
    # Determine which queries to test
    if args.query_idx is not None:
        query_indices = [args.query_idx]
    elif args.all_queries:
        query_indices = None  # Will test all
    else:
        # Default: test queries known to have issues
        query_indices = [11, 12, 16]  # From previous analysis
    
    # Run comparison
    results = run_full_comparison(
        dataset=args.dataset,
        n_envs=args.n_envs,
        query_indices=query_indices,
        max_steps=args.max_steps,
        corruption_mode=args.corruption,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    # Analyze divergent queries
    if args.analyze:
        print("\n" + "=" * 70)
        print("DETAILED ANALYSIS OF DIVERGENT QUERIES")
        print("=" * 70)
        
        for comp in results:
            if comp.first_divergence_step() is not None:
                analysis = analyze_divergence(comp)
                print(f"\nQuery {analysis['query_idx']}: {analysis['query']}")
                print(f"  First divergence: step {analysis['first_divergence_step']}")
                print(f"  Observations match: {analysis['observations_match_at_divergence']}")
                print(f"  Forward pass match: {analysis['forward_pass_match_at_divergence']}")
                print(f"  Possible causes: {analysis['possible_causes']}")
                print(f"  Conclusion: {analysis['conclusion']}")
                
                if analysis['step_details']:
                    print(f"  SB3 action: {analysis['step_details']['sb3_action']}")
                    print(f"  Tensor action: {analysis['step_details']['tensor_action']}")
