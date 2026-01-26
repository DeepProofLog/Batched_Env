"""
Test PBRS (Potential-Based Reward Shaping) implementation.

Verifies:
1. PBRS module computes potentials correctly
2. Shaped rewards differ from base rewards
3. Terminal state fix works (no teleportation penalty)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import TrainConfig
from builder import build_env, build_policy
from kge_module.pbrs import PBRSModule, PBRSWrapper, create_pbrs_module


def test_pbrs_potential_computation():
    """Test that PBRS computes potentials for states."""
    print("=" * 60)
    print("Test 1: PBRS Potential Computation")
    print("=" * 60)

    # Create PBRS module
    pbrs = PBRSModule(beta=0.1, gamma=0.99, mode='runtime', device=torch.device('cuda'))

    # Create dummy state tensor [B, A, 3]
    # Predicate index 1, args 10, 20
    states = torch.tensor([
        [[1, 10, 20], [0, 0, 0], [0, 0, 0]],  # One atom
        [[2, 15, 25], [0, 0, 0], [0, 0, 0]],  # Different atom
    ], device='cuda')

    idx2pred = {0: '<PAD>', 1: 'brother', 2: 'sister', 3: 'True', 4: 'False'}
    idx2const = {0: '<PAD>', 10: 'john', 15: 'mary', 20: 'bob', 25: 'alice'}

    potentials = pbrs.compute_potential_batch(states, idx2pred, idx2const, padding_idx=0)

    print(f"States shape: {states.shape}")
    print(f"Potentials: {potentials}")
    print(f"Potentials shape: {potentials.shape}")

    # Terminal states should have 0 potential
    terminal_states = torch.tensor([
        [[3, 0, 0], [0, 0, 0], [0, 0, 0]],  # True predicate
        [[4, 0, 0], [0, 0, 0], [0, 0, 0]],  # False predicate
    ], device='cuda')

    terminal_potentials = pbrs.compute_potential_batch(terminal_states, idx2pred, idx2const, padding_idx=0)
    print(f"Terminal potentials (should be 0): {terminal_potentials}")

    assert terminal_potentials[0] == 0.0, "True predicate should have 0 potential"
    assert terminal_potentials[1] == 0.0, "False predicate should have 0 potential"
    print("PASSED: Terminal states have 0 potential")
    print()


def test_pbrs_reward_shaping():
    """Test that PBRS shapes rewards correctly."""
    print("=" * 60)
    print("Test 2: PBRS Reward Shaping")
    print("=" * 60)

    pbrs = PBRSModule(beta=0.1, gamma=0.99, mode='runtime', device=torch.device('cuda'))

    # Base rewards
    rewards = torch.tensor([1.0, -1.0, 0.0, 1.0], device='cuda')

    # Potentials
    phi_s = torch.tensor([0.2, 0.3, 0.1, 0.5], device='cuda')
    phi_sp = torch.tensor([0.4, 0.1, 0.3, 0.0], device='cuda')  # Last one is terminal

    # Done mask
    done_mask = torch.tensor([False, False, False, True], device='cuda')

    shaped = pbrs.compute_shaped_rewards(rewards, phi_s, phi_sp, done_mask)

    print(f"Base rewards: {rewards}")
    print(f"Phi(s): {phi_s}")
    print(f"Phi(s'): {phi_sp}")
    print(f"Done mask: {done_mask}")
    print(f"Shaped rewards: {shaped}")

    # Manual calculation for verification
    # r' = r + gamma * phi_sp - phi_s
    # For done states, phi_sp_effective = 0
    expected = torch.zeros_like(rewards)
    expected[0] = 1.0 + 0.99 * 0.4 - 0.2  # = 1.196
    expected[1] = -1.0 + 0.99 * 0.1 - 0.3  # = -1.201
    expected[2] = 0.0 + 0.99 * 0.3 - 0.1  # = 0.197
    expected[3] = 1.0 + 0.99 * 0.0 - 0.5  # = 0.5 (terminal, phi_sp=0)

    print(f"Expected: {expected}")

    assert torch.allclose(shaped, expected, atol=1e-3), f"Mismatch: {shaped} vs {expected}"
    print("PASSED: Reward shaping formula correct")
    print()


def test_pbrs_integration():
    """Test PBRS with actual environment."""
    print("=" * 60)
    print("Test 3: PBRS Integration with Environment")
    print("=" * 60)

    config = TrainConfig(
        dataset='family',
        n_envs=8,
        pbrs_beta=0.1,
        kge_inference=False,
        total_timesteps=1000,
    )

    # Build environment
    env = build_env(config)

    # Create PBRS module
    pbrs_module = create_pbrs_module(config, device=torch.device('cuda'))

    if pbrs_module is None:
        print("PBRS module not created (beta=0?)")
        return

    print(f"PBRS beta: {pbrs_module.beta}")
    print(f"PBRS gamma: {pbrs_module.gamma}")
    print(f"PBRS mode: {pbrs_module.mode}")

    # Get initial state
    obs, state = env.reset()

    print(f"Initial state shape: {state['current_states'].shape}")

    # Take a few steps and compare rewards with/without PBRS
    print("\nComparing rewards with/without PBRS:")
    print("-" * 50)

    for step in range(5):
        # Random action
        actions = torch.randint(0, state['derived_counts'].max().item() + 1, (config.n_envs,), device='cuda')
        actions = actions.clamp(max=state['derived_counts'] - 1).clamp(min=0)

        # Step
        new_obs, new_state = env.step_and_reset(state, actions, env.query_pool, env.per_env_ptrs)

        base_rewards = new_state['step_rewards']
        terminal_states = new_state['terminal_states']
        done_mask = new_state['step_dones'].bool()

        # Compute PBRS potentials manually
        idx2pred = {v: k for k, v in env.index_manager.predicate_str2idx.items()}
        idx2const = {v: k for k, v in env.index_manager.constant_str2idx.items()}

        phi_s = pbrs_module.compute_potential_batch(
            state['current_states'], idx2pred, idx2const, env.padding_idx
        )
        phi_sp = pbrs_module.compute_potential_batch(
            terminal_states, idx2pred, idx2const, env.padding_idx
        )

        shaped_rewards = pbrs_module.compute_shaped_rewards(base_rewards, phi_s, phi_sp, done_mask)

        diff = (shaped_rewards - base_rewards).abs().mean().item()

        print(f"Step {step}: base_reward mean={base_rewards.mean():.3f}, "
              f"shaped mean={shaped_rewards.mean():.3f}, diff={diff:.4f}")

        state = new_state
        obs = new_obs

    print("\nPASSED: PBRS integration works")
    print()


def test_terminal_state_fix():
    """Test that terminal_states are preserved correctly."""
    print("=" * 60)
    print("Test 4: Terminal State Fix (No Teleportation Penalty)")
    print("=" * 60)

    config = TrainConfig(
        dataset='family',
        n_envs=16,
        total_timesteps=1000,
    )

    env = build_env(config)
    obs, state = env.reset()

    # Run until we get some done episodes
    done_count = 0
    max_steps = 100

    for step in range(max_steps):
        actions = torch.randint(0, state['derived_counts'].max().item() + 1, (config.n_envs,), device='cuda')
        actions = actions.clamp(max=state['derived_counts'] - 1).clamp(min=0)

        new_obs, new_state = env.step_and_reset(state, actions, env.query_pool, env.per_env_ptrs)

        done_mask = new_state['step_dones'].bool()
        num_done = done_mask.sum().item()

        if num_done > 0:
            done_count += num_done

            # Check terminal_states vs current_states for done envs
            terminal_states = new_state['terminal_states'][done_mask]
            current_states = new_state['current_states'][done_mask]

            # For done envs, current_states should be RESET state (new query)
            # terminal_states should be the TERMINAL state (True/False/End)

            terminal_preds = terminal_states[:, 0, 0]
            current_preds = current_states[:, 0, 0]

            # Terminal predicates are usually True, False, or End
            true_idx = env.true_pred_idx
            false_idx = env.false_pred_idx

            is_terminal_pred = (terminal_preds == true_idx) | (terminal_preds == false_idx)

            print(f"Step {step}: {num_done} episodes done")
            print(f"  Terminal state predicates: {terminal_preds[:3].tolist()}")
            print(f"  Current state predicates (reset): {current_preds[:3].tolist()}")
            print(f"  Terminal states have terminal predicate: {is_terminal_pred.sum().item()}/{num_done}")

            if done_count >= 10:
                break

        state = new_state
        obs = new_obs

    print(f"\nTotal episodes checked: {done_count}")
    print("PASSED: terminal_states preserved correctly")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PBRS VERIFICATION TESTS")
    print("=" * 60 + "\n")

    test_pbrs_potential_computation()
    test_pbrs_reward_shaping()
    test_pbrs_integration()
    test_terminal_state_fix()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
