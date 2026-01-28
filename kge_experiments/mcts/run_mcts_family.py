#!/usr/bin/env python
"""
MCTS Training on Family Dataset with PPO-compatible evaluation.

This script:
1. Trains using MCTS for action selection
2. Uses PPO's batched evaluation for proper MRR computation
3. Ensures semantic parity with PPO (same reward type, MRR calculation)
4. Saves model checkpoints

Usage (from MCTS root):
    python kge_experiments/mcts/run_mcts_family.py --timesteps 5000
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Ensure correct import path - must be run from MCTS root
script_dir = Path(__file__).resolve().parent
kge_dir = script_dir.parent
root_dir = kge_dir.parent

# Change to root directory and add to path for proper imports
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(kge_dir))  # For absolute imports within kge_experiments

import torch

from kge_experiments.builder import get_default_config, create_env, create_policy
from kge_experiments.mcts.config import MCTSConfig
from kge_experiments.mcts.tree import MCTS
from kge_experiments.mcts.replay_buffer import MCTSReplayBuffer, MCTSTrajectory, MCTSTransition
from kge_experiments.ppo import PPO


def create_mcts_config_from_base(config, **overrides):
    """Create MCTSConfig from base config with overrides."""
    return MCTSConfig(
        num_simulations=overrides.get('mcts_simulations', 25),
        pb_c_base=19652.0,
        pb_c_init=1.25,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        add_exploration_noise=True,
        value_nstep=5,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_decay_steps=50000,
        temperature_schedule='linear',
        hidden_dim=256,
        num_layers=4,
        unroll_steps=5,
        replay_buffer_size=100000,
        min_buffer_size=500,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_grad_norm=5.0,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        episodes_per_iteration=20,
        max_episode_steps=config.max_steps,
        device=str(config.device),
        compile=True,  # Enable CUDA graph compilation for optimized MCTS
        log_interval=10,
        verbose=True,
    )


class MCTSTrainerWithPPOEval:
    """MCTS trainer that uses PPO's evaluation infrastructure."""

    def __init__(self, policy, env, mcts_config, ppo_agent):
        self.policy = policy
        self.env = env
        self.config = mcts_config
        self.ppo_agent = ppo_agent  # For evaluation
        self.device = torch.device(mcts_config.device)

        # MCTS components
        self.mcts = MCTS(mcts_config)
        self.replay_buffer = MCTSReplayBuffer(
            max_size=mcts_config.replay_buffer_size,
            discount=mcts_config.discount,
            n_step=mcts_config.value_nstep,
            device=self.device,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=mcts_config.learning_rate,
            weight_decay=mcts_config.weight_decay,
        )

        # State
        self.num_timesteps = 0
        self.num_episodes = 0
        self.iteration = 0

    def collect_episode(self, add_noise=True):
        """Collect single episode using MCTS."""
        self.policy.eval()
        trajectory = MCTSTrajectory()

        # Ensure training mode
        if hasattr(self.env, 'train'):
            self.env.train()

        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, state = reset_result
        else:
            obs = reset_result
            state = getattr(self.env, "_state", None)

        obs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

        done = False
        step = 0

        while not done and step < self.config.max_episode_steps:
            action_mask = obs.get("action_mask", None)
            if action_mask is None and state is not None:
                action_mask = state.get("derived_counts", None)
            if action_mask is None:
                break

            if action_mask.dim() == 2:
                action_mask = action_mask.squeeze(0)

            # MCTS search
            temperature = self.config.get_temperature(self.num_timesteps)
            use_noise = add_noise and temperature > 0.1

            action, search_stats = self.mcts.search(
                env=self.env,
                env_state=state,
                networks=self.policy,
                obs=obs,
                action_mask=action_mask,
                add_noise=use_noise,
            )

            trans = MCTSTransition(
                obs={k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()},
                action=action,
                reward=0.0,
                done=False,
                visit_counts=search_stats.get("visit_counts", {}),
                root_value=search_stats.get("root_value", 0.0),
            )

            action_tensor = torch.tensor([action], device=self.device)
            step_result = self.env.step(state, action_tensor)

            if isinstance(step_result, tuple):
                new_obs, new_state = step_result
            else:
                new_obs = step_result
                new_state = getattr(self.env, "_state", state)

            reward_t = new_state.get("step_rewards", torch.zeros(1))
            done_t = new_state.get("done", torch.zeros(1, dtype=torch.bool))

            reward = reward_t.item() if isinstance(reward_t, torch.Tensor) and reward_t.numel() == 1 else float(reward_t[0])
            done = done_t.item() if isinstance(done_t, torch.Tensor) and done_t.numel() == 1 else bool(done_t[0])

            trans.reward = reward
            trans.done = done
            trajectory.add(trans)

            obs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in new_obs.items()}
            state = new_state.clone() if hasattr(new_state, 'clone') else new_state
            step += 1

        return trajectory

    def train_step(self):
        """Single training step on sampled batch."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {"skipped": True}

        self.policy.train()

        batch = self.replay_buffer.sample_batch(
            batch_size=self.config.batch_size,
            sequence_length=1,
        )

        obs = batch["obs"]
        policy_targets = batch["policy_targets"]
        value_targets = batch["value_targets"]

        # Forward
        logits = self.policy.get_logits(obs)
        values = self.policy.predict_values(obs)

        # Handle shape mismatch
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            if logits.shape[-1] != policy_targets.shape[-1]:
                if policy_targets.shape[-1] < logits.shape[-1]:
                    padding = torch.zeros(
                        policy_targets.shape[0],
                        logits.shape[-1] - policy_targets.shape[-1],
                        device=self.device,
                    )
                    policy_targets = torch.cat([policy_targets, padding], dim=-1)
                else:
                    policy_targets = policy_targets[:, :logits.shape[-1]]

            masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        else:
            masked_logits = logits

        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
        log_probs = torch.where(
            torch.isinf(log_probs),
            torch.full_like(log_probs, -100.0),
            log_probs
        )

        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
        if torch.isnan(policy_loss):
            policy_loss = torch.tensor(0.0, device=self.device)

        value_loss = torch.nn.functional.mse_loss(values.flatten(), value_targets)
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def learn(self, total_timesteps, eval_fn=None, eval_freq=0):
        """Main training loop."""
        self.num_timesteps = 0
        self.iteration = 0
        best_mrr = 0.0
        best_state = None

        print("=" * 70)
        print("MCTS Training on Family Dataset - Full Run")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Dataset: family")
        print(f"  MCTS simulations: {self.config.num_simulations}")
        print(f"  Episodes per iteration: {self.config.episodes_per_iteration}")
        print(f"  Total timesteps: {total_timesteps}")
        print()

        while self.num_timesteps < total_timesteps:
            self.iteration += 1
            iter_start = time.time()

            # Collect episodes
            rewards = []
            lengths = []
            successes = []

            for _ in range(self.config.episodes_per_iteration):
                traj = self.collect_episode(add_noise=True)
                self.replay_buffer.add_trajectory(traj)
                self.num_episodes += 1
                rewards.append(traj.total_reward)
                lengths.append(traj.length)
                if traj.transitions:
                    successes.append(traj.transitions[-1].reward > 0)

            self.num_timesteps += sum(lengths)

            # Train
            train_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            n_train = max(1, self.config.episodes_per_iteration // 4)
            for _ in range(n_train):
                step_m = self.train_step()
                if "skipped" not in step_m:
                    for k, v in step_m.items():
                        train_metrics[k] += v / n_train

            iter_time = time.time() - iter_start

            if self.iteration % self.config.log_interval == 0:
                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                success_rate = sum(successes) / len(successes) if successes else 0.0
                print(
                    f"[MCTS] Iter {self.iteration} | "
                    f"Timesteps: {self.num_timesteps}/{total_timesteps} | "
                    f"Episodes: {self.num_episodes} | "
                    f"Mean Reward: {mean_reward:.3f} | "
                    f"Success: {success_rate:.1%} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"Time: {iter_time:.2f}s"
                )

            # Evaluation
            if eval_fn and eval_freq > 0 and self.iteration % eval_freq == 0:
                eval_results = eval_fn()
                mrr = eval_results.get('MRR', 0.0)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_state = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
                print(f"  [Eval] MRR: {mrr:.4f} | Best: {best_mrr:.4f}")

        # Restore best if available
        if best_state is not None:
            print(f"\nRestoring best model with MRR: {best_mrr:.4f}")
            self.policy.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return {"best_mrr": best_mrr}

    def evaluate(self, queries, sampler, n_corruptions=100, corruption_modes=('head', 'tail'), verbose=False):
        """Evaluate using PPO's infrastructure for proper MRR."""
        self.policy.eval()
        return self.ppo_agent.evaluate(
            queries=queries,
            sampler=sampler,
            n_corruptions=n_corruptions,
            corruption_modes=corruption_modes,
            verbose=verbose,
        )

    def save(self, path):
        """Save checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "iteration": self.iteration,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.num_timesteps = ckpt.get("num_timesteps", 0)
        self.num_episodes = ckpt.get("num_episodes", 0)
        self.iteration = ckpt.get("iteration", 0)
        print(f"Model loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description="MCTS Training on Family Dataset")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total training timesteps")
    parser.add_argument("--mcts_simulations", type=int, default=25, help="MCTS simulations per action")
    parser.add_argument("--episodes_per_iter", type=int, default=20, help="Episodes per iteration")
    parser.add_argument("--eval_queries", type=int, default=500, help="Number of test queries for evaluation")
    parser.add_argument("--n_corruptions", type=int, default=100, help="Number of negative samples")
    parser.add_argument("--eval_freq", type=int, default=0, help="Evaluation frequency (0=final only)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save model")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Create config
    config = get_default_config(
        dataset='family',
        device=args.device,
        n_envs=1,  # Single env for MCTS (sequential search)
        max_steps=20,
        reward_type=4,  # Same as PPO default
        compile=True,  # Enable CUDA graph compilation for optimized MCTS
        verbose=True,
    )

    print("Building environment and policy...")

    # Create environment
    env = create_env(config)

    # Create PPO-compatible policy (ActorCriticPolicy)
    policy = create_policy(config, env)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create PPO agent for evaluation (shares policy)
    # We use PPO's evaluate() which has proper batched ranking
    ppo_agent = PPO(
        policy,
        env,
        config,
        eval_only=True,  # No training buffers needed
    )

    # Get test queries
    test_queries = config._components['test_queries']
    sampler = config._components['sampler']

    n_test = min(args.eval_queries, len(test_queries))
    test_queries_subset = test_queries[:n_test]
    print(f"\nTest queries: {len(test_queries_subset)}")

    # Create MCTS config
    mcts_config = create_mcts_config_from_base(
        config,
        mcts_simulations=args.mcts_simulations,
    )
    mcts_config.episodes_per_iteration = args.episodes_per_iter

    # Create trainer
    trainer = MCTSTrainerWithPPOEval(policy, env, mcts_config, ppo_agent)

    # Load model if specified
    if args.load_path:
        trainer.load(args.load_path)

    # Evaluation function
    def eval_fn():
        return trainer.evaluate(
            queries=test_queries_subset,
            sampler=sampler,
            n_corruptions=args.n_corruptions,
            corruption_modes=('head', 'tail'),
            verbose=False,
        )

    # Train
    print("\nTraining...")
    print("-" * 70)
    results = trainer.learn(
        total_timesteps=args.timesteps,
        eval_fn=eval_fn if args.eval_freq > 0 else None,
        eval_freq=args.eval_freq,
    )

    # Save model
    if args.save_path:
        trainer.save(args.save_path)
    else:
        # Default save path
        default_path = Path(__file__).parent.parent.parent / "models" / "mcts_family.pt"
        default_path.parent.mkdir(exist_ok=True)
        trainer.save(str(default_path))

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    # Warmup
    print("Warmup...")
    _ = trainer.evaluate(
        queries=test_queries_subset[:5],
        sampler=sampler,
        n_corruptions=5,
        corruption_modes=('head',),
        verbose=False,
    )

    # Full evaluation
    print(f"Evaluating {len(test_queries_subset)} queries with {args.n_corruptions} negatives...")
    eval_start = time.time()
    final_results = trainer.evaluate(
        queries=test_queries_subset,
        sampler=sampler,
        n_corruptions=args.n_corruptions,
        corruption_modes=('head', 'tail'),
        verbose=True,
    )
    eval_time = time.time() - eval_start

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"MRR:     {final_results.get('MRR', 0):.4f}")
    print(f"Hits@1:  {final_results.get('Hits@1', 0):.4f}")
    print(f"Hits@3:  {final_results.get('Hits@3', 0):.4f}")
    print(f"Hits@10: {final_results.get('Hits@10', 0):.4f}")
    print(f"Proven pos: {final_results.get('proven_pos', 0):.1%}")
    print(f"Eval time: {eval_time:.2f}s")


if __name__ == "__main__":
    main()
