#!/usr/bin/env python
"""
Fast MCTS Trainer - Optimized for speed.

Key optimizations:
1. Reduced MCTS simulations (10 instead of 25)
2. Batched neural network inference where possible
3. Direct policy inference for evaluation (no MCTS tree search)
4. Minimized state cloning
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Setup paths
script_dir = Path(__file__).resolve().parent
kge_dir = script_dir.parent
root_dir = kge_dir.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(kge_dir))

import torch
import torch.nn.functional as F
from torch import Tensor

from kge_experiments.builder import get_default_config, create_env, create_policy
from kge_experiments.mcts.config import MCTSConfig
from kge_experiments.mcts.tree import MCTS, Node, MinMaxStats
from kge_experiments.ppo import PPO


@dataclass
class FastMCTSConfig:
    """Optimized MCTS config for speed."""
    num_simulations: int = 10  # Reduced from 25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    discount: float = 0.99
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature: float = 1.0
    max_episode_steps: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    device: str = "cuda"


class FastMCTSSearch:
    """Optimized MCTS search with minimal overhead."""

    def __init__(self, config: FastMCTSConfig):
        self.config = config
        self.device = torch.device(config.device)

    def search(
        self,
        env: Any,
        state: Any,
        policy: torch.nn.Module,
        obs: Dict[str, Tensor],
        action_mask: Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """Fast MCTS search with reduced overhead."""

        # Get priors and value
        with torch.no_grad():
            logits = policy.get_logits(obs)
            if logits.dim() == 2:
                logits = logits.squeeze(0)
            value = policy.predict_values(obs)
            if isinstance(value, Tensor):
                value = value.item()

        # Mask invalid actions
        if action_mask.dim() == 2:
            action_mask = action_mask.squeeze(0)
        masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        priors = F.softmax(masked_logits, dim=-1)

        valid_actions = action_mask.nonzero(as_tuple=True)[0].tolist()
        if not valid_actions:
            return 0, {"visit_counts": {}, "root_value": 0.0}

        # Initialize root
        root = Node()
        root.expand(valid_actions, priors)

        # Add noise
        if self.config.root_exploration_fraction > 0:
            root.add_exploration_noise(
                self.config.root_dirichlet_alpha,
                self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # Clone state once per simulation
            sim_state = state.clone() if hasattr(state, 'clone') else state
            sim_obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()}

            # Selection: traverse to leaf
            while node.is_expanded():
                action, child = self._select_child(node, min_max_stats)
                if action is None:
                    break
                search_path.append(child)
                node = child

                # Step in environment
                action_tensor = torch.tensor([action], device=self.device)
                sim_obs, sim_state = env.step(sim_state, action_tensor)

                # Check done
                done = sim_state.get("done", torch.zeros(1, dtype=torch.bool))
                if done.bool().any():
                    break

            # Get leaf value
            done = sim_state.get("done", torch.zeros(1, dtype=torch.bool))
            if done.bool().any():
                success = sim_state.get("success", torch.zeros(1, dtype=torch.bool))
                leaf_value = 1.0 if success.bool().any() else -1.0
            else:
                with torch.no_grad():
                    leaf_value = policy.predict_values(sim_obs)
                    if isinstance(leaf_value, Tensor):
                        leaf_value = leaf_value.item()

                # Expand if not terminal
                action_mask_new = sim_obs.get("action_mask", None)
                if action_mask_new is not None and not node.is_expanded():
                    if action_mask_new.dim() == 2:
                        action_mask_new = action_mask_new.squeeze(0)
                    with torch.no_grad():
                        new_logits = policy.get_logits(sim_obs)
                        if new_logits.dim() == 2:
                            new_logits = new_logits.squeeze(0)
                    masked = new_logits.masked_fill(~action_mask_new.bool(), float("-inf"))
                    new_priors = F.softmax(masked, dim=-1)
                    new_valid = action_mask_new.nonzero(as_tuple=True)[0].tolist()
                    if new_valid:
                        node.expand(new_valid, new_priors)

            # Backprop
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += leaf_value
                min_max_stats.update(n.value())
                leaf_value = n.reward + self.config.discount * leaf_value

        # Select action by visit count
        visit_counts = {a: c.visit_count for a, c in root.children.items()}
        if not visit_counts:
            return valid_actions[0], {"visit_counts": {}, "root_value": 0.0}

        if self.config.temperature < 0.01:
            action = max(visit_counts, key=visit_counts.get)
        else:
            actions = list(visit_counts.keys())
            counts = torch.tensor([visit_counts[a] for a in actions], dtype=torch.float32)
            probs = counts ** (1.0 / self.config.temperature)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            action = actions[idx]

        return action, {"visit_counts": visit_counts, "root_value": root.value()}

    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[Optional[int], Optional[Node]]:
        """UCB selection."""
        if not node.children:
            return None, None

        import math
        pb_c = math.log((1 + node.visit_count + self.config.pb_c_base) / self.config.pb_c_base) + self.config.pb_c_init
        sqrt_visit = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0

        best_score = float("-inf")
        best_action, best_child = None, None

        for action, child in node.children.items():
            q = min_max_stats.normalize(child.value()) if child.visit_count > 0 else 0.0
            prior_score = pb_c * child.prior * sqrt_visit / (1 + child.visit_count)
            score = q + prior_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


class FastMCTSTrainer:
    """Fast MCTS trainer optimized for speed."""

    def __init__(self, policy, env, config: FastMCTSConfig, ppo_agent):
        self.policy = policy
        self.env = env
        self.config = config
        self.ppo_agent = ppo_agent  # For fast evaluation
        self.device = torch.device(config.device)

        self.mcts = FastMCTSSearch(config)
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )

        self.num_timesteps = 0
        self.num_episodes = 0

        # Simple experience buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.visit_count_buffer = []
        self.value_buffer = []

    def collect_episode(self) -> Dict[str, float]:
        """Collect single episode with MCTS."""
        self.policy.eval()

        if hasattr(self.env, 'train'):
            self.env.train()

        obs, state = self.env.reset()
        obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()}

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < self.config.max_episode_steps:
            action_mask = obs.get("action_mask")
            if action_mask is None:
                break

            if action_mask.dim() == 2:
                action_mask = action_mask.squeeze(0)

            # MCTS search
            action, stats = self.mcts.search(
                env=self.env,
                state=state,
                policy=self.policy,
                obs=obs,
                action_mask=action_mask,
            )

            # Store experience
            self.obs_buffer.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()})
            self.action_buffer.append(action)
            self.visit_count_buffer.append(stats.get("visit_counts", {}))
            self.value_buffer.append(stats.get("root_value", 0.0))

            # Step
            action_tensor = torch.tensor([action], device=self.device)
            new_obs, new_state = self.env.step(state, action_tensor)

            reward = new_state.get("step_rewards", torch.zeros(1)).item()
            done = new_state.get("done", torch.zeros(1, dtype=torch.bool)).bool().any().item()

            self.reward_buffer.append(reward)
            self.done_buffer.append(done)

            episode_reward += reward
            episode_length += 1

            obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in new_obs.items()}
            state = new_state.clone() if hasattr(new_state, 'clone') else new_state

        self.num_timesteps += episode_length
        self.num_episodes += 1

        return {
            "reward": episode_reward,
            "length": episode_length,
            "success": episode_reward > 0,
        }

    def train_step(self) -> Dict[str, float]:
        """Training step on collected experience."""
        if len(self.obs_buffer) < self.config.batch_size:
            return {"skipped": True}

        self.policy.train()

        # Sample batch
        indices = torch.randperm(len(self.obs_buffer))[:self.config.batch_size]

        # Build tensors
        batch_obs = {}
        for k in self.obs_buffer[0].keys():
            tensors = [self.obs_buffer[i][k] for i in indices]
            if isinstance(tensors[0], Tensor):
                batch_obs[k] = torch.cat([t if t.dim() > 0 else t.unsqueeze(0) for t in tensors], dim=0)

        # Forward
        logits = self.policy.get_logits(batch_obs)
        values = self.policy.predict_values(batch_obs)

        # Compute policy target from visit counts
        batch_size = len(indices)
        num_actions = logits.shape[-1]
        policy_targets = torch.zeros(batch_size, num_actions, device=self.device)

        for i, idx in enumerate(indices):
            vc = self.visit_count_buffer[idx.item()]
            total = sum(vc.values()) if vc else 1
            for a, c in vc.items():
                if a < num_actions:
                    policy_targets[i, a] = c / total

        # Policy loss (cross entropy with visit counts)
        action_mask = batch_obs.get("action_mask", None)
        if action_mask is not None:
            masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        else:
            masked_logits = logits

        log_probs = F.log_softmax(masked_logits, dim=-1)
        log_probs = torch.where(torch.isinf(log_probs), torch.full_like(log_probs, -100), log_probs)
        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

        # Value loss
        value_targets = torch.tensor([self.value_buffer[i.item()] for i in indices], device=self.device)
        value_loss = F.mse_loss(values.flatten(), value_targets)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item(),
        }

    def clear_buffer(self):
        """Clear experience buffer."""
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.visit_count_buffer.clear()
        self.value_buffer.clear()

    def learn(self, total_timesteps: int, episodes_per_iter: int = 10, log_interval: int = 5):
        """Main training loop."""
        print("=" * 70)
        print("Fast MCTS Training")
        print("=" * 70)
        print(f"MCTS simulations: {self.config.num_simulations}")
        print(f"Episodes per iter: {episodes_per_iter}")
        print(f"Total timesteps: {total_timesteps}")
        print("-" * 70, flush=True)

        iteration = 0

        while self.num_timesteps < total_timesteps:
            iteration += 1
            iter_start = time.time()

            # Collect episodes
            rewards = []
            successes = []
            for _ in range(episodes_per_iter):
                stats = self.collect_episode()
                rewards.append(stats["reward"])
                successes.append(stats["success"])

            # Train
            train_metrics = self.train_step()

            iter_time = time.time() - iter_start

            if iteration % log_interval == 0:
                mean_reward = sum(rewards) / len(rewards)
                success_rate = sum(successes) / len(successes)
                loss = train_metrics.get("total_loss", 0.0)
                print(
                    f"[MCTS] Iter {iteration} | "
                    f"Steps: {self.num_timesteps}/{total_timesteps} | "
                    f"Reward: {mean_reward:.3f} | "
                    f"Success: {success_rate:.1%} | "
                    f"Loss: {loss:.4f} | "
                    f"Time: {iter_time:.1f}s",
                    flush=True
                )

            # Clear buffer periodically
            if len(self.obs_buffer) > 5000:
                self.clear_buffer()

        print("-" * 70)
        print(f"Training complete: {self.num_timesteps} timesteps, {self.num_episodes} episodes", flush=True)

    def evaluate(self, queries, sampler, n_corruptions=100, corruption_modes=('head', 'tail'), verbose=False):
        """Evaluate using PPO's fast batched infrastructure."""
        self.policy.eval()
        return self.ppo_agent.evaluate(
            queries=queries,
            sampler=sampler,
            n_corruptions=n_corruptions,
            corruption_modes=corruption_modes,
            verbose=verbose,
        )

    def save(self, path: str):
        """Save model."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
        }, path)
        print(f"Model saved to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3000)
    parser.add_argument("--mcts_sims", type=int, default=10)
    parser.add_argument("--episodes_per_iter", type=int, default=10)
    parser.add_argument("--eval_queries", type=int, default=500)
    parser.add_argument("--n_corruptions", type=int, default=100)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    print("Building environment and policy...", flush=True)

    # Training config with n_envs=1 for MCTS (sequential)
    train_config = get_default_config(
        dataset='family',
        device='cuda',
        n_envs=1,  # MCTS requires single-env
        max_steps=20,
        reward_type=4,
        compile=False,  # Training doesn't need CUDA graphs
    )

    # Evaluation config with larger batch for fast evaluation
    eval_config = get_default_config(
        dataset='family',
        device='cuda',
        n_envs=75,  # Larger batch for fast evaluation
        max_steps=20,
        reward_type=4,
        compile=True,  # Enable CUDA graphs for fast evaluation
    )

    train_env = create_env(train_config)
    eval_env = create_env(eval_config)
    policy = create_policy(train_config, train_env)
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}", flush=True)

    # PPO for evaluation (uses eval_env with larger batch)
    ppo_agent = PPO(policy, eval_env, eval_config, eval_only=True)

    # Fast MCTS config
    mcts_config = FastMCTSConfig(
        num_simulations=args.mcts_sims,
        max_episode_steps=train_config.max_steps,
    )

    trainer = FastMCTSTrainer(policy, train_env, mcts_config, ppo_agent)

    # Train
    print("\nTraining...", flush=True)
    trainer.learn(
        total_timesteps=args.timesteps,
        episodes_per_iter=args.episodes_per_iter,
    )

    # Save
    if args.save_path:
        trainer.save(args.save_path)
    else:
        save_path = str(root_dir / "models" / "mcts_family_fast.pt")
        Path(save_path).parent.mkdir(exist_ok=True)
        trainer.save(save_path)

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation", flush=True)
    print("=" * 70)

    test_queries = eval_config._components['test_queries']
    sampler = eval_config._components['sampler']

    n_test = min(args.eval_queries, len(test_queries))
    test_subset = test_queries[:n_test]

    print(f"Evaluating {n_test} queries with {args.n_corruptions} negatives...", flush=True)

    # Warmup
    _ = trainer.evaluate(test_subset[:5], sampler, 5, ('head',), verbose=False)

    # Full eval
    eval_start = time.time()
    results = trainer.evaluate(
        test_subset, sampler, args.n_corruptions, ('head', 'tail'), verbose=True
    )
    eval_time = time.time() - eval_start

    print("\n" + "=" * 70)
    print("Results", flush=True)
    print("=" * 70)
    print(f"MRR:     {results.get('MRR', 0):.4f}")
    print(f"Hits@1:  {results.get('Hits@1', 0):.4f}")
    print(f"Hits@3:  {results.get('Hits@3', 0):.4f}")
    print(f"Hits@10: {results.get('Hits@10', 0):.4f}")
    print(f"Proven:  {results.get('proven_pos', 0):.1%}")
    print(f"Eval time: {eval_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
