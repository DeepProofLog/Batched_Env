"""
PPO Agent

This module provides the main PPO agent class that coordinates training.

Performance Note:
    Uses fast evaluation with environment reuse by default (model_eval_fast).
    To revert to the original slower evaluation, change the import in _run_evaluation:
        from model_eval import eval_corruptions_torchrl  # Slow (creates fresh envs)
    to:
        from model_eval_fast import eval_corruptions_fast  # Fast (reuses envs)
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List
from tensordict import TensorDict
import math
import warnings

import torch
import torch.nn as nn
from .ppo_rollout_custom import CustomRolloutCollector as RolloutCollector
# from .ppo_rollout_sync import RolloutCollector
from torchrl.objectives import ClipPPOLoss as PPOLossCls
from torchrl.objectives.value import GAE

class PPOAgent:
    """
    PPO Agent for Neural-guided Grounding.
    
    This class coordinates the training process, managing rollout collection,
    learning, and evaluation.
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_env: Any,
        eval_env: Any,
        sampler: Any,
        data_handler: Any,
        index_manager: Any = None,  # Add index_manager parameter
        args: Any = None,
        n_envs: int = 128,
        n_steps: int = 128,
        n_epochs: int = 10,
        batch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device('cpu'),
        model_save_path: Optional[Path] = None,
        eval_best_metric: str = "mrr_mean",
        verbose_cb: bool = False,
        verbose: bool = False,
        debug_mode: bool = False,
        min_multiaction_ratio: float = 0.05,
    ):
        """
        Initialize the PPO agent.
        
        Args:
            actor: Actor network
            critic: Critic network
            optimizer: Optimizer
            train_env: Training environment
            eval_env: Evaluation environment
            sampler: Negative sampler
            data_handler: Data handler
            args: Training arguments (optional, for accessing eval parameters)
            n_envs: Number of parallel environments
            n_steps: Steps per rollout
            n_epochs: Optimization epochs per update
            batch_size: Mini-batch size
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Max gradient norm
            device: Device to use
            model_save_path: Path to save models
            eval_best_metric: Metric to use for best model selection
            verbose_cb: If True, print debug information during callback collection
            debug_mode: When True, emit verbose rollout/optimization diagnostics
        """
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.train_env = train_env
        self.eval_env = eval_env
        self.sampler = sampler
        self.data_handler = data_handler
        self.index_manager = index_manager  # Store index_manager
        self.args = args
        self.verbose_cb = verbose_cb
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.min_multiaction_ratio = float(min_multiaction_ratio)
        
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.model_save_path = model_save_path
        self.eval_best_metric = eval_best_metric

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Tracking
        self.best_eval_metric = float('-inf')
        self.best_model_path = None
        self.global_step = 0
        
        # Persistent rollout collector (created on first use)
        self._rollout_collector: Optional[RolloutCollector] = None

    def _debug_print_rollout_stats(self, experiences: List[TensorDict], stats: Dict[str, Any]) -> None:
        """Emit quick diagnostics about the most recent rollout."""
        try:
            rewards = torch.stack(
                [td["next"]["reward"].detach().view(-1) for td in experiences],
                dim=0,
            ).cpu()
            dones = torch.stack(
                [td["next"]["done"].detach().view(-1) for td in experiences],
                dim=0,
            ).cpu()
        except KeyError:
            print("[debug] Rollout stats unavailable (missing reward/done)")
            return

        reward_flat = rewards.view(-1)
        done_flat = dones.view(-1)
        unique_rewards = torch.unique(reward_flat)
        print(
            "    [debug] Rewards μ={:.3f} σ={:.3f} min={:.1f} max={:.1f} unique={}".format(
                reward_flat.mean().item(),
                reward_flat.std(unbiased=False).item(),
                reward_flat.min().item(),
                reward_flat.max().item(),
                [float(x) for x in unique_rewards],
            )
        )
        print(
            f"    [debug] Done ratio={done_flat.float().mean().item():.3f} "
            f"({int(done_flat.sum().item())}/{done_flat.numel()})"
        )

        try:
            mask_counts = torch.stack(
                [td["action_mask"].sum(dim=-1).detach().cpu() for td in experiences],
                dim=0,
            )
            print(
                "    [debug] Action mask valid counts in batch: min={} max={} mean={:.2f}".format(
                    int(mask_counts.min().item()),
                    int(mask_counts.max().item()),
                    mask_counts.float().mean().item(),
                )
            )
        except KeyError:
            pass

        if hasattr(self.train_env, "derived_states_counts"):
            counts = self.train_env.derived_states_counts[: min(4, self.n_envs)]
            try:
                counts_cpu = counts.detach().view(-1).tolist()
                print(f"    [debug] Derived states count (first envs): {counts_cpu}")
            except Exception:
                pass

        if stats.get("episode_info"):
            last_episode = stats["episode_info"][-1]
            ep_return = last_episode.get("r", None)
            ep_len = last_episode.get("l", None)
            print(
                f"    [debug] Last episode stats: return={ep_return} length={ep_len} info_keys={list(last_episode.keys())}"
            )

    def _grad_norm(self, params: List[nn.Parameter]) -> float:
        """Compute global L2 norm for a parameter collection."""
        sq_sum = 0.0
        for p in params:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            sq_sum += float(param_norm.item() ** 2)
        return math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0

    def _validate_action_space(self, flat_td: TensorDict) -> Dict[str, float]:
        """Ensure the rollout exposes more than one action for a healthy fraction of states."""
        action_mask = flat_td.get("action_mask", None)
        if action_mask is None:
            raise RuntimeError("Rollout data is missing 'action_mask'; cannot compute policy logits reliably.")

        valid_counts = action_mask.sum(dim=-1)
        if "next" in flat_td.keys() and isinstance(flat_td["next"], TensorDict):
            next_done = flat_td["next"].get("done", None)
            if next_done is not None:
                active_mask = ~next_done.squeeze(-1).bool()
            else:
                active_mask = torch.ones_like(valid_counts, dtype=torch.bool)
        else:
            active_mask = torch.ones_like(valid_counts, dtype=torch.bool)

        active_counts = valid_counts[active_mask]
        if active_counts.numel() == 0:
            return {"active_samples": 0, "multi_action_ratio": 0.0, "single_action_ratio": 0.0}

        zero_ratio = (active_counts == 0).float().mean().item()
        if zero_ratio > 0:
            raise RuntimeError(
                f"Encountered {zero_ratio*100:.1f}% states with zero valid actions. "
                "Check the environment's derived-state generator."
            )

        single_ratio = (active_counts == 1).float().mean().item()
        multi_ratio = max(0.0, 1.0 - single_ratio)
        if multi_ratio < self.min_multiaction_ratio:
            warnings.warn(
                "Only {:.1f}% of active states expose more than one valid action "
                "(threshold {:.1f}%). Consider enabling end-of-proof actions or reviewing skip_unary_actions."
                .format(multi_ratio * 100.0, self.min_multiaction_ratio * 100.0)
            )

        return {
            "active_samples": int(active_counts.numel()),
            "multi_action_ratio": multi_ratio,
            "single_action_ratio": single_ratio,
        }

    def _debug_print_optimizer_stats(self, metrics: Dict[str, float]) -> None:
        """Print optimizer diagnostics for the latest gradient step."""
        print(
            "    [debug] Optimizer: policy_loss={pol:.4f} value_loss={val:.4f} "
            "entropy={ent:.4f} approx_kl={kl:.4f} clip_frac={clip:.3f} "
            "explained_var={ev:.3f}".format(
                pol=metrics.get("policy_loss", 0.0),
                val=metrics.get("value_loss", 0.0),
                ent=metrics.get("entropy", 0.0),
                kl=metrics.get("approx_kl", 0.0),
                clip=metrics.get("clip_fraction", 0.0),
                ev=metrics.get("explained_variance", 0.0),
            )
        )
        if "grad_actor_norm" in metrics:
            print(
                "    [debug] Grad norms: actor={actor:.4f} critic={critic:.4f} total_pre_clip={total:.4f}".format(
                    actor=metrics.get("grad_actor_norm", 0.0),
                    critic=metrics.get("grad_critic_norm", 0.0),
                    total=metrics.get("grad_total_norm", 0.0),
                )
            )

    
    
    def compute_advantages(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE(γ, λ) using **TorchRL's GAE** on a single stacked TensorDict.

        Returns
        -------
        advantages : Tensor shaped [n_steps, n_envs]
        returns    : Tensor shaped [n_steps, n_envs] (a.k.a. value targets)
        """
        # Stack to [T, B]
        batch_td_time = torch.stack(experiences, dim=0)
        
        # Move to device for computation
        batch_td_time = batch_td_time.to(self.device)

        # Ensure value predictions exist for both current and next observations
        with torch.no_grad():
            # value at current state
            if "state_value" not in batch_td_time.keys():
                self.critic(batch_td_time)
            # value at next state
            nxt = batch_td_time.get("next")
            if "state_value" not in nxt.keys():
                self.critic(nxt)

            # --- Treat time limits as NON-terminal for GAE ---
            # If both keys exist, compute advantages using TERMINATION only,
            # so we bootstrap across time-limit truncations.
            if "terminated" in nxt.keys():
                done_for_gae = nxt.get("terminated")
                # ensure shape is [T, B] not [T, B, 1]
                if done_for_gae.ndim > 2 or (done_for_gae.ndim == 2 and done_for_gae.shape[-1] == 1):
                    done_for_gae = done_for_gae.squeeze(-1)
                nxt.set("done", done_for_gae)

            # Ensure all tensors have consistent shapes (squeeze extra dimensions from reward/done/value)
            # GAE expects all tensors to have the same shape [T, B]
            if "state_value" in nxt.keys():
                value = nxt["state_value"]
                if value.ndim > 2:  # Should be [T, B] but might be [T, B, 1]
                    nxt.set("state_value", value.squeeze(-1))
                elif value.ndim == 2 and value.shape[-1] == 1:  # [T*B, 1]
                    nxt.set("state_value", value.squeeze(-1))
            
            if "reward" in nxt.keys():
                reward = nxt["reward"]
                if reward.ndim > 2:  # Should be [T, B] but might be [T, B, 1]
                    nxt.set("reward", reward.squeeze(-1))
                elif reward.ndim == 2 and reward.shape[-1] == 1:  # [T*B, 1]
                    nxt.set("reward", reward.squeeze(-1))
            
            if "done" in nxt.keys():
                done = nxt["done"]
                if done.ndim > 2:  # Should be [T, B] but might be [T, B, 1]
                    nxt.set("done", done.squeeze(-1))
                elif done.ndim == 2 and done.shape[-1] == 1:  # [T*B, 1]
                    nxt.set("done", done.squeeze(-1))
            
            # Also ensure state_value has the right shape
            if "state_value" in batch_td_time.keys():
                state_value = batch_td_time["state_value"]
                if state_value.ndim > 2:
                    batch_td_time.set("state_value", state_value.squeeze(-1))
            
            if "state_value" in nxt.keys():
                next_value = nxt["state_value"]
                if next_value.ndim > 2:
                    nxt.set("state_value", next_value.squeeze(-1))
            
            # Also squeeze terminated if it exists
            if "terminated" in nxt.keys():
                terminated = nxt["terminated"]
                if terminated.ndim > 2:
                    nxt.set("terminated", terminated.squeeze(-1))

            # Permute tensordict from [T, B] to [B, T] for GAE (GAE expects batch first)
            batch_td_time = batch_td_time.permute(1, 0)


            # Initialize GAE - value_network=None means we already computed values
            # Uses default keys: "state_value", "advantage", "value_target"
            # Default time_dim=-2 works for [B, T] layout
            gae = GAE(
                gamma=self.gamma,
                lmbda=self.gae_lambda,
                value_network=None,
            )
            batch_td_time = gae(batch_td_time)
            
            # Permute back to [T, B]
            batch_td_time = batch_td_time.permute(1, 0)

        advantages = batch_td_time.get("advantage")
        value_targets = batch_td_time.get("value_target")

        # Reshape to [T, B] if they came out flattened
        if advantages.dim() == 1:
            advantages = advantages.view(n_steps, n_envs)
        if value_targets.dim() == 1:
            value_targets = value_targets.view(n_steps, n_envs)

        return advantages, value_targets

    def learn(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
        metrics_callback: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Optimize with **TorchRL's PPOLoss** on in-place TensorDict minibatches."""

        # Stack to [T, B]
        batch_td_time = torch.stack(experiences, dim=0)

        # Compute advantages (and value targets) with TorchRL GAE
        advantages, value_targets = self.compute_advantages(experiences, n_steps, n_envs)

        # Attach to the stacked TD then flatten to [T*B]
        batch_td_time.set("advantage", advantages)
        batch_td_time.set("value_target", value_targets)
        flat_td = batch_td_time.reshape(n_steps * n_envs)
        
        # Move to device for optimization (data might be on CPU from collector)
        flat_td = flat_td.to(self.device)
        
        # Ensure log-prob and action keys have shapes expected by TorchRL.
        if "sample_log_prob" in flat_td.keys():
            log_prob = flat_td.get("sample_log_prob")
            if log_prob.dim() > 1:
                log_prob = log_prob.squeeze(-1)
            flat_td.set("sample_log_prob", log_prob)
        
        if "action" in flat_td.keys():
            action = flat_td.get("action")
            if action.dim() > 1:
                if action.shape[-1] == 1:
                    action = action.squeeze(-1)
                else:
                    action = action.argmax(dim=-1)
            flat_td.set("action", action.to(torch.long))


        action_space_stats = self._validate_action_space(flat_td)

        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        all_params = actor_params + critic_params
        grad_actor_sum = 0.0
        grad_critic_sum = 0.0
        grad_total_sum = 0.0
        grad_measurements = 0

        loss_module = PPOLossCls(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_range,
            entropy_coeff=self.ent_coef,
            normalize_advantage=True,  # <-- CHANGED: Let TorchRL handle normalization
            critic_coeff=self.value_coef,
        )
        # Key mapping for compatibility across TorchRL versions
        if hasattr(loss_module, "set_keys"):
            try:
                loss_module.set_keys(
                    action="action",
                    sample_log_prob="sample_log_prob",
                    value="state_value",
                    advantage="advantage",
                    value_target="value_target",
                )
            except (TypeError, KeyError):
                # Some versions only accept a subset (action, sample_log_prob)
                try:
                    loss_module.set_keys(action="action", sample_log_prob="sample_log_prob")
                except (TypeError, KeyError):
                    pass  # Use defaults

        # Optimization
        flat_bs = n_steps * n_envs
        num_minibatches = max(1, flat_bs // int(self.batch_size))
        indices = torch.arange(flat_bs, device=self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.n_epochs):
            perm = indices[torch.randperm(flat_bs, device=self.device)]
            
            # Reset per-epoch accumulators
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_approx_kl = 0.0
            epoch_clip_fraction = 0.0
            epoch_updates = 0
            
            for mb in range(num_minibatches):
                mb_idx = perm[mb * int(self.batch_size):(mb + 1) * int(self.batch_size)]
                mb_td = flat_td[mb_idx]

                # Compute PPO loss terms using TorchRL's ClipPPOLoss
                loss_out = loss_module(mb_td)

                # Robust extraction across TorchRL versions
                policy_loss = (
                    loss_out.get("loss_objective", None)
                    or loss_out.get("loss_policy", None)
                    or loss_out.get("loss_actor", None)
                    or loss_out.get("loss", None)
                )
                if policy_loss is None:
                    assert False, "Could not find policy loss in PPOLoss output"

                value_loss = loss_out.get("loss_critic", torch.tensor(0.0, device=self.device))
                
                entropy = loss_out.get("entropy", None)
                if entropy is None:
                    loss_entropy = loss_out.get("loss_entropy", None)
                    if loss_entropy is not None:
                        entropy = -loss_entropy
                if entropy is None:
                    raise RuntimeError(
                        "ClipPPOLoss did not return an entropy term. "
                        "Please update TorchRL or ensure entropy_coeff > 0."
                    )
                entropy = entropy.mean()
                if torch.isnan(entropy) or torch.isinf(entropy):
                    raise RuntimeError("Entropy from PPOLoss is NaN/Inf; check logits/action masking.")
                if "action_mask" in mb_td.keys():
                    mask = mb_td.get("action_mask")
                    zero_valid = (mask.sum(dim=-1) == 0)
                    if zero_valid.any():
                        raise RuntimeError(
                            f"Action mask supplied zero valid actions for {int(zero_valid.sum().item())} environments."
                        )
                
                if num_updates < 3 and self.verbose: # Modified debug print
                    print(
                        "[AGENT]  DEBUG loss:"
                        f" epoch={epoch} mb={mb} pg={policy_loss.item():.6f}"
                        f" v={(self.value_coef * value_loss.item()):.6f}"
                        f" ent={entropy.item():.6e}"
                    )

                # Build total loss (value and entropy terms already scaled in some versions;
                # we scale explicitly here for consistency).
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.debug_mode:
                    actor_grad_norm = self._grad_norm(actor_params)
                    critic_grad_norm = self._grad_norm(critic_params)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    all_params,
                    max_norm=self.max_grad_norm
                )
                if self.debug_mode:
                    grad_actor_sum += actor_grad_norm
                    grad_critic_sum += critic_grad_norm
                    grad_total_sum += float(total_norm if isinstance(total_norm, torch.Tensor) else total_norm)
                    grad_measurements += 1
                self.optimizer.step()

                # ----- Metrics (approx_kl & clip_fraction) -----
                approx_kl = loss_out.get("approx_kl", None)
                clip_fraction = loss_out.get("clip_fraction", None)

                if (approx_kl is None) or (clip_fraction is None):
                    # recompute cheaply from logits to keep metrics informative
                    with torch.no_grad():
                        mb_logits_td = mb_td.clone(False)
                        # Run the actor's underlying net without resampling
                        try:
                            # Get the underlying TensorDictModule that computes logits
                            # ProbabilisticActor wraps a TensorDictModule in self.module[0]
                            if hasattr(self.actor, 'module') and len(self.actor.module) > 0:
                                self.actor.module[0](mb_logits_td)  # writes 'logits'
                            else:
                                # Fallback: extract logits key if already present
                                pass
                            
                            from torch.distributions import Categorical
                            logits = mb_logits_td.get("logits")
                            dist = Categorical(logits=logits)
                        except Exception as e:
                            # Fallback: compute logits/value via the shared model, if exposed
                            # This branch is rarely needed.
                            from torch.distributions import Categorical
                            # assume logits already in td (collector path)
                            logits = mb_logits_td.get("logits")
                            # Make sure it's a plain tensor without distribution metadata
                            if hasattr(logits, '_unimplemented'):
                                logits = logits.clone()
                            dist = Categorical(logits=logits)

                        approx_action = mb_td.get("action")
                        if approx_action.dim() > 1:
                            approx_action = approx_action.squeeze(-1)
                        approx_action = approx_action.to(torch.long)
                        new_log_prob = dist.log_prob(approx_action)
                        old_log_prob = mb_td.get("sample_log_prob")
                        if old_log_prob.dim() > 1:
                            old_log_prob = old_log_prob.squeeze(-1)
                        log_ratio = new_log_prob - old_log_prob
                        ratio = torch.exp(log_ratio)
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fraction = (ratio.sub(1).abs() > self.clip_range).float().mean()
                        if not torch.isfinite(approx_kl):
                            approx_kl = torch.tensor(0.0, device=self.device)
                        if not torch.isfinite(clip_fraction):
                            clip_fraction = torch.tensor(0.0, device=self.device)

                # Accumulate (both for total and per-epoch)
                # PPO policy loss can be positive or negative depending on the advantage
                pl_val = float(policy_loss.detach().cpu().item())
                vl_val = float(value_loss.detach().cpu().item())
                ent_val = float(entropy.detach().cpu().item())
                
                epoch_policy_loss += pl_val
                epoch_value_loss += vl_val
                epoch_entropy += ent_val
                epoch_updates += 1

                if epoch == 0 and mb < 3 and self.verbose:
                    print(f"[AGENT]  Epoch {epoch}, Minibatch {mb} loss: pg={pl_val:.6f}, v={vl_val:.6f}, ent={ent_val:.6f}")
                
                # Track raw policy loss (without abs) for both total and epoch metrics
                total_policy_loss += pl_val
                total_value_loss += vl_val
                total_entropy += ent_val
                total_approx_kl += float(approx_kl.detach().cpu().item())
                total_clip_fraction += float(clip_fraction.detach().cpu().item())
                num_updates += 1

            # Per-epoch callback
            if metrics_callback is not None and epoch_updates > 0:
                avg_pl = epoch_policy_loss / epoch_updates
                if epoch == 0 and self.verbose:
                    print(f"[AGENT]  DEBUG epoch {epoch+1}: sum={epoch_policy_loss:.6f}, count={epoch_updates}, avg={avg_pl:.6f}")
                metrics_callback.on_training_epoch(
                    epoch=epoch + 1,
                    n_epochs=self.n_epochs,
                    policy_loss=avg_pl,
                    value_loss=epoch_value_loss / epoch_updates,
                    entropy=epoch_entropy / epoch_updates,
                )

        # Aggregate metrics
        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_entropy = total_entropy / max(1, num_updates)
        avg_approx_kl = total_approx_kl / max(1, num_updates)
        avg_clip_fraction = total_clip_fraction / max(1, num_updates)

        # Explained variance
        with torch.no_grad():
            vpred = flat_td.get("state_value").flatten()
            vtarget = flat_td.get("value_target").flatten()
            var_vtarget = torch.var(vtarget)
            var_residual = torch.var(vtarget - vpred)
            explained_var = (1 - (var_residual / (var_vtarget + 1e-8))).item()

        metrics = {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction,
            "explained_variance": explained_var,
            "n_updates": int(num_updates),
        }
        if self.debug_mode and grad_measurements > 0:
            metrics["grad_actor_norm"] = grad_actor_sum / grad_measurements
            metrics["grad_critic_norm"] = grad_critic_sum / grad_measurements
            metrics["grad_total_norm"] = grad_total_sum / grad_measurements
        metrics.update(action_space_stats)
        return metrics
    def train(
            self,
            total_timesteps: int,
            eval_callback: Optional[Callable] = None,
            rollout_callback: Optional[Callable] = None,
            callback_manager: Optional[Any] = None,
            logger: Optional[Any] = None,
        ) -> Tuple[nn.Module, nn.Module]:
            """
            Train the PPO agent.
            
            Args:
                total_timesteps: Total timesteps to train
                eval_callback: Optional evaluation callback
                rollout_callback: Optional rollout callback
                callback_manager: Optional callback manager for stats accumulation
                logger: Optional logger
                
            Returns:
                Tuple of (trained_actor, trained_critic)
            """
            if total_timesteps <= 0:
                raise ValueError("total_timesteps must be positive")
            
            # Calculate training parameters
            steps_per_iteration = self.n_steps * self.n_envs
            n_iterations = total_timesteps // steps_per_iteration
            
            print(f"\n{'='*60}")
            print(f"Starting TorchRL PPO Training")
            print(f"{'='*60}")
            print(f"Total timesteps: {total_timesteps:,}")
            print(f"Steps per iteration: {steps_per_iteration}")
            print(f"Number of iterations: {n_iterations}")
            print(f"Mini-batch size: {self.batch_size}")
            print(f"Optimization epochs per iteration: {self.n_epochs}")
            print(f"{'='*60}\n")
            
            # Training loop
            for iteration in range(n_iterations):
                iteration_start_time = time.time()
                print(f"\nIteration {iteration + 1}/{n_iterations}")
                
                # ==================== Data Collection ====================
                if rollout_callback is not None:
                    rollout_callback.on_rollout_start()
                
                # Create persistent collector on first iteration
                if self._rollout_collector is None:
                    print("  Creating persistent rollout collector (one-time setup)...")
                    self._rollout_collector = RolloutCollector(
                        env=self.train_env,
                        actor=self.actor,
                        n_envs=self.n_envs,
                        n_steps=self.n_steps,
                        device=self.device,
                        debug=self.debug_mode,
                    )
                    print("  Collector ready!\n")
                
                # Use the persistent collector
                experiences, stats = self._rollout_collector.collect(
                    critic=self.critic,
                    rollout_callback=(
                        rollout_callback.on_step if rollout_callback is not None else None
                    ),
                )
                if self.debug_mode:
                    self._debug_print_rollout_stats(experiences, stats)
                
                if rollout_callback is not None:
                    rollout_callback.on_rollout_end()
                
                # Update global step counter
                self.global_step += steps_per_iteration
                
                print(f"  Collected {steps_per_iteration} transitions")
                
                # Accumulate episode stats
                if callback_manager is not None and stats["episode_info"]:
                    if self.verbose_cb:
                        print(f"[PPOAgent] Accumulating {len(stats['episode_info'])} episode stats for training")
                    callback_manager.accumulate_episode_stats(stats["episode_info"], mode="train")
                print(f"  Rollout time: {time.time() - iteration_start_time:.2f}s\n")
                # ==================== Policy Optimization ====================
                print(f"Training model")
                training_start_time = time.time()
                
                train_metrics = self.learn(
                    experiences=experiences,
                    n_steps=self.n_steps,
                    n_envs=self.n_envs,
                    metrics_callback=callback_manager.train_callback if callback_manager else None,
                )
                if self.debug_mode:
                    self._debug_print_optimizer_stats(train_metrics)
                
                # Log training metrics
                if logger is not None:
                    logger.log_training_step(
                        global_step=self.global_step,
                        policy_loss=train_metrics["policy_loss"],
                        value_loss=train_metrics["value_loss"],
                        entropy=train_metrics["entropy"],
                    )
                
                # Prepare metrics for callback
                formatted_metrics = {
                    "approx_kl": f"{train_metrics['approx_kl']:.4f}",
                    "clip_fraction": f"{train_metrics['clip_fraction']:.3f}",
                    "clip_range": f"{self.clip_range:.1f}",
                    "entropy_loss": f"{-train_metrics['entropy']:.5f}",
                    "explained_variance": f"{train_metrics['explained_variance']:.2f}",
                    "learning_rate": f"{self.optimizer.param_groups[0]['lr']:.4f}",
                    "loss": f"{train_metrics['policy_loss'] + train_metrics['value_loss']:.3f}",
                    "n_updates": str(train_metrics["n_updates"]),
                    "policy_gradient_loss": f"{train_metrics['policy_loss']:.3f}",
                    "value_loss": f"{train_metrics['value_loss']:.2f}",
                }
                training_time = time.time() - training_start_time
                print(f"Time to train {training_time:.2f}s\n")
                # ==================== Evaluation ====================
                eval_start_time = time.time()
                print('---------------evaluation started---------------')
                if eval_callback is not None and eval_callback.should_evaluate(iteration + 1):
                    eval_callback.on_evaluation_start(iteration + 1, self.global_step)
                    
                    # Run evaluation to get metrics
                    eval_metrics = self._run_evaluation(
                        actor=self.actor,
                        eval_env=self.eval_env,
                        sampler=self.sampler,
                        data_handler=self.data_handler,
                        callback=eval_callback,
                    )
                    
                    # Notify callback of evaluation end with actual metrics
                    is_new_best = eval_callback.on_evaluation_end(
                        iteration + 1, self.global_step, eval_metrics
                    )
                    
                    if is_new_best and self.model_save_path is not None:
                        self.best_model_path = self._save_checkpoint(
                            epoch=iteration + 1,
                            timestep=self.global_step,
                            metrics=eval_metrics,
                            prefix="best_eval",
                        )
                        print(f"  ★ New best model saved!")
                print(f'---------------evaluation finished---------------  took {time.time() - eval_start_time:.2f} seconds')
                # ==================== End of Iteration ====================
                if callback_manager is not None:
                    callback_manager.on_iteration_end(
                        iteration=iteration + 1,
                        global_step=self.global_step,
                        train_metrics=formatted_metrics,
                        n_envs=self.n_envs,
                    )
                
                # ==================== Periodic Checkpoint ====================
                if (iteration + 1) % 10 == 0 and self.model_save_path is not None:
                    self._save_checkpoint(
                        epoch=iteration + 1,
                        timestep=self.global_step,
                        metrics=train_metrics,
                        prefix="last_epoch",
                    )
            
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"{'='*60}\n")
            
            return self.actor, self.critic
    
    def _run_evaluation(
        self,
        actor: nn.Module,
        eval_env: Any,
        sampler: Any,
        data_handler: Any,
        callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on the validation set.
        
        Args:
            actor: Actor network to evaluate
            eval_env: Evaluation environment
            sampler: Negative sampler
            data_handler: Data handler with validation queries
            callback: Optional callback for accumulating episode stats
            
        Returns:
            Dictionary of evaluation metrics
        """
        from model_eval import evaluate_ranking_metrics
        
        actor.eval()
        
        # Prepare evaluation data
        # Get the materialized valid split
        valid_split = data_handler.get_materialized_split('valid')
        eval_queries_tensor = valid_split.queries
        
        # Get number of eval queries from args if available
        n_eval_queries = eval_queries_tensor.shape[0]
        if self.args and hasattr(self.args, 'n_eval_queries') and self.args.n_eval_queries:
            n_eval_queries = min(self.args.n_eval_queries, n_eval_queries)
        
        eval_data = eval_queries_tensor[:n_eval_queries]
        eval_depths = (
            data_handler.valid_queries_depths[:n_eval_queries]
            if hasattr(data_handler, 'valid_queries_depths') and data_handler.valid_queries_depths is not None
            else None
        )
        
        # Determine n_corruptions from args
        if self.args and hasattr(self.args, 'eval_neg_samples'):
            n_corruptions = self.args.eval_neg_samples
        else:
            raise ValueError("n_corruptions not specified in args")
               
        # Run evaluation to get metrics
        try:
            # Prepare queries for evaluation
            queries = eval_data
            
            # queries is already a tensor from the materialized split
            # It should be [N, A, D] where A is atoms per query, D is dimensions per atom
            if queries.dim() == 2:  # [N, 3] -> [N, 1, 3]
                queries = queries.unsqueeze(1)
            
            # Get corruption scheme from args
            corruption_scheme = ['head', 'tail']
            if self.args and hasattr(self.args, 'corruption_scheme'):
                corruption_scheme = self.args.corruption_scheme
            
            print(f"  Evaluating on {queries.shape[0]} queries with {n_corruptions} corruptions per query...")
            
            # Call evaluate_ranking_metrics with the correct signature
            metrics = evaluate_ranking_metrics(
                actor=actor,
                env=eval_env,
                queries=queries,
                sampler=sampler,
                n_corruptions=n_corruptions,
                corruption_modes=corruption_scheme,
                deterministic=True,
                verbose=False,
            )
            
            print(f"  Evaluation complete: MRR={metrics.get('MRR', 0.0):.4f}, Hits@1={metrics.get('Hits@1', 0.0):.4f}, Hits@10={metrics.get('Hits@10', 0.0):.4f}")
            
            # Transform keys to match expected format (lowercase with _mean suffix)
            # evaluate_ranking_metrics returns: MRR, Hits@1, Hits@3, Hits@10, per_mode
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
            
            metrics = transformed_metrics
            
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        actor.train()
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        timestep: int,
        metrics: dict,
        prefix: str = "checkpoint",
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            timestep: Current timestep
            metrics: Training metrics
            prefix: Filename prefix
            
        Returns:
            Path to saved checkpoint
        """
        if self.model_save_path is None:
            raise ValueError("model_save_path not set")
        
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'timestep': timestep,
            'metrics': metrics,
        }
        
        filename = self.model_save_path / f"{prefix}_epoch_{epoch}_step_{timestep}.pt"
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")
        
        return filename
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Tuple[int, int, dict]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, timestep, metrics)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        timestep = checkpoint['timestep']
        metrics = checkpoint.get('metrics', {})
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {epoch}, Timestep: {timestep}")
        
        return epoch, timestep, metrics
    
    def cleanup(self):
        """Clean up resources, especially the persistent collector."""
        if self._rollout_collector is not None:
            print("Shutting down persistent rollout collector...")
            self._rollout_collector.shutdown()
            self._rollout_collector = None
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
