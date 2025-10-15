import numpy as np
import os
import json
from typing import Optional, Union, Any, List, Dict, Tuple
import time
import sys
from collections import defaultdict

# from model_eval import evaluate_policy as evaluate_policy_mrr
from model_eval import eval_corruptions as eval_corruptions_mrr
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


def _format_depth_key(depth_value: Any) -> str:
    """Normalize depth IDs so metrics share consistent naming."""
    if depth_value in (None, -1):
        return "unknown"
    try:
        return str(int(depth_value))
    except (TypeError, ValueError):
        return "unknown"


def _format_stat_string(mean: Optional[float], std: Optional[float], count: int) -> str:
    """Return metric display as 'mean +/- std (count)' with fixed precision."""
    if mean is None or std is None or count == 0:
        return "N/A"
    return f"{mean:.3f} +/- {std:.2f} ({count})"


class CustomEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param model_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        model_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        name = 'rl_model',
        ):

        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=model_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

        self.name = name
        self.best_epoch = None

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.model_path = model_path
        self.log_path = log_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []
        # Consolidated tracking: (label, depth) -> list of episode stats
        self._episode_stats: defaultdict[tuple[int, str], List[Dict[str, float]]] = defaultdict(list)
        self._last_episode_id: Dict[int, int] = {}


    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            start = time.time()
            print('---------------evaluation started---------------')
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self._reset_eval_depth_stats()
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._evaluation_step_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            # Add to current Logger
            overall_reward_display = self._format_metric(mean_reward, std_reward, len(episode_rewards))
            overall_length_display = self._format_metric(mean_ep_length, std_ep_length, len(episode_lengths))
            self.logger.record("eval/reward_overall", overall_reward_display)
            self.logger.record("eval/length mean +/- std", overall_length_display)

            label_metrics = self._compute_label_metrics()
            label_log_entries = self._log_label_metrics(label_metrics)
            depth_log_entries = self._log_depth_reward_stats()
            success_log_entries = self._log_depth_success_stats()

            success_rate = float("nan")
            if len(self._is_success_buffer) > 0:
                success_rate = float(np.mean(self._is_success_buffer))
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/total_timesteps", self.num_timesteps, exclude="tensorboard")

            # If a new best model is found, save it
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = float(mean_reward)
                self.best_epoch = self.num_timesteps
                if self.model_path is not None:
                    self.model.save(os.path.join(self.model_path, f"best_eval_{self.name}.zip"))
                    print(f"Saved new best eval model to {self.model_path}")
                    self.write_info(description='best_eval')
                
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.log_path:
                log_payload = {
                    "eval/reward_mean_overall": mean_reward,
                    "eval/reward_std_overall": std_reward,
                    "eval/mean_ep_length": mean_ep_length,
                    "eval/std_ep_length": std_ep_length,
                    "eval/success_rate": success_rate,
                    "eval/num_timesteps": self.num_timesteps,
                    "eval/reward_count_total": len(episode_rewards),
                    "eval/count_ep_length_total": len(episode_lengths),
                }
                log_payload.update(label_log_entries)
                log_payload.update(depth_log_entries)
                log_payload.update(success_log_entries)
                self._log_values(log_payload)

            if self.logger is not None:
                # Mirror training-style output for eval metrics
                self.logger.dump(step=self.num_timesteps)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            print(f'---------------evaluation finished---------------  took {time.time()-start:.2f} seconds')
        return continue_training


    def _reset_eval_depth_stats(self) -> None:
        """Reset consolidated episode tracking."""
        self._episode_stats = defaultdict(list)
        self._last_episode_id = {}

    def _accumulate_depth_rewards(self, infos: List[Dict[str, Any]], append_success: bool = False) -> None:
        """Accumulate episode stats by label and depth."""
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue
            label = info.get("label")
            if label is None:
                continue
            episode_data = info.get("episode")
            if not isinstance(episode_data, dict):
                continue
            
            # Check for duplicate episodes
            episode_idx: Optional[int] = info.get("episode_idx")
            if episode_idx is not None:
                if self._last_episode_id.get(env_idx) == episode_idx:
                    continue
                self._last_episode_id[env_idx] = episode_idx
            else:
                episode_id = id(episode_data)
                if self._last_episode_id.get(env_idx) == episode_id:
                    continue
                self._last_episode_id[env_idx] = episode_id
            
            # Extract stats
            reward = episode_data.get("r")
            length = episode_data.get("l")
            depth_value = info.get("query_depth")
            success_flag = bool(info.get("is_success", False))
            
            try:
                label_value = int(label)
            except (TypeError, ValueError):
                continue
            
            # Store in consolidated structure
            depth_key = _format_depth_key(depth_value)
            stats = {
                "reward": float(reward) if reward is not None else None,
                "length": float(length) if length is not None else None,
                "success": 1.0 if success_flag else 0.0,
                "depth_raw": depth_value,  # Keep raw depth for detailed logging
            }
            self._episode_stats[(label_value, depth_key)].append(stats)
            
            if append_success:
                self._is_success_buffer.append(success_flag)

    @staticmethod
    def _format_metric(mean: Optional[float], std: Optional[float], count: int) -> str:
        return _format_stat_string(mean, std, count)

    def _compute_single_label_metrics(
        self,
        rewards: List[float],
        lengths: List[float],
    ) -> Dict[str, Union[float, int, None, bool]]:
        stats: Dict[str, Union[float, int, None, bool]] = {
            "reward_mean": None,
            "reward_std": None,
            "reward_count": 0,
            "length_mean": None,
            "length_std": None,
            "length_count": 0,
        }

        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            stats["reward_mean"] = float(rewards_arr.mean())
            stats["reward_std"] = float(rewards_arr.std()) if rewards_arr.size > 1 else 0.0
            stats["reward_count"] = len(rewards)

        if lengths:
            lengths_arr = np.asarray(lengths, dtype=np.float32)
            stats["length_mean"] = float(lengths_arr.mean())
            stats["length_std"] = float(lengths_arr.std()) if lengths_arr.size > 1 else 0.0
            stats["length_count"] = len(lengths)

        stats["has_data"] = bool(rewards or lengths)
        return stats

    def _compute_label_metrics(self) -> Dict[str, Dict[str, Union[float, int, None, bool]]]:
        """Compute metrics aggregated by label from consolidated stats."""
        label_metrics: Dict[str, Dict[str, Union[float, int, None, bool]]] = {}
        for label_value, label_name in ((1, "pos"), (0, "neg")):
            rewards = []
            lengths = []
            for (lbl, _), episodes in self._episode_stats.items():
                if lbl == label_value:
                    for ep in episodes:
                        if ep.get("reward") is not None:
                            rewards.append(ep["reward"])
                        if ep.get("length") is not None:
                            lengths.append(ep["length"])
            label_metrics[label_name] = self._compute_single_label_metrics(rewards, lengths)
        return label_metrics

    def _log_label_metrics(
        self,
        label_metrics: Dict[str, Dict[str, Union[float, int, None, bool]]],
    ) -> Dict[str, Union[float, int]]:
        label_payload: Dict[str, Union[float, int]] = {}
        for label_name, stats in label_metrics.items():
            reward_mean = stats.get("reward_mean")
            reward_std = stats.get("reward_std")
            reward_count = int(stats.get("reward_count", 0))
            reward_display = self._format_metric(reward_mean, reward_std, reward_count)
            if reward_mean is not None and reward_std is not None:
                label_payload[f"eval/reward_mean_{label_name}"] = float(reward_mean)
                label_payload[f"eval/reward_std_{label_name}"] = float(reward_std)
            if self.logger is not None:
                self.logger.record(f"eval/reward_label_{label_name}", reward_display)
            label_payload[f"eval/reward_count_{label_name}"] = reward_count

            length_mean = stats.get("length_mean")
            length_std = stats.get("length_std")
            length_count = int(stats.get("length_count", 0))
            length_display = self._format_metric(length_mean, length_std, length_count)
            if length_mean is not None and length_std is not None:
                label_payload[f"eval/mean_ep_length_{label_name}"] = float(length_mean)
                label_payload[f"eval/std_ep_length_{label_name}"] = float(length_std)
            if self.logger is not None:
                self.logger.record(f"eval/len_{label_name}", length_display)
            label_payload[f"eval/count_ep_length_{label_name}"] = length_count
        return label_payload

    def _compute_overall_episode_stats(self) -> Dict[str, Union[float, int, None]]:
        """Compute overall stats from all episodes."""
        all_rewards: List[float] = []
        all_lengths: List[float] = []
        for episodes in self._episode_stats.values():
            for ep in episodes:
                if ep.get("reward") is not None:
                    all_rewards.append(ep["reward"])
                if ep.get("length") is not None:
                    all_lengths.append(ep["length"])

        stats: Dict[str, Union[float, int, None]] = {
            "reward_mean": None,
            "reward_std": None,
            "reward_count": 0,
            "length_mean": None,
            "length_std": None,
            "length_count": 0,
        }

        if all_rewards:
            rewards_arr = np.asarray(all_rewards, dtype=np.float32)
            stats["reward_mean"] = float(rewards_arr.mean())
            stats["reward_std"] = float(rewards_arr.std()) if rewards_arr.size > 1 else 0.0
            stats["reward_count"] = len(all_rewards)

        if all_lengths:
            lengths_arr = np.asarray(all_lengths, dtype=np.float32)
            stats["length_mean"] = float(lengths_arr.mean())
            stats["length_std"] = float(lengths_arr.std()) if lengths_arr.size > 1 else 0.0
            stats["length_count"] = len(all_lengths)

        return stats

    def _log_depth_reward_stats(self) -> Dict[str, Union[float, int]]:
        """Log reward stats by depth from consolidated episode stats."""
        depth_log_entries: Dict[str, Union[float, int]] = {}
        if not self._episode_stats:
            return depth_log_entries

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
            rewards = [ep["reward"] for ep in episodes if ep.get("reward") is not None]
            if not rewards:
                continue
            
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            count = rewards_arr.size
            mean_depth = float(rewards_arr.mean())
            std_depth = float(rewards_arr.std()) if count > 1 else 0.0
            label_str = "pos" if label == 1 else "neg" if label == 0 else f"label{label}"
            reward_display = _format_stat_string(mean_depth, std_depth, count)
            
            if self.logger is not None and label_str != "neg":  # for negatives we dont have depth info
                self.logger.record(f"eval/reward_d_{depth_key}_{label_str}", reward_display)
            
            depth_log_entries[f"eval/reward_depth_{depth_key}_{label_str}_mean"] = mean_depth
            depth_log_entries[f"eval/reward_depth_{depth_key}_{label_str}_std"] = std_depth
            depth_log_entries[f"eval/reward_depth_{depth_key}_{label_str}_count"] = count

        return depth_log_entries

    def _log_depth_success_stats(self) -> Dict[str, Union[float, int]]:
        """Log success stats by depth from consolidated episode stats."""
        success_entries: Dict[str, Union[float, int]] = {}

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

        # Log by depth
        for (label, depth_key), episodes in sorted(self._episode_stats.items(), key=_order_key):
            successes = [ep["success"] for ep in episodes if "success" in ep]
            if not successes:
                continue
            
            values_arr = np.asarray(successes, dtype=np.float32)
            count = values_arr.size
            mean_success = float(values_arr.mean())
            std_success = float(values_arr.std()) if count > 1 else 0.0
            label_str = "pos" if label == 1 else "neg" if label == 0 else f"label{label}"
            display = _format_stat_string(mean_success, std_success, count)
            key_display = f"eval/proven_d_{depth_key}_{label_str}"
            
            if self.logger is not None:
                self.logger.record(key_display, display)
            
            success_entries[f"eval/proven_depth_{depth_key}_{label_str}_mean"] = mean_success
            success_entries[f"eval/proven_depth_{depth_key}_{label_str}_std"] = std_success
            success_entries[f"eval/proven_depth_{depth_key}_{label_str}_count"] = int(count)

        # Log overall by label
        for label in (1, 0):
            successes = []
            for (lbl, _), episodes in self._episode_stats.items():
                if lbl == label:
                    successes.extend([ep["success"] for ep in episodes if "success" in ep])
            
            if not successes:
                continue
            
            values_arr = np.asarray(successes, dtype=np.float32)
            count = values_arr.size
            mean_success = float(values_arr.mean())
            std_success = float(values_arr.std()) if count > 1 else 0.0
            label_str = "pos" if label == 1 else "neg"
            display = _format_stat_string(mean_success, std_success, count)
            
            if self.logger is not None:
                self.logger.record(f"eval/proven_{label_str}", display)
            
            success_entries[f"eval/proven_{label_str}_mean"] = mean_success
            success_entries[f"eval/proven_{label_str}_std"] = std_success
            success_entries[f"eval/proven_{label_str}_count"] = int(count)

        return success_entries

    def _evaluation_step_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        stop_training = bool(self._log_success_callback(locals_, globals_))
        self._accumulate_depth_rewards(locals_.get("infos", []))
        return stop_training

    def _log_values(self, logs):
        """Logs values based on the current headers."""
        with open(self.log_path, "a") as f:
            f.write(";".join(f'{k}:{v}' for k, v in logs.items())+ "\n")

    def write_info(self, description: str):
        """Save checkpoint metadata."""
        info = {
            'metric': 'eval/reward_mean_overall',
            'best_value': float(self.best_mean_reward),
            'timesteps': self.num_timesteps,
            'epoch': self.best_epoch,
        }
        info_path = os.path.join(self.model_path, f'info_{description}_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_best_ckpt(self,env):
        """
        Restore the best model found during evaluation.
        """
        if self.model_path is None:
            print("Warning: `model_path` is not set. Cannot restore model.")
            return

        model_files = [f for f in os.listdir(self.model_path) if '.zip' in f and 'best_eval_' in f]
        if len(model_files) >= 1:
            model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
            self.model = PPO.load(
                os.path.join(self.model_path, model_files[0]),
                env=env,          # or whatever env you need
                device=self.model.device,
                print_system_info=False
            )
            # load the info file to get the best epoch and value
            info_path = os.path.join(self.model_path, f'info_best_eval_{self.name}.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                best_timestep = info.get('timesteps', None)
                best_mean_reward = info.get('best_value', None)
                print(f'Restored best val model from step {best_timestep}, with mean_reward={best_mean_reward:.3f}.')
            else:
                raise ValueError(f"Warning: Info file not found: {info_path}")
        else:
            raise ValueError("No best model found to restore.")
        return self.model




class CustomEvalCallbackMRR(CustomEvalCallback):
    """
    Callback for evaluating an agent. It uses the `eval_corruptions` function 
    for comprehensive MRR metrics but is adapted to behave identically to 
    `CustomEvalCallback` by mapping the results to the same internal variables.

    This version saves the best model according to a configurable evaluation metric (default: AUC-PR).
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        sampler: Any,
        eval_data: List[Any],
        eval_data_depths: Optional[List[int]] = None,
        n_corruptions: Optional[int] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        model_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        name='rl_model',
        consult_janus: bool = False,
        corruption_scheme: Optional[List[str]] = None,
        best_metric: str = "auc_pr",
    ):
        self.verbose = verbose
        assert eval_env.type_ == "custom_dummy", "Requires custom_dummy VecEnv"
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            model_path=model_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
            name=name,
        )
        
        # Specific attributes for MRR evaluation
        assert hasattr(eval_env, 'type_') and eval_env.type_ == "custom_dummy", "Requires custom_dummy VecEnv"
        self.sampler = sampler
        self.eval_data = eval_data
        self.eval_data_depths = eval_data_depths
        self.n_corruptions = n_corruptions
        self.corruption_scheme = corruption_scheme
        self.consult_janus = consult_janus
        metric_options = {
            "auc_pr": {
                "display_name": "AUC-PR",
                "log_key": "eval/auc_pr",
            },
            "mrr": {
                "display_name": "MRR",
                "log_key": "eval/_mrr",
            },
        }
        metric_key = (best_metric or "auc_pr").lower()
        if metric_key not in metric_options:
            allowed = ", ".join(sorted(metric_options.keys()))
            raise ValueError(f"Unsupported best_metric '{best_metric}'. Allowed values: {allowed}.")
        self._metric_options = metric_options
        self.best_metric = metric_key
        self._metric_display_name = metric_options[metric_key]["display_name"]
        self._metric_log_key = metric_options[metric_key]["log_key"]
        self.best_metric_value = -np.inf


    def _on_step(self) -> bool:
        """
        This method is called by the model after each call to `env.step()`.
        """
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            start = time.time()
            print('---------------evaluation started---------------')
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer and depth tracking
            self.model.policy.set_training_mode(False)
            self._is_success_buffer = []
            self._reset_eval_depth_stats()
            
            # Run MRR evaluation
            eval_results = eval_corruptions_mrr(
                self.model,
                self.eval_env,
                self.eval_data,
                self.sampler,
                n_corruptions=self.n_corruptions,
                deterministic=self.deterministic,
                verbose=self.verbose,
                corruption_scheme=self.corruption_scheme,
                plot=False,
                info_callback=self._info_callback_eval,
                data_depths=self.eval_data_depths,
            )
            self.model.policy.set_training_mode(True)

            # Extract MRR-specific metrics
            mean_mrr = eval_results.get('mrr_mean', 0.0)
            auc_pr = eval_results.get('average_precision', 0.0)

            # Log MRR metrics
            self.logger.record("eval/_mrr", f"{mean_mrr:.3f}")
            self.logger.record("eval/auc_pr", f"{auc_pr:.3f}")

            # Compute and log standard metrics from accumulated stats
            label_metrics = self._compute_label_metrics()
            label_log_entries = self._log_label_metrics(label_metrics)
            overall_stats = self._compute_overall_episode_stats()
            
            overall_reward_mean = overall_stats.get("reward_mean")
            overall_reward_std = overall_stats.get("reward_std")
            overall_length_mean = overall_stats.get("length_mean")
            overall_length_std = overall_stats.get("length_std")
            overall_reward_count = int(overall_stats.get("reward_count", 0))
            overall_length_count = int(overall_stats.get("length_count", 0))

            if self.logger is not None:
                overall_reward_display = self._format_metric(
                    overall_reward_mean, overall_reward_std, overall_reward_count
                )
                self.logger.record("eval/reward_overall", overall_reward_display)
                overall_length_display = self._format_metric(
                    overall_length_mean, overall_length_std, overall_length_count
                )
                self.logger.record("eval/length mean +/- std", overall_length_display)

            depth_log_entries = self._log_depth_reward_stats()
            success_log_entries = self._log_depth_success_stats()

            success_rate = float("nan")
            if len(self._is_success_buffer) > 0:
                success_rate = float(np.mean(self._is_success_buffer))
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/total_timesteps", self.num_timesteps, exclude="tensorboard")

            # Check if best metric improved
            metric_values = {
                "auc_pr": float(auc_pr),
                "mrr": float(mean_mrr),
            }
            current_metric_value = metric_values[self.best_metric]
            if current_metric_value > self.best_metric_value:
                print(f"New best {self._metric_display_name} in eval: {current_metric_value:.4f}!")
                self.best_metric_value = current_metric_value
                self.best_epoch = self.num_timesteps
                if self.model_path is not None:
                    self.model.save(os.path.join(self.model_path, f"best_eval_{self.name}.zip"))
                    self.write_info(description='best_eval')
                
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.log_path:
                # Extract detailed length metrics if available
                mean_ep_length_pos_true = eval_results.get('episode_len_pos_true_mean', 0.0)
                std_ep_length_pos_true = eval_results.get('episode_len_pos_true_std', 0.0)
                mean_ep_length_pos_false = eval_results.get('episode_len_pos_false_mean', 0.0)
                std_ep_length_pos_false = eval_results.get('episode_len_pos_false_std', 0.0)
                mean_ep_length_neg_true = eval_results.get('episode_len_neg_true_mean', 0.0)
                std_ep_length_neg_true = eval_results.get('episode_len_neg_true_std', 0.0)
                mean_ep_length_neg_false = eval_results.get('episode_len_neg_false_mean', 0.0)
                std_ep_length_neg_false = eval_results.get('episode_len_neg_false_std', 0.0)

                log_payload = {
                    "eval/num_timesteps": self.num_timesteps,
                    "eval/mrr_mean": mean_mrr,
                    "eval/auc_pr": auc_pr,
                    "eval/mean_ep_length_pos_true": mean_ep_length_pos_true,
                    "eval/std_ep_length_pos_true": std_ep_length_pos_true,
                    "eval/mean_ep_length_pos_false": mean_ep_length_pos_false,
                    "eval/std_ep_length_pos_false": std_ep_length_pos_false,
                    "eval/mean_ep_length_neg_true": mean_ep_length_neg_true,
                    "eval/std_ep_length_neg_true": std_ep_length_neg_true,
                    "eval/mean_ep_length_neg_false": mean_ep_length_neg_false,
                    "eval/std_ep_length_neg_false": std_ep_length_neg_false,
                    "eval/success_rate": success_rate,
                    "eval/reward_count_total": overall_reward_count,
                    "eval/count_ep_length_total": overall_length_count,
                }
                
                if overall_reward_mean is not None and overall_reward_std is not None:
                    log_payload["eval/reward_mean_overall"] = float(overall_reward_mean)
                    log_payload["eval/reward_std_overall"] = float(overall_reward_std)
                if overall_length_mean is not None and overall_length_std is not None:
                    log_payload["eval/mean_ep_length"] = float(overall_length_mean)
                    log_payload["eval/std_ep_length"] = float(overall_length_std)
                
                log_payload.update(label_log_entries)
                log_payload.update(depth_log_entries)
                log_payload.update(success_log_entries)
                self._log_values(log_payload)

            if self.logger is not None:
                self.logger.dump(step=self.num_timesteps)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            print(f'\n---------------evaluation finished---------------  took {time.time()-start:.2f} seconds')
        return continue_training

    def _info_callback_eval(self, infos: List[Dict[str, Any]]) -> None:
        self._accumulate_depth_rewards(infos, append_success=True)

    # Save info based on AUC-PR/MRR
    def write_info(self, description: str):
        """Save checkpoint metadata."""
        info = {
            'metric': self._metric_log_key,
            'best_value': float(self.best_metric_value),
            'timesteps': self.num_timesteps,
            'epoch': self.best_epoch,
        }
        info_path = os.path.join(self.model_path, f'info_{description}_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    # Restore and report based on AUC-PR/MRR
    def restore_best_ckpt(self, env):
        """
        Restore the best model found during evaluation.
        """
        if self.model_path is None:
            print("Warning: `model_path` is not set. Cannot restore model.")
            return

        model_files = [f for f in os.listdir(self.model_path) if '.zip' in f and 'best_eval_' in f]
        if len(model_files) >= 1:
            model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
            self.model = PPO.load(
                os.path.join(self.model_path, model_files[0]),
                env=env,
                device=self.model.device,
                print_system_info=False
            )
            info_path = os.path.join(self.model_path, f'info_best_eval_{self.name}.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                best_timestep = info.get('timesteps', None)
                best_metric_value = info.get('best_value', None)
                metric_key = info.get('metric', self._metric_log_key)
                display_name = self._metric_display_name
                for cfg in self._metric_options.values():
                    if metric_key == cfg["log_key"]:
                        display_name = cfg["display_name"]
                        break
                if best_metric_value is not None:
                    print(
                        f"Restored best val model from step {best_timestep}, with {display_name}={best_metric_value:.3f}."
                    )
                    self.best_metric_value = float(best_metric_value)
                else:
                    print(f"Restored best val model from step {best_timestep}.")
                if best_timestep is not None:
                    self.best_epoch = best_timestep
            else:
                raise ValueError(f"Warning: Info file not found: {info_path}")
        else:
            raise ValueError("No best model found to restore.")
        return self.model




class KGELogitGainSchedulerCallback(BaseCallback):
    """logit-level KGE shaping during training: Anneals the gain used.
    Gain is the multiplicative factor applied to the KGE logits before adding
    them to the policy logits. Higher gain increases the influence of KGE shaping.
    The gain is annealed from an initial value to a final value over a specified
    number of training steps, optionally after a warmup period.
    :param warmup_steps: amount of steps to wait before starting the annealing.
    :param anneal_steps: number of steps over which to gradually change (anneal) the gain."""

    def __init__(
        self,
        initial_gain: float = 1.0,
        final_gain: float = 0.2,
        anneal_steps: int = 300000,
        warmup_steps: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_gain = float(initial_gain)
        self.final_gain = float(final_gain)
        self.anneal_steps = max(0, int(anneal_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self._last_logged_gain: Optional[float] = None

    def _init_callback(self) -> None:
        policy = getattr(self.model, "policy", None)
        if hasattr(policy, "set_kge_logit_gain"):
            policy.set_kge_logit_gain(self.initial_gain)
            self._last_logged_gain = self.initial_gain
            if self.verbose > 0:
                print(f"[KGEGain] Initialized gain to {self.initial_gain:.4f}")

    def _on_step(self) -> bool:
        policy = getattr(self.model, "policy", None)
        if not hasattr(policy, "update_kge_logit_gain"):
            return True

        policy.update_kge_logit_gain(self.num_timesteps)

        if self.verbose > 0:
            current_gain = getattr(policy, "kge_logit_gain", None)
            if current_gain is not None and self._last_logged_gain != current_gain:
                print(f"[KGEGain] Step {self.num_timesteps}: gain -> {float(current_gain):.4f}")
                self._last_logged_gain = float(current_gain)
        return True


class TopKCurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning of top_k_actions parameter.
    Gradually reduces the number of actions from initial to final value based on training progress.
    
    :param initial_k: Initial value for top_k_actions (None for no filtering, or integer)
    :param final_k: Final value for top_k_actions
    :param total_timesteps: Total training timesteps (to calculate progress)
    :param schedule: Schedule type - 'linear', 'exponential', or 'step'
    :param step_thresholds: For 'step' schedule, list of (progress, k_value) tuples
    :param start_filter_timesteps: Number of timesteps to wait before enabling filtering
    :param verbose: Verbosity level
    """
    
    def __init__(
        self,
        initial_k: Optional[int] = None,
        final_k: int = 5,
        total_timesteps: int = 1000000,
        schedule: str = 'linear',
        step_thresholds: Optional[List[Tuple[float, int]]] = None,
        start_filter_timesteps: int = 0,
        verbose: int = 1,
    ):
        super(TopKCurriculumCallback, self).__init__(verbose)
        self.initial_k = initial_k
        self.final_k = final_k
        self.total_timesteps = total_timesteps
        self.schedule = schedule
        self.step_thresholds = step_thresholds or [(0.0, initial_k), (0.5, 10), (1.0, final_k)]
        self.start_filter_timesteps = max(0, int(start_filter_timesteps))
        self.current_k = None if self.start_filter_timesteps > 0 else initial_k
        self._last_progress = 0.0
        
        # Validate inputs
        if self.schedule == 'step' and not self.step_thresholds:
            raise ValueError("step_thresholds must be provided when using 'step' schedule")
        
        if self.schedule not in ['linear', 'exponential', 'step','cte']:
            raise ValueError(f"Unknown schedule type: {self.schedule}")
    
    def _init_callback(self) -> None:
        """Initialize callback."""
        # Set initial k value
        if hasattr(self.model.policy, 'top_k_actions'):
            initial_value = None if self.start_filter_timesteps > 0 else self.initial_k
            self.model.policy.top_k_actions = initial_value
            if self.verbose > 0:
                start_msg = (
                    f"delayed start ({self.start_filter_timesteps} timesteps)"
                    if self.start_filter_timesteps > 0
                    else "immediate start"
                )
                print(
                    "TopK Curriculum: Starting with top_k_actions = "
                    f"{initial_value if initial_value is not None else 'None (no filtering)'}"
                    f" [{start_msg}]",
                    flush=True,
                )
    
    def _compute_k_value(self, progress: float) -> Optional[int]:
        """
        Compute the current k value based on training progress.
        
        :param progress: Training progress from 0.0 to 1.0
        :return: Current k value (None for no filtering, or integer)
        """
        if self.schedule == 'linear':
            # Linear interpolation between initial and final
            if self.initial_k is None:
                # If starting with None, switch to large value then reduce
                effective_initial = 100  # Large value to simulate "no filtering"
                k = effective_initial - progress * (effective_initial - self.final_k)
                return None if progress < 0.1 else int(k)
            else:
                k = self.initial_k - progress * (self.initial_k - self.final_k)
                return max(int(k), self.final_k)
        
        elif self.schedule == 'exponential':
            # Exponential decay
            if self.initial_k is None:
                effective_initial = 20
                k = effective_initial * (self.final_k / effective_initial) ** progress
                return None if progress < 0.1 else int(k)
            else:
                k = self.initial_k * (self.final_k / self.initial_k) ** progress
                return max(int(k), self.final_k)
        
        elif self.schedule == 'step':
            # Step-based schedule
            for i in range(len(self.step_thresholds) - 1):
                if progress >= self.step_thresholds[i][0] and progress < self.step_thresholds[i + 1][0]:
                    return self.step_thresholds[i][1]
            # Return final value if progress >= last threshold
            return self.step_thresholds[-1][1]
        elif self.schedule == 'cte':
            return self.final_k

    def _on_step(self) -> bool:
        """
        This method is called after every environment step.
        Update top_k_actions based on training progress.
        """
        # Warmup phase without filtering
        if self.start_filter_timesteps > 0 and self.num_timesteps < self.start_filter_timesteps:
            progress = 0.0
            self._last_progress = progress
            new_k = None
        else:
            # Calculate training progress after warmup
            effective_total = max(self.total_timesteps - self.start_filter_timesteps, 1)
            progressed_timesteps = max(self.num_timesteps - self.start_filter_timesteps, 0)
            progress = min(progressed_timesteps / effective_total, 1.0)
            self._last_progress = progress

            # Compute new k value
            new_k = self._compute_k_value(progress)
        
        # Update if changed
        if new_k != self.current_k:
            self.current_k = new_k
            if hasattr(self.model.policy, 'top_k_actions'):
                self.model.policy.top_k_actions = new_k
                
                if self.verbose > 0:
                    k_display = "None (no filtering)" if new_k is None else str(new_k)
                    print(
                        "TopK Curriculum update -> "
                        f"progress: {progress:.2%}, top_k_actions: {k_display}",
                        flush=True,
                    )
                
        # Periodic status even if k didn't change
        elif self.verbose > 1 and self.num_timesteps % 10000 == 0:
            k_display = "None" if self.current_k is None else str(self.current_k)
            print(f"TopK Curriculum status -> progress: {progress:.2%}, top_k_actions: {k_display}", flush=True)
        
        return True

    def _on_rollout_end(self) -> None:
        k_display = "None" if self.current_k is None else self.current_k
        print(
            f"TopK Curriculum status -> timesteps: {self.num_timesteps:,}"
            f", progress: {self._last_progress:.2%}, top_k_actions: {k_display}",
            flush=True,
        )


class DepthProofStatsCallback(BaseCallback):
    """Track episode rewards per query depth during training rollouts."""

    def __init__(self, prefix: str = "rollout", track_negative: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.prefix = prefix
        self.track_negative = track_negative
        self._stats: defaultdict[tuple[int, str], List[float]] = defaultdict(list)
        self._success_values: defaultdict[tuple[int, str], List[float]] = defaultdict(list)
        self._success_values_by_label: defaultdict[int, List[float]] = defaultdict(list)
        self._last_episode_id: Dict[int, int] = {}

    def _init_callback(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._stats = defaultdict(list)
        self._success_values = defaultdict(list)
        self._success_values_by_label = defaultdict(list)
        self._last_episode_id = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue
            label = info.get("label")
            if label is None:
                continue
            if not self.track_negative and label != 1:
                continue
            episode_data = info.get("episode")
            reward = episode_data.get("r") if isinstance(episode_data, dict) else None
            if reward is None:
                continue
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
            depth_key = _format_depth_key(info.get("query_depth"))
            success_flag = bool(info.get("is_success", False))
            self._success_values[(label, depth_key)].append(1.0 if success_flag else 0.0)
            self._success_values_by_label[label].append(1.0 if success_flag else 0.0)
            key = (label, depth_key)
            self._stats[key].append(float(reward))
        return True

    def _log_stats(self) -> None:
        if not self._stats:
            self._reset()
            return
        for (label, depth_key), rewards in self._stats.items():
            if not rewards:
                continue
            label_str = "pos" if label == 1 else "neg"
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            count = rewards_arr.size
            mean_reward = float(rewards_arr.mean()) if count else 0.0
            std_reward = float(rewards_arr.std()) if count > 1 else 0.0
            reward_display = _format_stat_string(mean_reward, std_reward, count)
            if self.logger is not None and label_str != "neg":  # for negatives we dont have depth info
                self.logger.record(
                    f"{self.prefix}/reward_d_{depth_key}_{label_str}",
                    reward_display
                )
            if self.verbose > 0:
                print(
                    f"Depth {depth_key} ({label_str}) -> rwd: {reward_display}"
                )
        if self.logger is not None:
            def _success_sort(item: Tuple[tuple[int, str], List[float]]) -> Tuple[int, Union[int, float]]:
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

            for (label, depth_key), values in sorted(self._success_values.items(), key=_success_sort):
                if not values:
                    continue
                values_arr = np.asarray(values, dtype=np.float32)
                count = values_arr.size
                mean_success = float(values_arr.mean())
                std_success = float(values_arr.std()) if count > 1 else 0.0
                success_display = _format_stat_string(mean_success, std_success, count)
                label_str = "pos" if label == 1 else "neg"
                key = f"{self.prefix}/proven_d_{depth_key}_{label_str}"
                self.logger.record(key, success_display)
                if self.verbose > 0:
                    success_count = int(values_arr.sum())
                    print(f"Depth {depth_key} ({label_str}) -> proven: {success_display} | successes: {success_count}")

            for label in (1, 0):
                if label == 0 and not self.track_negative:
                    continue
                values = self._success_values_by_label.get(label, [])
                if not values:
                    continue
                values_arr = np.asarray(values, dtype=np.float32)
                count = values_arr.size
                mean_success = float(values_arr.mean())
                std_success = float(values_arr.std()) if count > 1 else 0.0
                success_display = _format_stat_string(mean_success, std_success, count)
                total_key = f"{self.prefix}/proven_{'pos' if label == 1 else 'neg'}"
                self.logger.record(total_key, success_display)
        self._reset()

    def _on_rollout_end(self) -> None:
        self._log_stats()

    def _on_training_end(self) -> None:
        self._log_stats()


class SB3TrainCheckpoint(BaseCallback):
    """Callback to save SB3 model weights when a monitored metric improves."""

    def __init__(
        self,
        model,
        monitor: str = "rollout/ep_rew_mean",
        model_path: Optional[str] = None,
        log_path: Optional[str] = None,
        maximize: bool = True,
        verbose: bool = True,
        frequency: int = 1,
        total_steps: int = 0,
        name: str = None
    ):
        """
        Monitors a metric and saves the last training ckpt.

        Args:
            model: The RL model to monitor and save.
            monitor: The metric name to monitor.
            model_path: Path to save the model checkpoints.
            save_best_only: If True, only save when the metric improves.
            maximize: If True, maximize the metric; otherwise, minimize.
            verbose: Print messages when saving.
            frequency: Frequency (in episodes) of checking for improvement.
            name: Custom name for the checkpoint.
        """
        super().__init__(verbose)
        self.model_ = model
        self.monitor = monitor
        self.model_path = model_path
        self.log_path = log_path
        self.maximize = maximize
        self.frequency = frequency
        self.name = name

        self.best_value = -sys.float_info.max if maximize else sys.float_info.max
        self.best_epoch = None
        self.current_value = None
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        """Called at each environment step. Skips until frequency is reached."""
        # Ensure frequency is handled in episode_end
        return True

    def _on_rollout_end(self) -> None:
        """Check metric and save model at the end of each rollout."""
        logs = {key: value for key, value in self.logger.name_to_value.items()}
        if self.log_path and logs:
                self._log_values(logs)

        if self.monitor not in logs:
            print(f'Metric "{self.monitor}" not found. Available: {logs.keys()}')
            return

        self.current_value = logs[self.monitor]
        improved = (self.maximize and self.current_value > self.best_value) or \
                   (not self.maximize and self.current_value < self.best_value)

        if improved:
            # Update best value
            self.best_value = self.current_value
            self.best_epoch = self.num_timesteps

            if self.verbose:
                print(f'Improved {self.monitor} to {self.current_value:.4f} in train')

        if self.model_path:
            save_path = os.path.join(self.model_path, f"last_epoch_{self.name}.zip")
            self.model_.save(save_path)
            self.write_info('last_epoch')
            if self.verbose:
                print(f"Saved final training model to {save_path}")

    def _log_headers(self, headers):
        """Logs headers to the file."""
        with open(self.log_file, "w") as f:
            f.write(";".join(headers) + "\n")
        self.headers_written = True
        return headers

    def _log_values(self, logs):
        """Logs values based on the current headers."""
        with open(self.log_path, "a") as f:
            f.write(";".join(f'{k}:{v}' for k, v in logs.items())+ "\n")

    def write_info(self, description: str):
        """Save checkpoint metadata."""
        if not self.model_path:
            return
        info = {
            'metric': self.monitor,
            f'last_{self.monitor}': float(self.current_value) if self.current_value is not None else None,
            'timesteps': self.num_timesteps,
            'finished_train': self.num_timesteps >= self.total_steps,
        }
        info_path = os.path.join(self.model_path,f'info_{description}_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_last_ckpt(self,env):
        """Restore the last train model."""
        if self.model_path is None:
            print("Warning: `model_path` is not set. Cannot restore model.")
            return
        
        model_files = [f for f in os.listdir(self.model_path) if '.zip' in f and 'last_epoch_' in f]
        # if there's more than one, choose the most recent
        if len(model_files) >= 1:
            model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
            self.model = PPO.load(
                os.path.join(self.model_path, model_files[0]),
                env=env,          # or whatever env you need
                device=self.model.device,
                print_system_info=False
            )        # load the info file to get the best epoch and value
            info_path = os.path.join(self.model_path, f'info_last_epoch_{self.name}.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                last_timestep = info.get('timesteps', 0)
                last_rwd = info.get('last_rollout/ep_rew_mean', 0.0)
                print(f'Restored last train model from step {last_timestep}, with last_mean_reward={last_rwd:.3f}.')
            else:
                raise ValueError(f"Info file not found: {info_path}")
        else:
            raise ValueError("No last model found to restore.")
        return self.model



class EpochTimingCallback(BaseCallback):
    """
    Custom callback to measure and print the time taken for each epoch.
    """
    def __init__(self, verbose=0):
        super(EpochTimingCallback, self).__init__(verbose)
        self.verbose = verbose
        self.epoch_start_time = None
        self.epoch_end_times = []

    def _on_rollout_start(self) -> None:
        """
        Called at the start of a new rollout (beginning of an epoch in PPO).
        """
        self.epoch_start_time = time.time()

    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (end of an epoch in PPO).
        """
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_end_times.append(epoch_time)
            if self.verbose > 0:
                print(f"Epoch {len(self.epoch_end_times)} completed in {epoch_time:.2f} seconds.")

    def _on_training_end(self) -> None:
        """
        Called once the training is completed.
        """
        total_time = sum(self.epoch_end_times)
        avg_time = total_time / len(self.epoch_end_times) if self.epoch_end_times else 0
        if self.verbose > 0:
            print(f"\nTraining completed!")
            print(f"Total training time: {total_time:.2f} seconds.")
            print(f"Average epoch time: {avg_time:.2f} seconds.")

    def _on_step(self) -> bool:
        """
        Dummy implementation of the required abstract method `_on_step`.
        This callback does not act on steps.
        """
        return True
    




class LogToFileCallback(BaseCallback):
    def __init__(self, log_file: str, verbose: int = 0):
        super(LogToFileCallback, self).__init__(verbose)
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.headers_written = False

    def _log_headers(self, headers):
        """Logs headers to the file."""
        with open(self.log_file, "w") as f:
            f.write(";".join(headers) + "\n")
        self.headers_written = True
        return headers

    def _log_values(self, logs):
        """Logs values based on the current headers."""
        with open(self.log_file, "a") as f:
            f.write(";".join(f'{k}:{v}' for k, v in logs.items())+ "\n")

    def _on_rollout_end(self) -> None:
        """Logs after each rollout, when meaningful metrics are available."""
        logs = {key: value for key, value in self.logger.name_to_value.items()}
        if logs:
                self._log_values(logs)

    def _on_step(self) -> bool:
        """Required by BaseCallback. Returns True to allow training to continue."""
        return True
    




class _EvalDepthRewardTracker:
    """Accumulates reward and success breakdowns grouped by label and depth."""

    def __init__(self) -> None:
        self._rewards_by_depth: Dict[str, List[float]] = defaultdict(list)
        self._success_values_by_depth: defaultdict[Tuple[int, str], List[float]] = defaultdict(list)
        self._success_values_by_label: defaultdict[int, List[float]] = defaultdict(list)
        self._last_episode_id: Dict[int, int] = {}

    def __call__(self, infos: List[Dict[str, Any]]) -> None:
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
            self._success_values_by_depth[(label_value, depth_key)].append(1.0 if success_flag else 0.0)
            self._success_values_by_label[label_value].append(1.0 if success_flag else 0.0)

            if label_value != 1:
                continue
            reward = episode_data.get("r")
            if reward is None:
                continue
            self._rewards_by_depth[depth_key].append(float(reward))

    def metrics(self) -> Dict[str, Union[float, int, str]]:
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
