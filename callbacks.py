import numpy as np
import os
import json
from typing import Optional, Union, Any, List, Dict
import time
import sys

import env
from model_eval import evaluate_policy as evaluate_policy_mrr
from model_eval import eval_corruptions as eval_corruptions_mrr
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

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
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.3f} +/- {std_reward:.3f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            self.logger.name_to_value = {k: v for k, v in self.logger.name_to_value.items() if 
                                         "eval" not in k and "total_timesteps" not in k}

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.model_path is not None:
                    self.model.save(os.path.join(self.model_path, f"best_eval_{self.name}.zip"))
                    self.write_info()
                self.best_mean_reward = float(mean_reward)
                self.best_epoch = self.num_timesteps
                
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.log_path:
                self._log_values({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward, "eval/mean_ep_length": mean_ep_length, 
                            "eval/std_ep_length": std_ep_length, "eval/num_timesteps": self.num_timesteps})
                

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            print(f'---------------evaluation finished---------------  took {time.time()-start:.2f} seconds')
        return continue_training


    def _log_values(self, logs):
        """Logs values based on the current headers."""
        with open(self.log_path, "a") as f:
            f.write(";".join(f'{k}:{v}' for k, v in logs.items())+ "\n")

    def write_info(self):
        """Save checkpoint metadata."""
        info = {
            'metric': 'best_mean_reward',
            'best_metric_value': float(self.best_mean_reward),
            'n_calls': self.n_calls,
            'timesteps': self.num_timesteps,
        }
        info_path = os.path.join(self.model_path, f'info_best_eval_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_best_ckpt(self):
        """Restore the best model."""
        if self.best_epoch: # use best model from best_model
            model_files = [f for f in os.listdir(self.model_path) if 'best_eval' in f and 'info' not in f and '.zip' in f]
            if len(model_files) > 1:
                model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
                print('Restoring last model modified:', model_files[0])
            self.model.load(os.path.join(self.model_path, model_files[0]),print_system_info=False)
            print(f'Restored best model from step {self.best_epoch}, with best_mean_reward={self.best_mean_reward:.3f}.')
        else:
            print(f'No best model found for {self.name}.')
    
    def _on_training_end(self) -> bool:
        if self.best_model_save_path:
            self.write_info()       




class SB3ModelCheckpoint(BaseCallback):
    """Callback to save SB3 model weights when a monitored metric improves."""

    def __init__(
        self,
        model,
        monitor: str = "rollout/ep_rew_mean",
        model_path: Optional[str] = None,
        log_path: Optional[str] = None,
        save_best_only: bool = True,
        maximize: bool = True,
        verbose: bool = True,
        frequency: int = 1,
        total_steps: int = 0,
        name: str = None
    ):
        """
        Initialize checkpoint callback.

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
        self.save_best_only = save_best_only
        self.maximize = maximize
        self.frequency = frequency
        self.name = name

        self.best_value = -sys.float_info.max if maximize else sys.float_info.max
        self.best_epoch = None
        # self.best_variance = 0
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        """Called at each environment step. Skips until frequency is reached."""
        # Ensure frequency is handled in episode_end
        return True

    def _on_rollout_end(self) -> None:
        """Check metric and save model at the end of each rollout."""
        # Skip if not checking at this rollout frequency
        # if self.n_calls % self.frequency != 0:
        #     return

        logs = {key: value for key, value in self.logger.name_to_value.items()}
        # print('logs train:', ', '.join([":".join((str(k), str(np.round(v, 3)))) for k, v in logs.items()]))
        if self.log_path and logs:
                self._log_values(logs)


        if self.monitor not in logs:
            print(f'Metric "{self.monitor}" not found. Available: {logs.keys()}')
            return
        current_value = logs[self.monitor]
        # current_variance = logs.get("train/explained_variance", None)
        improved = (self.maximize and current_value > self.best_value) or \
                   (not self.maximize and current_value < self.best_value)

        if improved:
            # Update best value
            self.best_value = current_value
            self.best_epoch = self.num_timesteps
            # reset the best variance every time the best value is updated
            # self.best_variance = logs.get("train/explained_variance", None)
            if self.verbose:
                print(f'Improved {self.monitor} to {current_value:.4f}')

            # Save model
            if self.model_path:
                self.model_.save(os.path.join(self.model_path, f"best_train_{self.name}.zip"))
                self.write_info('best_train')
        
        if self.model_path:
            self.model_.save(os.path.join(self.model_path, f"last_train_{self.name}.zip"))
            self.write_info('last_train')

        # if current_variance is not None:
        #     as_good = (self.maximize and current_value >= self.best_value and current_variance>self.best_variance) or \
        #            (not self.maximize and current_value <= self.best_value and current_variance>self.best_variance)

        #     if as_good:
        #         self.best_variance = current_variance
        #         if self.model_path:
        #             self.model_.save(os.path.join(self.model_path, f"best_variance_{self.name}.zip"))
        #             self.write_info('best_variance')
    
    def _on_training_end(self) -> bool:
        if self.model_path:
            self.model_.save(os.path.join(self.model_path, f"last_epoch_{self.name}.zip"))
            self.write_info('last_epoch')

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

    def write_info(self,name):
        """Save checkpoint metadata."""
        if not self.model_path:
            return
        info = {
            'metric': self.monitor,
            'best_value': float(self.best_value),
            'timesteps': self.num_timesteps,
            'finished_train': self.num_timesteps >= self.total_steps,
        }
        info_path = os.path.join(self.model_path,f'info_{name}_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_best_ckpt(self):
        """Restore the best model."""
        if self.best_epoch: # use best model from best_model
            model_files = [f for f in os.listdir(self.model_path) if self.name in f and '.zip' in f and 'last_epoch_' in f]
            # if there's more than one, choose the most recent
            if len(model_files) > 1:
                model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
                print('Restoring last model modified:', model_files[0])
            self.model.load(os.path.join(self.model_path, model_files[0]),print_system_info=True)
            print(f'Restored best model from step {self.best_epoch}, with best_mean_reward={self.best_value:.3f}.')
        else:
            print(f'No best model found for {self.name}.')



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
    













class CustomEvalCallbackMRR(CustomEvalCallback):
    """
    Callback for evaluating an agent, including MRR calculation.
    Inherits from CustomEvalCallback and overrides the _on_step method
    to incorporate MRR evaluation using a provided sampler and data.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        sampler: Any,  # Sampler for generating negative samples
        eval_data: List[Any],  # Data for MRR evaluation (positive queries)
        n_corruptions: Optional[int] = None, # Number of negative corruptions per positive
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
        self.sampler = sampler
        self.eval_data = eval_data
        self.n_corruptions = n_corruptions
        self.consult_janus = consult_janus
        self.best_mean_mrr = -np.inf # Track best MRR for saving best model based on MRR

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

            # Perform MRR evaluation
            mrr_eval_results = eval_corruptions_mrr(
                self.model,
                self.eval_env,
                self.eval_data,
                self.sampler,
                n_corruptions=self.n_corruptions,
                deterministic=self.deterministic,
                verbose=self.verbose,
                consult_janus=self.consult_janus,
            )

            mean_mrr = mrr_eval_results.get('mrr_mean', 0.0)
            mean_reward_pos = mrr_eval_results.get('rewards_pos_mean', 0.0)
            mean_reward_neg = mrr_eval_results.get('rewards_neg_mean', 0.0)
            hits1 = mrr_eval_results.get('hits1_mean', 0.0)
            hits3 = mrr_eval_results.get('hits3_mean', 0.0)
            hits10 = mrr_eval_results.get('hits10_mean', 0.0)
            auc_pr = mrr_eval_results.get('auc_pr', 0.0)

            print(f"Eval num_timesteps={self.num_timesteps}")
            print(f"MRR: {mean_mrr:.4f}")
            print(f"Rewards Positive: {mean_reward_pos:.4f}, Negative: {mean_reward_neg:.4f}")
            print(f"Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
            print(f"AUC-PR: {auc_pr:.4f}")

            # Record MRR and other metrics to the logger
            self.logger.record("eval/mrr_mean", float(mean_mrr))
            self.logger.record("eval/rewards_pos_mean", float(mean_reward_pos))
            self.logger.record("eval/rewards_neg_mean", float(mean_reward_neg))
            self.logger.record("eval/hits1_mean", float(hits1))
            self.logger.record("eval/hits3_mean", float(hits3))
            self.logger.record("eval/hits10_mean", float(hits10))
            self.logger.record("eval/auc_pr", float(auc_pr))
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            # Original reward-based evaluation (if still desired, though MRR is primary)
            # You might want to remove or adjust this if MRR is the sole metric
            # Or you could consider one of these as the primary metric for saving the best model
            # For simplicity, let's keep the best model saving based on MRR for CustomEvalCallbackMRR
            if mean_mrr > self.best_mean_mrr:
                print("New best mean MRR!")
                if self.model_path is not None:
                    # Save the model with a clear indicator for MRR
                    self.model.save(os.path.join(self.model_path, f"best_eval_mrr_{self.name}.zip"))
                    # Update info with MRR metrics
                    self.write_mrr_info(mrr_eval_results)
                self.best_mean_mrr = float(mean_mrr)
                self.best_epoch = self.num_timesteps
                
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.log_path:
                # Log all relevant metrics, including MRR ones
                log_metrics = {
                    "eval/mrr_mean": mean_mrr,
                    "eval/rewards_pos_mean": mean_reward_pos,
                    "eval/rewards_neg_mean": mean_reward_neg,
                    "eval/hits1_mean": hits1,
                    "eval/hits3_mean": hits3,
                    "eval/hits10_mean": hits10,
                    "eval/auc_pr": auc_pr,
                    "eval/num_timesteps": self.num_timesteps,
                }
                self._log_values(log_metrics)
                

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            print(f'---------------evaluation finished---------------  took {time.time()-start:.2f} seconds')
            # print(aaaaaa)
        return continue_training

    def write_mrr_info(self, mrr_results: Dict[str, Any]):
        """Save checkpoint metadata with MRR specific information."""
        info = {
            'metric': 'best_mean_mrr',
            'best_metric_value': float(self.best_mean_mrr),
            'n_calls': self.n_calls,
            'timesteps': self.num_timesteps,
            'mrr_results': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in mrr_results.items()}
        }
        info_path = os.path.join(self.model_path, f'info_best_eval_mrr_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_best_ckpt(self):
        """Restore the best model based on MRR."""
        if self.best_epoch:
            model_files = [f for f in os.listdir(self.model_path) if f'best_eval_mrr_{self.name}' in f and '.zip' in f]
            if len(model_files) > 1:
                model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
                print('Restoring last model modified based on MRR:', model_files[0])
            if model_files:
                self.model.load(os.path.join(self.model_path, model_files[0]), print_system_info=False)
                print(f'Restored best model from step {self.best_epoch}, with best_mean_mrr={self.best_mean_mrr:.3f}.')
            else:
                print(f'No best MRR model found for {self.name}.')
        else:
            print(f'No best MRR model found for {self.name}.')

    def _on_training_end(self) -> bool:
        # Override to save the info file for the MRR-based best model
        if self.best_model_save_path:
            # Need to re-evaluate or store the last MRR results to save them correctly at the end
            # For simplicity, we just call write_mrr_info if a best MRR model was found during training.
            # In a more robust implementation, you might want to perform a final MRR eval here.
            if self.best_mean_mrr > -np.inf: # Only if at least one evaluation happened
                 # This would ideally take the last computed mrr_eval_results
                 # For now, we'll just use the best_mean_mrr stored.
                 # To save full results, mrr_eval_results would need to be stored as a class member
                 # or re-computed.
                self.write_mrr_info({'mrr_mean': self.best_mean_mrr})
        return True