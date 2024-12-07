import numpy as np
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy

# from train import eval_test
from model_SB3 import eval_test
from utils import Term

# def evaluate_policy(
#     model: "type_aliases.PolicyPredictor",
#     env: Union[gym.Env, VecEnv],
#     n_eval_episodes: int = 10,
#     deterministic: bool = True,
#     render: bool = False,
#     callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
#     reward_threshold: Optional[float] = None,
#     return_episode_rewards: bool = False,
#     warn: bool = True,
# ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
#     """
#     Runs policy for ``n_eval_episodes`` episodes and returns average reward.
#     If a vector env is passed in, this divides the episodes to evaluate onto the
#     different elements of the vector env. This static division of work is done to
#     remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
#     details and discussion.

#     .. note::
#         If environment has not been wrapped with ``Monitor`` wrapper, reward and
#         episode lengths are counted as it appears with ``env.step`` calls. If
#         the environment contains wrappers that modify rewards or episode lengths
#         (e.g. reward scaling, early episode reset), these will affect the evaluation
#         results as well. You can avoid this by wrapping environment with ``Monitor``
#         wrapper before anything else.

#     :param model: The RL agent you want to evaluate. This can be any object
#         that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
#         or policy (``BasePolicy``).
#     :param env: The gym environment or ``VecEnv`` environment.
#     :param n_eval_episodes: Number of episode to evaluate the agent
#     :param deterministic: Whether to use deterministic or stochastic actions
#     :param render: Whether to render the environment or not
#     :param callback: callback function to do additional checks,
#         called after each step. Gets locals() and globals() passed as parameters.
#     :param reward_threshold: Minimum expected reward per episode,
#         this will raise an error if the performance is not met
#     :param return_episode_rewards: If True, a list of rewards and episode lengths
#         per episode will be returned instead of the mean.
#     :param warn: If True (default), warns user about lack of a Monitor wrapper in the
#         evaluation environment.
#     :return: Mean reward per episode, std of reward per episode.
#         Returns ([float], [int]) when ``return_episode_rewards`` is True, first
#         list containing per-episode rewards and second containing per-episode lengths
#         (in number of steps).
#     """
#     is_monitor_wrapped = False
#     # Avoid circular import
#     from stable_baselines3.common.monitor import Monitor

#     if not isinstance(env, VecEnv):
#         env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

#     is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

#     if not is_monitor_wrapped and warn:
#         warnings.warn(
#             "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
#             "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
#             "Consider wrapping environment first with ``Monitor`` wrapper.",
#             UserWarning,
#         )

#     n_envs = env.num_envs
#     print('n_envs:', n_envs)
#     episode_rewards = []
#     episode_lengths = []

#     episode_counts = np.zeros(n_envs, dtype="int")
#     # Divides episodes among different sub environments in the vector as evenly as possible
#     episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

#     current_rewards = np.zeros(n_envs)
#     current_lengths = np.zeros(n_envs, dtype="int")
#     print('ressetting evallllll')
#     observations = env.reset()
#     states = None
#     episode_starts = np.ones((env.num_envs,), dtype=bool)
#     while (episode_counts < episode_count_targets).any():
#         print('starting eval')
#         actions, states = model.predict(
#             observations,  # type: ignore[arg-type]
#             state=states,
#             episode_start=episode_starts,
#             deterministic=deterministic,
#         )
#         print('000')
#         new_observations, rewards, dones, infos = env.step(actions)
#         current_rewards += rewards
#         current_lengths += 1
#         for i in range(n_envs):
#             print('111')
#             if episode_counts[i] < episode_count_targets[i]:
#                 # unpack values so that the callback can access the local variables
#                 reward = rewards[i]
#                 done = dones[i]
#                 info = infos[i]
#                 episode_starts[i] = done

#                 if callback is not None:
#                     callback(locals(), globals())

#                 if dones[i]:
#                     print('done')
#                     if is_monitor_wrapped:
#                         # Atari wrapper can send a "done" signal when
#                         # the agent loses a life, but it does not correspond
#                         # to the true end of episode
#                         if "episode" in info.keys():
#                             # Do not trust "done" with episode endings.
#                             # Monitor wrapper includes "episode" key in info if environment
#                             # has been wrapped with it. Use those rewards instead.
#                             episode_rewards.append(info["episode"]["r"])
#                             episode_lengths.append(info["episode"]["l"])
#                             # Only increment at the real end of an episode
#                             episode_counts[i] += 1
#                     else:
#                         episode_rewards.append(current_rewards[i])
#                         episode_lengths.append(current_lengths[i])
#                         episode_counts[i] += 1
#                     current_rewards[i] = 0
#                     current_lengths[i] = 0

#         observations = new_observations

#         if render:
#             env.render()
#         print('ending eval')

#     mean_reward = np.mean(episode_rewards)
#     std_reward = np.std(episode_rewards)
#     if reward_threshold is not None:
#         assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
#     if return_episode_rewards:
#         return episode_rewards, episode_lengths
#     return mean_reward, std_reward

def evaluate_custom(queries: List[Term],
                    labels: List[bool],
                    env: gym.Env,
                    model: "type_aliases.PolicyPredictor",
                    deterministic: bool = True,):
    rewards, episode_len = eval_test(queries,labels,env,model,deterministic=deterministic)
    return rewards, episode_len

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
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
        super().__init__(callback_after_eval, verbose=verbose)

        self.name = name
        self.best_epoch = None
        self.best_model_state_dict = None
        self.best_value = None

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
        # Logs will be written in ``evaluations.npz``
        # if log_path is not None:
        #     log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.model_path:
            os.makedirs(self.model_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
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

                # np.savez(
                #     self.log_path,
                #     timesteps=self.evaluations_timesteps,
                #     results=self.evaluations_results,
                #     ep_lengths=self.evaluations_length,
                #     **kwargs,
                # )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
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
            # self.logger.dump(self.num_timesteps)
            # remove keys that contain the word "eval" or "total_timesteps" instead of using .dump
            # logs = {key: value for key, value in self.logger.name_to_value.items()}
            # print('logs eval:', ', '.join([":".join((str(k), str(np.round(v, 3)))) for k, v in logs.items()]))
            self.logger.name_to_value = {k: v for k, v in self.logger.name_to_value.items() if 
                                         "eval" not in k and "total_timesteps" not in k}


            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = float(mean_reward)
                if self.verbose >= 1:
                    print("New best mean reward!")

                if self.model_path:
                    self.model.save(os.path.join(self.model_path, f"best_eval_{self.name}.zip"))
                    self.write_info()

                self.best_epoch = self.num_timesteps
                self.best_value = self.best_mean_reward
                self.best_model_state_dict = self.model.policy.state_dict()
                    
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.log_path:
                self._log_values({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward, "eval/mean_ep_length": mean_ep_length, 
                              "eval/std_ep_length": std_ep_length, "eval/num_timesteps": self.num_timesteps})
                

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

    def _log_values(self, logs):
        """Logs values based on the current headers."""
        with open(self.log_path, "a") as f:
            f.write(";".join(f'{k}:{v}' for k, v in logs.items())+ "\n")

    def write_info(self):
        """Save checkpoint metadata."""
        if not self.model_path:
            return
        info = {
            'best_value': float(self.best_mean_reward),
            'epoch': self.n_calls,
            'metric': 'best_mean_reward',
            'num_timesteps': self.num_timesteps,
        }
        info_path = os.path.join(self.model_path,f'eval_info_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_best_ckpt(self):
        """Restore the best model."""
        if self.best_epoch: # use best model from best_model
            if self.best_model_state_dict is not None:
                # Copy model parameters
                self.model.policy.load_state_dict(self.best_model_state_dict)
                # Copy optimizer state if needed
                if hasattr(self.model, 'optimizer'):
                    raise NotImplementedError("Optimizer state restoration is not implemented yet (there shuold be none)")

                print(f'Restored best model from step {self.best_epoch}, with best_mean_reward={self.best_value:.3f}.')
        else:
            print(f'No best model found for {self.name}.')




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
        # print('logs:', ', '.join([":".join((str(k), str(np.round(v, 3)))) for k, v in logs.items()]))
        if logs:
            # if not self.headers_written:
            #     self._log_headers(logs)
            # else:
                self._log_values(logs)

    def _on_step(self) -> bool:
        """Required by BaseCallback. Returns True to allow training to continue."""
        return True
    

from stable_baselines3.common.callbacks import BaseCallback
import sys
import json
import os
import copy


class SB3ModelCheckpoint(BaseCallback):
    """Callback to save SB3 model weights when a monitored metric improves."""

    def __init__(
        self,
        model,
        monitor: str = 'train/loss',
        model_path: Optional[str] = None,
        log_path: Optional[str] = None,
        save_best_only: bool = True,
        maximize: bool = False,
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
        self.best_model_state_dict = None
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
            # if not self.headers_written:
            #     self._log_headers(logs)
            # else:
                self._log_values(logs)


        if self.monitor not in logs:
            print(f'Metric "{self.monitor}" not found. Available: {logs.keys()}')
            return
        current_value = logs[self.monitor]
        improved = (self.maximize and current_value > self.best_value) or \
                   (not self.maximize and current_value < self.best_value)

        if improved:
            # Update best value
            self.best_value = current_value
            self.best_epoch = self.num_timesteps
            self.best_model_state_dict = copy.deepcopy(self.model_.policy.state_dict())
            if self.verbose:
                print(f'Improved {self.monitor} to {current_value:.4f}')

            # Save model
            if self.model_path:
                self.model_.save(os.path.join(self.model_path, f"best_train_{self.name}.zip"))
                self.write_info()
    
    def _on_training_end(self) -> bool:
        # Write the completion message to the info file
        self.write_info()

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

    def write_info(self):
        """Save checkpoint metadata."""
        if not self.model_path:
            return
        info = {
            'best_value': float(self.best_value),
            'epoch': self.best_epoch,
            'metric': self.monitor,
            'maximize': self.maximize,
            'timesteps': self.num_timesteps,
            'finished_train': self.num_timesteps >= self.total_steps,
        }
        info_path = os.path.join(self.model_path,f'train_info_{self.name}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def restore_best_ckpt(self):
        """Restore the best model."""
        if self.best_epoch: # use best model from best_model
            if self.best_model_state_dict is not None:
                # Copy model parameters
                self.model_.policy.load_state_dict(self.best_model_state_dict)
                # Copy optimizer state if needed
                if hasattr(self.model_, 'optimizer'):
                    raise NotImplementedError("Optimizer state restoration is not implemented yet (there shuold be none)")

                print(f'Restored best model from step {self.best_epoch}, with {self.monitor}={self.best_value:.3f}.')
        else:
            print(f'No best model found for {self.name}.')

