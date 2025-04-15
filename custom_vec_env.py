import numpy as np
import torch # Make sure torch is imported if used within step/reset
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from stable_baselines3.common.vec_env.base_vec_env import tile_images # For render
import gymnasium as gym
# Assuming your LogicEnv_gym_batch returns obs, rew, term, trunc, info
# where obs is dict {'key': tensor(bs, ...)}, rew is scalar, term is bool, trunc is bool, info is dict.
# If env returns (bs,) arrays for rew/term/trunc, modifications needed here or in env.


import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info


class BatchedDummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fn: Callable[[], gym.Env], batch_size: int = 1):
        self.env = _patch_env(env_fn())
        self.bs = batch_size
        super().__init__(1, self.env.observation_space, self.env.action_space)
        obs_space = self.env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.bs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.bs,), dtype=bool)
        self.buf_rews = np.zeros((self.bs,), dtype=np.float32)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.bs)]
        self.metadata = self.env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # self.actions shape is (batch_size, *action_shape)
        # We expect the env's step function to handle this batch.
        # Expected return types:
        # obses: (batch_size, *obs_shape) or Dict[str, (batch_size, *)]
        # rewards: (batch_size,)
        # terminateds: (batch_size,)
        # truncateds: (batch_size,)
        # infos: Dict[str, Any] - Often contains info arrays like _final_observation
        obses, rewards, terminateds, truncateds, infos = self.env.step(self.actions)

        # Update buffers directly with batched results
        self.buf_rews = np.asarray(rewards, dtype=np.float32) # Ensure float32
        self.buf_dones = np.logical_or(terminateds, truncateds)
        self._save_obs(obses) # Save the entire batch of observations

        if all(self.buf_dones):
            # If all environments are done, we need to reset them
            # Note: This assumes the env's reset method can handle batch resets
            # save final observation where user can get it, then reset 
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]



class BatchedDummyVecEnv_old(VecEnv):
    """
    A DummyVecEnv wrapper designed for the case where n_envs=1, but the
    single underlying environment expects a batch of actions (shape `(bs,)`)
    and returns observations with a batch dimension (e.g., {'key': tensor(bs, ...)}).

    It passes the full action array to the env's step method.
    It assumes the env's step/reset methods return observations matching
    the env's observation_space (which includes the batch dim) and
    scalar reward, bool terminated, bool truncated, dict info, matching the
    standard single-environment Gym API return signature.
    """
    def __init__(self, env_fns):
        if len(env_fns) != 1:
            raise ValueError("BatchedDummyVecEnv is strictly for n_envs=1.")
        super().__init__(env_fns)

        # Verify the underlying env has the expected batch dimension in its observation space
        self.internal_batch_size = -1
        first_space = next(iter(self.observation_space.spaces.values()))
        if isinstance(first_space, gym.spaces.Box):
             if len(first_space.shape) > 0:
                 self.internal_batch_size = first_space.shape[0]
                 print(f"BatchedDummyVecEnv inferred internal batch size: {self.internal_batch_size}")
             else:
                 raise ValueError("Observation space values must have shapes including the batch dimension.")
        else:
             # Add checks for other space types if necessary
             raise ValueError("Observation space values must be gym.spaces.Box for shape checking.")

        if self.internal_batch_size <= 0:
             raise ValueError("Could not infer internal batch size from observation space shape.")

        # Check if env has batch_size attribute and if it matches
        try:
            env_bs = self.envs[0].batch_size
            if env_bs != self.internal_batch_size:
                print(f"Warning: Env internal batch size ({env_bs}) does not match "
                      f"obs space batch dim ({self.internal_batch_size}). Using obs space dim.")
        except AttributeError:
            print("Warning: Underlying env does not have 'batch_size' attribute for cross-checking.")


    def step_wait(self) -> VecEnvStepReturn:
        # Overrides step_wait specifically for the n_envs=1 case

        # self.actions should have shape (internal_batch_size,) from the policy
        if self.actions.shape != (self.internal_batch_size,):
            raise ValueError(
                f"BatchedDummyVecEnv received actions with unexpected shape {self.actions.shape}. "
                f"Expected ({self.internal_batch_size},)."
            )

        env_idx = 0 # Only one environment

        # Pass the ENTIRE action array self.actions to the underlying env
        # Env step should return: obs_dict(bs,...), reward(scalar), term(bool), trunc(bool), info(dict)
        obs, reward, terminated, truncated, info = self.envs[env_idx].step(self.actions)

        # Store the scalar reward and boolean dones directly
        self.buf_rews[env_idx] = reward
        self.buf_dones[env_idx] = terminated or truncated # VecEnv uses combined done

        # Store the info dict
        # Add TimeLimit.truncated info if truncation occurred
        # See Monitor wrapper logic: info["TimeLimit.truncated"] = truncated and not terminated
        info["TimeLimit.truncated"] = truncated and not terminated
        self.buf_infos[env_idx] = info

        if self.buf_dones[env_idx]:
            # Store the observation corresponding to the termination/truncation
            # This is crucial for algorithms like PPO which use it for TD calculation
            # Note: obs already contains the state *after* the step leading to termination
            # SB3 Monitor usually handles storing the obs *before* reset if info["TimeLimit.truncated"] is true.
            # Here we store the obs returned by step directly in the info.
            # If PPO needs obs *before* reset on truncation, env logic/monitor might need adjustment.
            self.buf_infos[env_idx]["terminal_observation"] = obs

            # Reset the single environment
            # reset() should return: obs_dict(bs,...), info(dict)
            maybe_info = self.envs[env_idx].reset(seed=self.seeds[env_idx])
            if isinstance(maybe_info, tuple) and len(maybe_info) == 2:
                 reset_obs, reset_info = maybe_info
                 # Add reset info to the buf_infos if needed (usually handled by Monitor)
                 # self.buf_infos[env_idx].update(reset_info) # Or store under a specific key
            else: # Old gym interface? Assume just obs
                 reset_obs = maybe_info

            obs = reset_obs # Use the new observation after reset

        # Save the current observation (either from step or reset)
        # _save_obs expects the observation in the correct format (dict of tensors)
        self._save_obs(env_idx, obs)

        # _obs_from_buf adds the n_envs dimension back (shape (1, bs, ...))
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy())

    # You might also need to override reset if the env's reset returns info
    def reset(self, **kwargs) -> np.ndarray:
         # Simplified reset - assumes env.reset returns obs (dict) or (obs, info)
         self.buf_dones[0] = False # Reset done flag for the single env
         maybe_info = self.envs[0].reset(seed=self._seeds[0], **kwargs)
         if isinstance(maybe_info, tuple) and len(maybe_info) == 2:
              obs, info = maybe_info
              # Store info if needed, e.g., in self.buf_infos[0] or handle otherwise
              # Note: Standard SB3 often expects reset info to be handled by Monitor
         else:
              obs = maybe_info
         self._save_obs(0, obs)
         return self._obs_from_buf()
