import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional, List

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info


class CustomDummyVecEnv(VecEnv):
    """
    A lightweight vectorized env wrapper that stops stepping envs once they
   ’ve done their allotted episodes.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len({id(env.unwrapped) for env in self.envs}) != len(self.envs):
            raise ValueError("Each env_fn must return a fresh env instance")
        env0 = self.envs[0]
        super().__init__(len(env_fns), env0.observation_space, env0.action_space)

        # buffers
        self.keys, shapes, dtypes = obs_space_info(env0.observation_space)
        self.buf_obs = OrderedDict(
            (k, np.zeros((self.num_envs, *shapes[k]), dtype=dtypes[k]))
            for k in self.keys
        )
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.metadata = env0.metadata

        # track per-env episode counts/targets (to be set externally)
        self._episode_count = np.zeros(self.num_envs, dtype=int)
        self._episode_target = np.full(self.num_envs, fill_value=np.inf, dtype=int)

        # only step/reset active envs
        self.active_envs = np.ones(self.num_envs, dtype=bool)
        self.type_ = "custom_dummy"

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for idx in np.nonzero(self.active_envs)[0]:
            obs, rew, terminated, truncated, info = self.envs[idx].step(self.actions[idx])
            done = terminated or truncated
            self.buf_rews[idx] = rew
            self.buf_dones[idx] = done
            info["TimeLimit.truncated"] = truncated and not terminated

            if done:
                # count this episode
                self._episode_count[idx] += 1
                # if reached its target, deactivate
                if self._episode_count[idx] >= self._episode_target[idx]:
                    self.active_envs[idx] = False
                # save final obs & reset
                info["terminal_observation"] = obs
                obs, reset_info = self.envs[idx].reset()
                self.buf_infos[idx] = reset_info
            else:
                self.buf_infos[idx] = info

            # store obs
            if isinstance(obs, dict):
                for k in self.keys:
                    self.buf_obs[k][idx] = obs[k]
            else:
                # assume array
                self.buf_obs[self.keys[0]][idx] = obs

        return self._obs_from_buf(), self.buf_rews.copy(), self.buf_dones.copy(), deepcopy(self.buf_infos)

    def reset(self) -> Any:
        for idx in np.nonzero(self.active_envs)[0]:
            maybe_opts = {"options": self._options[idx]} if self._options[idx] else {}
            obs, info = self.envs[idx].reset(seed=self._seeds[idx], **maybe_opts)
            self.buf_infos[idx] = info
            if isinstance(obs, dict):
                for k in self.keys:
                    self.buf_obs[k][idx] = obs[k]
            else:
                self.buf_obs[self.keys[0]][idx] = obs

        # once reset, clear seeds/options
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
