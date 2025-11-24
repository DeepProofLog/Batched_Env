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

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from .sb3_env import LogicEnv_gym

def create_environments(args, data_handler, index_manager, kge_engine=None, detailed_eval_env=False):
    """
    Creates and seeds the training, evaluation, and callback environments.
    """
    facts_set = set(data_handler.facts)
    shaping_gamma = args.pbrs_gamma if args.pbrs_gamma is not None else args.gamma
    env_verbose = getattr(args, "verbose_env", 0)
    prover_verbose = getattr(args, "verbose_prover", 0)

    def make_env(mode='train', seed=0, queries=None, labels=None, query_depths=None, facts=None, verbose=0, prover_verbose=0, env_id: Optional[int] = None, train_stride: int = 1, use_shared_train_ptr: bool = False):
        def _init():
            env = LogicEnv_gym(
                index_manager=index_manager,
                data_handler=data_handler,
                queries=queries,
                labels=labels,
                query_depths=query_depths,
                facts=facts,
                mode=mode,
                corruption_mode=args.corruption_mode,
                corruption_scheme=args.corruption_scheme,
                train_neg_ratio=args.train_neg_ratio,
                seed=seed,
                max_depth=args.max_depth,
                memory_pruning=args.memory_pruning,
                endt_action=args.endt_action,
                endf_action=args.endf_action,
                skip_unary_actions=args.skip_unary_actions,
                padding_atoms=args.padding_atoms,
                padding_states=args.padding_states,
                device='cpu',
                engine=args.engine,
                train_stride=train_stride,
                initial_train_idx=env_id or 0,
                use_shared_train_ptr=use_shared_train_ptr,
                kge_action=args.kge_action,
                reward_type=args.reward_type,
                shaping_beta=args.pbrs_beta,
                shaping_gamma=shaping_gamma,
                kge_inference_engine=kge_engine,
                canonical_action_order=args.canonical_action_order,
                verbose=verbose,
                prover_verbose=prover_verbose,
            )
            env = Monitor(env)
            return env
        return _init

    ss = np.random.SeedSequence(args.seed_run_i)
    child_seeds = ss.spawn(3)
    rng_env = np.random.Generator(np.random.PCG64(child_seeds[0]))
    rng_eval = np.random.Generator(np.random.PCG64(child_seeds[1]))
    rng_callback = np.random.Generator(np.random.PCG64(child_seeds[2]))

    env_seeds = rng_env.integers(0, 2**10, size=args.n_envs)
    eval_env_seeds = rng_eval.integers(0, 2**10, size=args.n_eval_envs)
    callback_env_seeds = rng_callback.integers(0, 2**10, size=1)

    # Reset shared training pointer for string envs to mirror batched round-robin
    LogicEnv_gym._global_train_ptr = 0
    LogicEnv_gym._global_train_len = len(data_handler.train_queries)

    env_fns = [make_env(
        mode='train',
        seed=int(env_seeds[i]),
        queries=data_handler.train_queries,
        labels=[1] * len(data_handler.train_queries),
        query_depths=data_handler.train_queries_depths,
        facts=facts_set,
        verbose=env_verbose,
        prover_verbose=prover_verbose,
        env_id=i,
        train_stride=1,
        use_shared_train_ptr=True,
    ) for i in range(args.n_envs)]

    eval_env_fns = [make_env(
        mode='eval',
        seed=int(eval_env_seeds[i]),
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        verbose=env_verbose,
        prover_verbose=prover_verbose,
    ) for i in range(args.n_eval_envs)]
    
    callback_env_fns = [make_env(
        mode='eval_with_restart',
        seed=int(callback_env_seeds[i]),
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        verbose=env_verbose,
        prover_verbose=prover_verbose,
    ) for i in range(1)]


    env = DummyVecEnv(env_fns)
    eval_env = CustomDummyVecEnv(eval_env_fns)
    if detailed_eval_env:
        callback_env = CustomDummyVecEnv(callback_env_fns)
    else:
        callback_env = DummyVecEnv(callback_env_fns)

    # use SubprocVecEnv if you want to use multiple processes
    # env = SubprocVecEnv(env_fns)
    # eval_env = SubprocVecEnv(eval_env_fns)
    # callback_env = SubprocVecEnv(callback_env_fns)

    return env, eval_env, callback_env

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
        self._episode_count = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_target = np.full(self.num_envs, fill_value=np.inf, dtype=np.int32)
        self._episode_step_id = np.zeros(self.num_envs, dtype=np.int32)

        # only step/reset active envs
        self.active_envs = np.ones(self.num_envs, dtype=bool)
        self.type_ = "custom_dummy"

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        infos_to_return: List[dict] = [{} for _ in range(self.num_envs)]

        for idx in np.nonzero(self.active_envs)[0]:
            obs, rew, terminated, truncated, info = self.envs[idx].step(self.actions[idx])
            done = terminated or truncated
            self.buf_rews[idx] = rew
            self.buf_dones[idx] = done
            step_info = dict(info)

            if done:
                # count this episode
                self._episode_count[idx] += 1
                self._episode_step_id[idx] += 1
                # if reached its target, deactivate
                if self._episode_count[idx] >= self._episode_target[idx]:
                    self.active_envs[idx] = False
                # save final obs & reset
                step_info["terminal_observation"] = obs
                step_info["episode_idx"] = int(self._episode_step_id[idx])
                obs, reset_info = self.envs[idx].reset()

                # store reset info for next step (without lingering episode data)
                self.buf_infos[idx] = dict(reset_info)
            else:
                self.buf_infos[idx] = info

            # store obs
            if isinstance(obs, dict):
                for k in self.keys:
                    self.buf_obs[k][idx] = obs[k]
            else:
                # assume array
                self.buf_obs[self.keys[0]][idx] = obs

            infos_to_return[idx] = step_info

        # Non-active envs retain previous info buffer without episode metadata
        for idx in np.nonzero(~self.active_envs)[0]:
            infos_to_return[idx] = {}

        # OPTIMIZATION: Use a shallow copy instead of a deepcopy.
        # This is much faster and safe for SB3's training loop.
        return self._obs_from_buf(), self.buf_rews.copy(), self.buf_dones.copy(), infos_to_return

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
        # return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))
        return dict_to_obs(
            self.observation_space,
            {k: v.copy() for k, v in self.buf_obs.items()}
        )

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
