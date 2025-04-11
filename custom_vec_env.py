import numpy as np
import torch # Make sure torch is imported if used within step/reset
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from stable_baselines3.common.vec_env.base_vec_env import tile_images # For render
import gymnasium as gym
# Assuming your LogicEnv_gym_batch returns obs, rew, term, trunc, info
# where obs is dict {'key': tensor(bs, ...)}, rew is scalar, term is bool, trunc is bool, info is dict.
# If env returns (bs,) arrays for rew/term/trunc, modifications needed here or in env.

class BatchedDummyVecEnv(DummyVecEnv):
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
