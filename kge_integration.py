# FILE NAME: kge_integration.py. Contains helper functions for KGE integration strategies.
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple
from stable_baselines3 import PPO
from kge_inference import KGEInference
from index_manager import IndexManager
from model import CustomActorCriticPolicy

def _init_kge_engine(args: Any) -> Optional[KGEInference]:
    """Create KGE inference engine when requested."""
    needs_engine = (
        args.use_kge_action
        or args.kge_integration_strategy in {"sum_eval", "logit_shaping", "kickstart"}
        or float(getattr(args, "pbrs_beta", 0.0)) != 0.0
        or getattr(args, "policy_init", "none") in {"warm_start", "kl_reg"}
    )
    if needs_engine:
        print("\nInitializing KGE Inference Engine...", flush=True)
        kge_engine_backend = getattr(args, "kge_engine", "tf")
        engine = KGEInference(
            dataset_name=args.dataset_name,
            base_path=args.data_path,
            checkpoint_dir=args.kge_checkpoint_dir,
            run_signature=args.kge_run_signature,
            seed=0,
            scores_file_path=args.kge_scores_file,
            backend=kge_engine_backend,
        )
        print("KGE Inference Engine Initialized.\n")
        return engine
    return None


def _attach_kge_to_policy(
    model: PPO,
    im: IndexManager,
    engine: Optional[KGEInference],
    device: torch.device,
    use_kge_action: bool,
    integration_strategy: Optional[str],
    policy_init: str,
) -> None:
    policy = model.policy
    policy.kge_integration_strategy = integration_strategy
    needs_kge_attachment = (
        integration_strategy in {"train", "train_bias", "logit_shaping", "kickstart"}
        or policy_init in {"warm_start", "kl_reg"}
    )

    if hasattr(policy, "kge_fusion_mlp") and policy.kge_fusion_mlp is not None:
        policy.kge_fusion_mlp.to(device)

    if not needs_kge_attachment or engine is None or im is None:
        policy.kge_inference_engine = None
        policy.index_manager = None
        empty_indices = torch.empty(0, dtype=torch.int32, device=device)
        policy.kge_indices_tensor = empty_indices
        return

    policy.kge_inference_engine = engine
    policy.index_manager = im
    if use_kge_action and integration_strategy in {"train", "train_bias"}:
        kge_indices = [idx for pred, idx in im.predicate_str2idx.items() if pred.endswith("_kge")]
        policy.kge_indices_tensor = torch.tensor(kge_indices, device=device, dtype=torch.int32)
    else:
        policy.kge_indices_tensor = torch.empty(0, dtype=torch.int32, device=device)


# ------------------------------
# Policy warm-start helpers
# ------------------------------

def _split_vec_obs(obs: Any) -> List[dict]:
    """Split a vectorized observation into a list of per-env dicts."""
    if isinstance(obs, dict):
        first_key = next(iter(obs))
        num_envs = obs[first_key].shape[0]
        samples: List[dict] = []
        for idx in range(num_envs):
            samples.append({k: np.array(v[idx], copy=True) for k, v in obs.items()})
        return samples

    arr = np.array(obs, copy=True)
    if arr.ndim == 0:
        return [arr]
    return [np.array(arr[idx], copy=True) for idx in range(arr.shape[0])]


def _collect_observations(env, steps: int) -> List[dict]:
    """Collect a list of raw observations by rolling the environment."""
    if steps <= 0:
        return []

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    collected: List[dict] = []
    while len(collected) < steps:
        collected.extend(_split_vec_obs(obs))
        if len(collected) >= steps:
            break

        raw_action = env.action_space.sample()
        if isinstance(raw_action, np.ndarray):
            actions = np.repeat(raw_action[None, ...], env.num_envs, axis=0)
        else:
            actions = np.full((env.num_envs,), raw_action)

        obs, _, _, _ = env.step(actions)

    return collected[:steps]


def _obs_collate(obs_batch: List[dict], device: torch.device) -> dict:
    """Stack a batch of dict observations into torch tensors on the target device."""
    if not obs_batch:
        return {}

    keys = ("sub_index", "derived_sub_indices", "action_mask")
    batch = {}
    for key in keys:
        tensors = []
        for sample in obs_batch:
            value = sample[key]
            tensor = torch.as_tensor(value)
            tensors.append(tensor)
        stacked = torch.stack(tensors, dim=0)
        if key == "action_mask":
            stacked = stacked.to(dtype=torch.bool)
        elif key in {"sub_index", "derived_sub_indices"}:
            stacked = stacked.to(dtype=torch.long)
        elif stacked.dtype == torch.float64:
            stacked = stacked.to(dtype=torch.float32)
        batch[key] = stacked.to(device)
    return batch


@torch.no_grad()
def _build_teacher_logits_from_kge(
    policy: CustomActorCriticPolicy,
    obs: dict,
    ref_logits: torch.Tensor,
    index_manager: Optional[IndexManager],
    kge_engine: Optional[KGEInference],
    temperature: float,
) -> torch.Tensor:
    """Create teacher logits aligned with the student's action logits using KGE scores."""
    device, dtype = ref_logits.device, ref_logits.dtype
    mask = obs.get("action_mask")
    if mask is None:
        return torch.full_like(ref_logits, float("-inf"))
    mask_bool = mask.to(device=device, dtype=torch.bool)

    teacher = torch.full_like(ref_logits, float("-inf"))
    if policy is None or kge_engine is None or index_manager is None:
        teacher = teacher.masked_fill(mask_bool, 0.0)
    else:
        gather_fn = getattr(policy, "_gather_kge_candidates", None)
        transform_fn = getattr(policy, "_transform_kge_scores", None)
        derived = obs.get("derived_sub_indices")
        if gather_fn is None or transform_fn is None or derived is None:
            teacher = teacher.masked_fill(mask_bool, 0.0)
        else:
            candidates, atom_list = gather_fn(derived, mask_bool)
            if not atom_list:
                teacher = teacher.masked_fill(mask_bool, 0.0)
            else:
                predictor = getattr(kge_engine, "predict_batch", None)
                raw_scores = predictor(atom_list) if predictor is not None else None
                if raw_scores is None:
                    score_tensor = torch.zeros(len(atom_list), device=device, dtype=dtype)
                else:
                    score_tensor = transform_fn(raw_scores, device=device, dtype=dtype)
                for atom_key, score in zip(atom_list, score_tensor):
                    for b_idx, a_idx in candidates[atom_key]:
                        teacher[b_idx, a_idx] = score

    teacher = teacher.masked_fill(~mask_bool, float("-inf"))
    if temperature != 1.0:
        teacher = teacher / max(temperature, 1e-8)
    return teacher


def _forward_kl_teacher_student(student_logits: torch.Tensor, teacher_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute forward KL divergence KL(teacher || student)."""
    t_logp = F.log_softmax(teacher_logits, dim=1)
    t_prob = torch.clamp(torch.exp(t_logp), min=eps)
    s_logp = F.log_softmax(student_logits, dim=1)
    kl = torch.sum(t_prob * (t_logp - s_logp), dim=1)
    return torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0).mean()


def policy_bc_warmstart(
    model: PPO,
    callback_env,
    index_manager: Optional[IndexManager],
    kge_engine: Optional[KGEInference],
    *,
    steps: int,
    batch_size: int,
    collect_steps: int,
    freeze_value: bool,
    temperature: float,
    verbose: int = 1,
) -> None:
    """Run a short behavior-cloning phase to initialize the policy from the KGE teacher."""
    if kge_engine is None or index_manager is None:
        print("[BC warmstart] Skipped: KGE teacher not available.")
        return

    device = model.device
    policy: CustomActorCriticPolicy = model.policy  # type: ignore[assignment]
    optimizer = policy.optimizer
    if optimizer is None:
        print("[BC warmstart] Skipped: policy optimizer not initialized.")
        return

    obs_pool = _collect_observations(callback_env, collect_steps)
    if not obs_pool:
        print("[BC warmstart] Skipped: no observations collected.")
        return

    frozen_params: List[torch.nn.Parameter] = []
    if freeze_value and hasattr(policy.mlp_extractor, "value_network"):
        for param in policy.mlp_extractor.value_network.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
                frozen_params.append(param)

    policy.set_training_mode(True)
    policy.train()

    report_every = max(1, steps // 10)
    for step_idx in range(steps):
        sample_size = min(batch_size, len(obs_pool))
        replace = len(obs_pool) < batch_size
        indices = np.random.choice(len(obs_pool), size=sample_size, replace=replace)
        mini_batch = [obs_pool[int(i)] for i in indices]
        batch = _obs_collate(mini_batch, device)
        if not batch:
            continue

        dist = policy.get_distribution(batch)
        student_logits = getattr(dist.distribution, "logits", None)
        if student_logits is None:
            raise ValueError("Policy distribution does not expose logits for warm-start.")

        teacher_logits = _build_teacher_logits_from_kge(
            policy,
            batch,
            student_logits,
            index_manager,
            kge_engine,
            temperature,
        )

        mask = batch.get("action_mask")
        if mask is None:
            continue
        mask_bool = mask.to(dtype=torch.bool)
        teacher_logits = teacher_logits.masked_fill(~mask_bool, float("-inf"))

        valid_rows = torch.isfinite(teacher_logits).any(dim=1)
        if not valid_rows.any():
            continue

        loss = _forward_kl_teacher_student(student_logits[valid_rows], teacher_logits[valid_rows])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
        optimizer.step()

        if verbose and (step_idx % report_every == 0 or step_idx == steps - 1):
            print(
                f"[BC warmstart] step={step_idx + 1:5d}/{steps}  kl={float(loss):.4f}  pool={len(obs_pool)}"
            )

    for param in frozen_params:
        param.requires_grad_(True)


# ------------------------------
# Critic warm-start helpers
# ------------------------------

@torch.no_grad()
def _phi_from_obs(
    policy: CustomActorCriticPolicy,
    obs_dict: dict,
    index_manager: Optional[IndexManager],
    kge_engine: Optional[KGEInference],
) -> float:
    """Compute a simple KGE-based potential Phi(s) for a single observation."""
    if policy is None or index_manager is None or kge_engine is None:
        return 0.0

    device = policy.device
    batch = _obs_collate([obs_dict], device)
    action_mask = batch.get("action_mask")
    if action_mask is None or action_mask.numel() == 0:
        return 0.0

    dummy_logits = torch.zeros((1, action_mask.shape[1]), device=device, dtype=torch.float32)
    teacher_logits = _build_teacher_logits_from_kge(
        policy,
        batch,
        dummy_logits,
        index_manager,
        kge_engine,
        temperature=1.0,
    )
    finite_logits = torch.isfinite(teacher_logits)
    if not finite_logits.any():
        return 0.0

    safe_logits = torch.where(
        finite_logits,
        teacher_logits,
        torch.full_like(teacher_logits, -1e9),
    )
    phi_value = safe_logits.max(dim=1).values
    return float(phi_value.item())


def _calibrate_affine(
    env,
    policy: CustomActorCriticPolicy,
    index_manager: Optional[IndexManager],
    kge_engine: Optional[KGEInference],
    *,
    n_states: int = 512,
    rollouts_per_state: int = 3,
    horizon: int = 8,
    gamma: float = 0.99,
) -> Tuple[float, float]:
    """Fit an affine transform V0(s) = a * Phi(s) + b via short rollouts."""
    if env is None or getattr(env, "num_envs", 1) != 1:
        print("[Critic warmstart] Calibrate requires single-environment VecEnv. Using default scale.")
        return 1.0, 0.0

    if rollouts_per_state > 1:
        print("[Critic warmstart] Multiple rollouts per state not supported; using a single rollout.")
    rollout_reps = 1

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    states_collected = 0
    features: List[List[float]] = []
    targets: List[List[float]] = []

    while states_collected < n_states:
        obs_list = _split_vec_obs(obs)
        if not obs_list:
            obs = env.reset()
            continue
        obs_dict = obs_list[0]
        phi_s = _phi_from_obs(policy, obs_dict, index_manager, kge_engine)

        returns: List[float] = []
        working_obs = obs
        for _ in range(rollout_reps):
            current_obs = working_obs
            discounted_return = 0.0
            discount = 1.0
            done = False
            for _ in range(horizon):
                batch = _obs_collate(_split_vec_obs(current_obs), policy.device)
                if not batch:
                    done = True
                    break
                dist = policy.get_distribution(batch)
                actions_tensor = dist.get_actions()
                actions = actions_tensor.detach().cpu().numpy()
                if actions.ndim == 0:
                    actions = actions.reshape(1)
                next_obs, rewards, dones, _ = env.step(actions)
                discounted_return += discount * float(np.mean(rewards))
                discount *= gamma
                current_obs = next_obs
                if np.any(dones):
                    done = True
                    break
            returns.append(discounted_return)
            working_obs = env.reset() if done else current_obs
            if isinstance(working_obs, tuple):
                working_obs = working_obs[0]

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        if returns:
            features.append([phi_s, 1.0])
            targets.append([float(np.mean(returns))])
            states_collected += 1

    X = torch.tensor(features, dtype=torch.float32, device=policy.device)
    Y = torch.tensor(targets, dtype=torch.float32, device=policy.device)

    if X.shape[0] == 0:
        return 1.0, 0.0

    try:
        result = torch.linalg.lstsq(X, Y)
        theta = result.solution
    except AttributeError:
        theta, _ = torch.lstsq(Y, X)

    theta = theta.reshape(-1)
    a = float(theta[0].item()) if theta.numel() > 0 else 1.0
    b = float(theta[1].item()) if theta.numel() > 1 else 0.0
    return a, b


def critic_warmstart(
    model: PPO,
    callback_env,
    index_manager: Optional[IndexManager],
    kge_engine: Optional[KGEInference],
    *,
    steps: int,
    collect_steps: int,
    calibrate: bool,
    rollouts_per_state: int,
    horizon: int,
    gamma: float,
    verbose: int = 1,
) -> None:
    """Initialize critic by fitting value estimates to a KGE-derived potential."""
    if kge_engine is None or index_manager is None:
        print("[Critic warmstart] Skipped: KGE teacher not available.")
        return

    device = model.device
    policy: CustomActorCriticPolicy = model.policy  # type: ignore[assignment]
    optimizer = policy.optimizer
    if optimizer is None:
        print("[Critic warmstart] Skipped: policy optimizer not initialized.")
        return

    obs_pool = _collect_observations(callback_env, collect_steps)
    if not obs_pool:
        print("[Critic warmstart] Skipped: no observations collected.")
        return

    phi_vals = [_phi_from_obs(policy, obs, index_manager, kge_engine) for obs in obs_pool]

    if calibrate and phi_vals:
        a, b = _calibrate_affine(
            callback_env,
            policy,
            index_manager,
            kge_engine,
            n_states=min(512, len(obs_pool)),
            rollouts_per_state=rollouts_per_state,
            horizon=horizon,
            gamma=gamma,
        )
    else:
        a = 1.0
        b = 0.0

    targets = [a * val + b for val in phi_vals]

    actor_params: List[torch.nn.Parameter] = []
    if hasattr(policy.mlp_extractor, "policy_network"):
        for param in policy.mlp_extractor.policy_network.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
                actor_params.append(param)

    policy.set_training_mode(True)
    policy.train()

    report_every = max(1, steps // 10)
    for step_idx in range(steps):
        batch_size = min(1024, len(obs_pool))
        if batch_size == 0:
            break
        indices = np.random.choice(len(obs_pool), size=batch_size, replace=len(obs_pool) < batch_size)
        batch_obs = [obs_pool[int(i)] for i in indices]
        batch_targets = torch.tensor([targets[int(i)] for i in indices], device=device, dtype=torch.float32)

        batch = _obs_collate(batch_obs, device)
        if not batch:
            continue

        values = policy.predict_values(batch).view(-1)
        loss = F.mse_loss(values, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
        optimizer.step()

        if verbose and (step_idx % report_every == 0 or step_idx == steps - 1):
            print(f"[Critic warmstart] step={step_idx + 1:5d}/{steps}  mse={float(loss):.5f}")

    for param in actor_params:
        param.requires_grad_(True)