"""
Model Parity Tests.

Tests verifying that the tensor-based ActorCriticPolicy produces the same
results as the SB3 CustomActorCriticPolicy using real EmbedderLearnable.
"""
from pathlib import Path
import sys
from typing import Dict

import gymnasium as gym
import torch
import pytest
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))

# Import real embedders from both implementations
from embeddings import EmbedderLearnable
from model import ActorCriticPolicy

# Try to import SB3 components for parity testing
try:
    from sb3.sb3_model import CustomActorCriticPolicy as SB3CustomActorCriticPolicy
    from sb3.sb3_model import CustomCombinedExtractor as SB3CustomCombinedExtractor
    from sb3.sb3_embeddings import EmbedderLearnable as SB3EmbedderLearnable
    SB3_AVAILABLE = True
except ImportError as e:
    SB3_AVAILABLE = False
    _SB3_IMPORT_ERROR = e


# ============================================================================
# Default Test Parameters
# ============================================================================

DEFAULT_N_CONSTANTS = 100
DEFAULT_N_PREDICATES = 20
DEFAULT_N_VARS = 10
DEFAULT_EMBED_DIM = 64
DEFAULT_MAX_ARITY = 2
DEFAULT_PADDING_ATOMS = 3


# ============================================================================
# Helpers
# ============================================================================

def create_embedder(
    n_constants: int = DEFAULT_N_CONSTANTS,
    n_predicates: int = DEFAULT_N_PREDICATES,
    n_vars: int = DEFAULT_N_VARS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    seed: int = 42,
    device: str = "cpu"
) -> EmbedderLearnable:
    """Create a real EmbedderLearnable for the tensor-based implementation."""
    torch.manual_seed(seed)
    return EmbedderLearnable(
        n_constants=n_constants,
        n_predicates=n_predicates,
        n_vars=n_vars,
        max_arity=DEFAULT_MAX_ARITY,
        padding_atoms=DEFAULT_PADDING_ATOMS,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=embed_dim,
        predicate_embedding_size=embed_dim,
        atom_embedding_size=embed_dim,
        kge_regularization=0.0,
        kge_dropout_rate=0.0,
        device=device,
    )


def create_sb3_embedder(
    n_constants: int = DEFAULT_N_CONSTANTS,
    n_predicates: int = DEFAULT_N_PREDICATES,
    n_vars: int = DEFAULT_N_VARS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    seed: int = 42,
    device: str = "cpu"
):
    """Create a real EmbedderLearnable for the SB3 implementation."""
    if not SB3_AVAILABLE:
        pytest.skip(f"SB3 dependencies unavailable: {_SB3_IMPORT_ERROR}")
    torch.manual_seed(seed)
    return SB3EmbedderLearnable(
        n_constants=n_constants,
        n_predicates=n_predicates,
        n_vars=n_vars,
        max_arity=DEFAULT_MAX_ARITY,
        padding_atoms=DEFAULT_PADDING_ATOMS,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=embed_dim,
        predicate_embedding_size=embed_dim,
        atom_embedding_size=embed_dim,
        kge_regularization=0.0,
        kge_dropout_rate=0.0,
        device=device,
    )


def create_random_observation(
    batch_size: int = 4,
    n_actions: int = 16,
    n_atoms: int = DEFAULT_PADDING_ATOMS,
    n_predicates: int = DEFAULT_N_PREDICATES,
    n_constants: int = DEFAULT_N_CONSTANTS,
    n_vars: int = DEFAULT_N_VARS,
    max_arity: int = DEFAULT_MAX_ARITY,
    seed: int = 42,
    device: str = "cpu"
) -> TensorDict:
    """Create a random observation TensorDict with valid indices."""
    torch.manual_seed(seed)
    
    # Predicate indices in [1, n_predicates], constant indices in [1, n_constants + n_vars]
    pred_indices = torch.randint(1, n_predicates + 1, (batch_size, 1, n_atoms, 1), dtype=torch.int32, device=device)
    const_indices = torch.randint(1, n_constants + n_vars + 1, (batch_size, 1, n_atoms, max_arity), dtype=torch.int32, device=device)
    sub_index = torch.cat([pred_indices, const_indices], dim=-1)
    
    action_pred_indices = torch.randint(1, n_predicates + 1, (batch_size, n_actions, n_atoms, 1), dtype=torch.int32, device=device)
    action_const_indices = torch.randint(1, n_constants + n_vars + 1, (batch_size, n_actions, n_atoms, max_arity), dtype=torch.int32, device=device)
    derived_sub_indices = torch.cat([action_pred_indices, action_const_indices], dim=-1)
    
    # All actions valid for simplicity
    action_mask = torch.ones(batch_size, n_actions, dtype=torch.bool, device=device)
    
    return TensorDict({
        "sub_index": sub_index,
        "derived_sub_indices": derived_sub_indices,
        "action_mask": action_mask,
    }, batch_size=torch.Size([batch_size]))


def create_sb3_observation(obs_td: TensorDict) -> Dict[str, torch.Tensor]:
    """Convert TensorDict observation to SB3 dict format."""
    return {
        "sub_index": obs_td["sub_index"],
        "derived_sub_indices": obs_td["derived_sub_indices"],
        "action_mask": obs_td["action_mask"].to(torch.uint8),
    }


# ============================================================================
# Parity Tests
# ============================================================================

@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_embedder_parity():
    """Test that tensor and SB3 embedders produce identical outputs."""
    embed_dim = DEFAULT_EMBED_DIM
    batch_size = 4
    n_actions = 16
    seed = 42
    
    tensor_emb = create_embedder(embed_dim=embed_dim, seed=seed)
    sb3_emb = create_sb3_embedder(embed_dim=embed_dim, seed=seed)
    
    obs = create_random_observation(batch_size=batch_size, n_actions=n_actions, seed=123)
    
    with torch.no_grad():
        tensor_obs_emb = tensor_emb.get_embeddings_batch(obs["sub_index"])
        sb3_obs_emb = sb3_emb.get_embeddings_batch(obs["sub_index"])
        tensor_action_emb = tensor_emb.get_embeddings_batch(obs["derived_sub_indices"])
        sb3_action_emb = sb3_emb.get_embeddings_batch(obs["derived_sub_indices"])
    
    assert torch.allclose(tensor_obs_emb, sb3_obs_emb, atol=1e-5), \
        f"Obs embeddings differ: max diff = {(tensor_obs_emb - sb3_obs_emb).abs().max().item()}"
    assert torch.allclose(tensor_action_emb, sb3_action_emb, atol=1e-5), \
        f"Action embeddings differ: max diff = {(tensor_action_emb - sb3_action_emb).abs().max().item()}"


@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_forward_parity():
    """Test that tensor and SB3 policies produce identical forward outputs (logits & values)."""
    embed_dim = DEFAULT_EMBED_DIM
    batch_size = 4
    n_actions = 16
    n_atoms = DEFAULT_PADDING_ATOMS
    max_arity = DEFAULT_MAX_ARITY
    seed = 42
    device = "cpu"
    
    # Create embedders with same seed
    tensor_emb = create_embedder(embed_dim=embed_dim, seed=seed)
    sb3_emb = create_sb3_embedder(embed_dim=embed_dim, seed=seed)
    
    # Create tensor policy
    torch.manual_seed(seed)
    tensor_policy = ActorCriticPolicy(
        embedder=tensor_emb,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device(device),
        action_dim=n_actions,
    )
    tensor_policy.eval()
    
    # Create SB3 policy
    obs_space = gym.spaces.Dict({
        "sub_index": gym.spaces.Box(low=0, high=DEFAULT_N_CONSTANTS + DEFAULT_N_VARS, shape=(1, n_atoms, max_arity + 1), dtype=int),
        "derived_sub_indices": gym.spaces.Box(low=0, high=DEFAULT_N_CONSTANTS + DEFAULT_N_VARS, shape=(n_actions, n_atoms, max_arity + 1), dtype=int),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(n_actions,), dtype=int),
    })
    action_space = gym.spaces.Discrete(n_actions)
    
    torch.manual_seed(seed)
    sb3_policy = SB3CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        features_extractor_class=SB3CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": sb3_emb, "features_dim": embed_dim},
        share_features_extractor=True,
    ).to(device)
    sb3_policy.eval()
    
    # Create test observations
    obs_td = create_random_observation(batch_size=batch_size, n_actions=n_actions, seed=123)
    obs_sb3 = create_sb3_observation(obs_td)
    
    # Forward pass
    with torch.no_grad():
        tensor_actions, tensor_values, tensor_log_probs = tensor_policy(obs_td, deterministic=True)
        sb3_actions, sb3_values, sb3_log_probs = sb3_policy(obs_sb3, deterministic=True)
    
    sb3_values = sb3_values.flatten()
    
    assert torch.equal(tensor_actions, sb3_actions), \
        f"Actions differ: tensor={tensor_actions}, sb3={sb3_actions}"
    assert torch.allclose(tensor_values, sb3_values, atol=1e-5), \
        f"Values differ: max diff = {(tensor_values - sb3_values).abs().max().item()}"
    assert torch.allclose(tensor_log_probs, sb3_log_probs, atol=1e-5), \
        f"Log probs differ: max diff = {(tensor_log_probs - sb3_log_probs).abs().max().item()}"


@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_evaluate_actions_parity():
    """Test that tensor and SB3 policies produce identical evaluate_actions outputs."""
    embed_dim = DEFAULT_EMBED_DIM
    batch_size = 4
    n_actions = 16
    n_atoms = DEFAULT_PADDING_ATOMS
    max_arity = DEFAULT_MAX_ARITY
    seed = 42
    device = "cpu"
    
    # Create embedders with same seed
    tensor_emb = create_embedder(embed_dim=embed_dim, seed=seed)
    sb3_emb = create_sb3_embedder(embed_dim=embed_dim, seed=seed)
    
    # Create tensor policy
    torch.manual_seed(seed)
    tensor_policy = ActorCriticPolicy(
        embedder=tensor_emb,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device(device),
        action_dim=n_actions,
    )
    tensor_policy.eval()
    
    # Create SB3 policy
    obs_space = gym.spaces.Dict({
        "sub_index": gym.spaces.Box(low=0, high=DEFAULT_N_CONSTANTS + DEFAULT_N_VARS, shape=(1, n_atoms, max_arity + 1), dtype=int),
        "derived_sub_indices": gym.spaces.Box(low=0, high=DEFAULT_N_CONSTANTS + DEFAULT_N_VARS, shape=(n_actions, n_atoms, max_arity + 1), dtype=int),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(n_actions,), dtype=int),
    })
    action_space = gym.spaces.Discrete(n_actions)
    
    torch.manual_seed(seed)
    sb3_policy = SB3CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        features_extractor_class=SB3CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": sb3_emb, "features_dim": embed_dim},
        share_features_extractor=True,
    ).to(device)
    sb3_policy.eval()
    
    # Create test observations and actions
    obs_td = create_random_observation(batch_size=batch_size, n_actions=n_actions, seed=123)
    obs_sb3 = create_sb3_observation(obs_td)
    actions = torch.arange(batch_size, device=device) % n_actions
    
    # Evaluate actions
    with torch.no_grad():
        tensor_values, tensor_log_probs, tensor_entropy = tensor_policy.evaluate_actions(obs_td, actions)
        sb3_values, sb3_log_probs, sb3_entropy = sb3_policy.evaluate_actions(obs_sb3, actions)
    
    sb3_values = sb3_values.flatten()
    sb3_log_probs = sb3_log_probs.flatten()
    sb3_entropy = sb3_entropy.flatten() if sb3_entropy is not None else None
    
    assert torch.allclose(tensor_values, sb3_values, atol=1e-5), \
        f"Values differ: max diff = {(tensor_values - sb3_values).abs().max().item()}"
    assert torch.allclose(tensor_log_probs, sb3_log_probs, atol=1e-5), \
        f"Log probs differ: max diff = {(tensor_log_probs - sb3_log_probs).abs().max().item()}"
    if sb3_entropy is not None:
        assert torch.allclose(tensor_entropy, sb3_entropy, atol=1e-5), \
            f"Entropy differs: max diff = {(tensor_entropy - sb3_entropy).abs().max().item()}"


@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_predict_values_parity():
    """Test that tensor and SB3 policies produce identical predict_values outputs."""
    embed_dim = DEFAULT_EMBED_DIM
    batch_size = 4
    n_actions = 16
    n_atoms = DEFAULT_PADDING_ATOMS
    max_arity = DEFAULT_MAX_ARITY
    seed = 42
    device = "cpu"
    
    # Create embedders with same seed
    tensor_emb = create_embedder(embed_dim=embed_dim, seed=seed)
    sb3_emb = create_sb3_embedder(embed_dim=embed_dim, seed=seed)
    
    # Create tensor policy
    torch.manual_seed(seed)
    tensor_policy = ActorCriticPolicy(
        embedder=tensor_emb,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device(device),
        action_dim=n_actions,
    )
    tensor_policy.eval()
    
    # Create SB3 policy
    obs_space = gym.spaces.Dict({
        "sub_index": gym.spaces.Box(low=0, high=DEFAULT_N_CONSTANTS + DEFAULT_N_VARS, shape=(1, n_atoms, max_arity + 1), dtype=int),
        "derived_sub_indices": gym.spaces.Box(low=0, high=DEFAULT_N_CONSTANTS + DEFAULT_N_VARS, shape=(n_actions, n_atoms, max_arity + 1), dtype=int),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(n_actions,), dtype=int),
    })
    action_space = gym.spaces.Discrete(n_actions)
    
    torch.manual_seed(seed)
    sb3_policy = SB3CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        features_extractor_class=SB3CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": sb3_emb, "features_dim": embed_dim},
        share_features_extractor=True,
    ).to(device)
    sb3_policy.eval()
    
    # Create test observations
    obs_td = create_random_observation(batch_size=batch_size, n_actions=n_actions, seed=123)
    obs_sb3 = create_sb3_observation(obs_td)
    
    # Predict values
    with torch.no_grad():
        tensor_values = tensor_policy.predict_values(obs_td)
        sb3_values = sb3_policy.predict_values(obs_sb3)
    
    sb3_values = sb3_values.flatten()
    
    assert torch.allclose(tensor_values, sb3_values, atol=1e-5), \
        f"Predicted values differ: max diff = {(tensor_values - sb3_values).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
