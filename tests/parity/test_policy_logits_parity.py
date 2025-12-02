from pathlib import Path
import sys

import torch
from tensordict import TensorDict
import gymnasium as gym
import pytest

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))

from model import create_actor_critic

try:
    from sb3.sb3_model import CustomActorCriticPolicy, CustomCombinedExtractor
except Exception as exc:  # pragma: no cover
    CustomActorCriticPolicy = None
    CustomCombinedExtractor = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _DummyEmbedder(torch.nn.Module):
    """Minimal embedder that mimics the interface expected by the policies."""

    def __init__(self, vocab_size: int = 32, embed_dim: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.embedding_dim = embed_dim

    def get_embeddings_batch(self, idx: torch.Tensor) -> torch.Tensor:
        # idx (..., 3) but we only care about the predicate id in position 0
        pred_idx = idx[..., 0].clamp(min=0)
        emb = self.embedding(pred_idx)
        # Mimic real embedder: aggregate across atoms dimension when present
        if emb.dim() >= 3:
            emb = emb.mean(dim=-2)
        return emb

    def forward(self, idx):
        return self.get_embeddings_batch(idx)


def _build_obs(batch_size: int, padding_atoms: int, padding_states: int, max_arity: int, vocab_size: int, device):
    sub_index = torch.randint(0, vocab_size, (batch_size, 1, padding_atoms, max_arity + 1), device=device, dtype=torch.int32)
    derived_sub_indices = torch.randint(0, vocab_size, (batch_size, padding_states, padding_atoms, max_arity + 1), device=device, dtype=torch.int32)
    action_mask = torch.ones((batch_size, padding_states), device=device, dtype=torch.bool)
    td = TensorDict(
        {
            "sub_index": sub_index,
            "derived_sub_indices": derived_sub_indices,
            "action_mask": action_mask,
        },
        batch_size=[batch_size],
        device=device,
    )
    sb3_obs = {
        "sub_index": sub_index,
        "derived_sub_indices": derived_sub_indices,
        "action_mask": action_mask.to(torch.uint8),
    }
    return td, sb3_obs


def test_policy_logits_align_with_sb3_initialization():
    if CustomActorCriticPolicy is None:
        pytest.skip(f"SB3 dependencies unavailable: {_IMPORT_ERROR}")
    device = torch.device("cpu")
    padding_atoms = 3
    padding_states = 4
    max_arity = 2
    vocab_size = 16
    embed_dim = 8

    torch.manual_seed(123)
    embedder_new = _DummyEmbedder(vocab_size=vocab_size, embed_dim=embed_dim).to(device)

    # New policy should match SB3 initialization purely from seeded RNG (no explicit copy)
    policy_new = create_actor_critic(
        embedder=embedder_new,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=8,
        dropout_prob=0.2,
        device=device,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        max_arity=max_arity,
        total_vocab_size=vocab_size,
        match_sb3_init=False,
    )

    # Standalone sb3 policy with the same seed
    obs_space = gym.spaces.Dict(
        {
            "sub_index": gym.spaces.Box(low=0, high=vocab_size, shape=(1, padding_atoms, max_arity + 1), dtype=int),
            "derived_sub_indices": gym.spaces.Box(low=0, high=vocab_size, shape=(padding_states, padding_atoms, max_arity + 1), dtype=int),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(padding_states,), dtype=int),
        }
    )
    action_space = gym.spaces.Discrete(padding_states)

    def _lr_schedule(_):
        return 0.0

    torch.manual_seed(123)
    embedder_sb3 = _DummyEmbedder(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    sb3_policy = CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=_lr_schedule,
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": embedder_sb3, "features_dim": embed_dim},
        share_features_extractor=True,
    ).to(device)

    # Disable dropout to make the forward passes deterministic for comparison
    policy_new.eval()
    sb3_policy.eval()

    # State dicts should match exactly (ignore unused custom_network mirror)
    max_diff = 0.0
    max_key = None
    for k, v in policy_new.state_dict().items():
        if k.startswith("custom_network"):
            continue
        assert k in sb3_policy.state_dict()
        diff = (v - sb3_policy.state_dict()[k]).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            max_key = k
    assert max_diff == 0.0, f"max diff {max_diff} at {max_key}"

    # Allow overriding batch size via CLI: pytest ... --parity-batch=20
    # parity_batch = int(pytest.config.getoption("--parity-batch", default=2)) if hasattr(pytest, "config") else 2
    batch_size = 20 #parity_batch
    obs_td, sb3_obs = _build_obs(batch_size=batch_size, padding_atoms=padding_atoms, padding_states=padding_states, max_arity=max_arity, vocab_size=vocab_size, device=device)
    actions = torch.arange(batch_size, device=device) % padding_states

    with torch.no_grad():
        values_new, logp_new, _ = policy_new.evaluate_actions(obs_td, actions)
        values_sb3, logp_sb3, _ = sb3_policy.evaluate_actions(sb3_obs, actions)

    values_new = values_new.view(-1)
    values_sb3 = values_sb3.view(-1)
    logp_new = logp_new.view(-1)
    logp_sb3 = logp_sb3.view(-1)

    assert values_new.shape == values_sb3.shape
    assert logp_new.shape == logp_sb3.shape
    assert torch.isfinite(values_new).all() and torch.isfinite(values_sb3).all()
    assert torch.isfinite(logp_new).all() and torch.isfinite(logp_sb3).all()
    max_value_diff = (values_new - values_sb3).abs().max().item()
    max_logp_diff = (logp_new - logp_sb3).abs().max().item()
    assert max_value_diff < 1e-5, f"max value diff: {max_value_diff}"
    assert max_logp_diff < 1e-5, f"max logp diff: {max_logp_diff}"
    print(f"VALUES: \n{values_new} \nvs \n{values_sb3}")
    print(f"LOGP: \n{logp_new} \nvs \n{logp_sb3}")

if __name__ == "__main__":
    test_policy_logits_align_with_sb3_initialization()
