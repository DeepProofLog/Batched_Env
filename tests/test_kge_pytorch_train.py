import json
from pathlib import Path

import torch

from kge_train import TrainConfig, train_model


def _write_train_file(path: Path) -> None:
    examples = [
        "parent(alice,bob).",
        "parent(bob,charlie).",
        "sibling(charlie,diana).",
        "parent(alice,diana).",
    ]
    path.write_text("\n".join(examples), encoding="utf-8")


def test_train_rotate_saves_artifacts(tmp_path):
    data_root = tmp_path / "data"
    dataset_dir = data_root / "toy"
    dataset_dir.mkdir(parents=True)
    _write_train_file(dataset_dir / "train.txt")

    save_dir = tmp_path / "models"
    cfg = TrainConfig(
        save_dir=str(save_dir),
        dataset="toy",
        data_root=str(data_root),
        dim=4,
        gamma=6.0,
        p=1,
        lr=1e-2,
        batch_size=2,
        neg_ratio=1,
        epochs=1,
        num_workers=0,
        amp=False,
        compile=False,
        cpu=True,
        seed=7,
    )

    artifacts = train_model(cfg)

    weights_path = save_dir / "weights.pth"
    config_path = save_dir / "config.json"
    entity_path = save_dir / "entity2id.json"
    relation_path = save_dir / "relation2id.json"

    assert weights_path.exists()
    assert config_path.exists()
    assert entity_path.exists()
    assert relation_path.exists()

    state_dict = torch.load(weights_path, map_location="cpu")
    assert any(key.startswith("ent_re") for key in state_dict)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["num_entities"] == len(artifacts.entity2id)
    assert config["num_relations"] == len(artifacts.relation2id)

    assert len(artifacts.entity2id) == 4
    assert len(artifacts.relation2id) == 2
