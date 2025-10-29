import csv
from pathlib import Path
import os
import sys
# Add the parent directory to sys.path to allow imports from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kge_predict import PredictConfig, predict
from kge_train import TrainConfig, train_model


def _prepare_dataset(root: Path) -> None:
    dataset_dir = root / "toy"
    dataset_dir.mkdir(parents=True)
    train_examples = [
        "parent(alice,bob).",
        "parent(bob,charlie).",
        "sibling(charlie,diana).",
    ]
    valid_examples = [
        "parent(alice,bob).",
        "sibling(charlie,diana).",
        "parent(eve,frank).",  # unseen entity, should be skipped
    ]
    (dataset_dir / "train.txt").write_text("\n".join(train_examples), encoding="utf-8")
    (dataset_dir / "valid.txt").write_text("\n".join(valid_examples), encoding="utf-8")


def test_predict_rotate_scores_written(tmp_path):
    data_root = tmp_path / "data"
    _prepare_dataset(data_root)

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
        seed=11,
    )
    train_model(cfg)

    output_path = tmp_path / "scores.csv"
    pred_cfg = PredictConfig(
        model_dir=str(save_dir),
        output_path=str(output_path),
        dataset="toy",
        data_root=str(data_root),
        input_split="valid.txt",
        batch_size=2,
        amp=False,
        cpu=True,
    )
    artifacts = predict(pred_cfg)

    assert artifacts.num_scored == 2
    assert artifacts.skipped == 1
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    # Header plus two scored triples
    assert len(rows) == 1 + artifacts.num_scored
    scores = [float(row[-1]) for row in rows[1:]]
    assert all(isinstance(score, float) for score in scores)
