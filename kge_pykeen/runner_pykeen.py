
#!/usr/bin/env python3
"""
pykeen_runner.py — Train & evaluate multiple KGE models with PyKEEN.

Features
- Works with either a named PyKEEN dataset (e.g., FB15k237, WN18RR) or user-provided triples files
- Trains a list of models sequentially (RotatE, ComplEx, TuckER, DistMult, TransE, PairRE, ...)
- Uses sLCWA training, filtered evaluation, and saves per-model artifacts + JSON/CSV metrics
- Supports CUDA via --device cuda and mixed precision via --use_amp (if available in your PyKEEN version)

Examples
---------
# Use a built-in dataset
python pykeen_runner.py --dataset FB15k237 --models RotatE,ComplEx,TuckER --epochs 50 --embedding_dim 1000 --device cuda --output_dir ./kge_pykeen/pykeen_runs

# Use your own triples
python pykeen_runner.py --train_path data/train.tsv --valid_path data/valid.tsv --test_path data/test.tsv \
    --delimiter tab --models RotatE,ComplEx --epochs 50 --embedding_dim 1000 --device cuda --output_dir ./kge_pykeen/pykeen_runs

Notes
-----
- Requires: `pip install pykeen`
- Model names should match PyKEEN classes, e.g., RotatE, ComplEx, TuckER, DistMult, TransE, PairRE.
- PairRE/HAKE/etc. may require recent PyKEEN versions; if a model isn't available, the script will skip it gracefully.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ------------- Safe imports (fail with a nice message) -------------
# Work around torchvision compatibility issue by preventing its import
import sys as _sys
_sys.modules['torchvision'] = None  # Prevent torchvision import which has compatibility issues

import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Suppress the harmless PyKEEN IterableDataset warning about batch counts
import warnings
warnings.filterwarnings('ignore', message='.*Length of IterableDataset.*', category=UserWarning)

try:
    import numpy as np
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.datasets import get_dataset
    from pykeen.triples import TriplesFactory
    from pykeen.evaluation import RankBasedEvaluator
    import pykeen.models as pk_models
    from pykeen.training.callbacks import TrainingCallback
except ImportError as e:
    print("This script requires PyKEEN. Please install it with `pip install pykeen` and retry.")
    raise
except RuntimeError as e:
    if "torchvision" in str(e):
        print("Warning: torchvision compatibility issue detected. Continuing without vision features...")
    else:
        raise

logging.getLogger("pykeen").setLevel(logging.WARNING)

# ------------- Utils -------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def select_best_gpu(min_free_gb: float = 1.0) -> Optional[int]:
    """
    Automatically select the GPU with the most free memory.
    
    Args:
        min_free_gb: Minimum free memory in GB required to consider a GPU
    
    Returns:
        GPU index with most free memory, or None if no suitable GPU found
    """
    if not torch.cuda.is_available():
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    
    # Get free memory for each GPU
    max_free_memory = 0
    best_gpu = None
    min_free_bytes = min_free_gb * 1e9
    
    for gpu_id in range(num_gpus):
        try:
            # Query memory without setting device (to avoid OOM on full GPUs)
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            used_memory = total_memory - free_memory
            
            print(f"GPU {gpu_id}: {free_memory / 1e9:.2f} GB free / {total_memory / 1e9:.2f} GB total "
                  f"({used_memory / 1e9:.2f} GB used, {100 * used_memory / total_memory:.1f}% utilized)")
            
            # Only consider GPUs with sufficient free memory
            if free_memory >= min_free_bytes and free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id
        except Exception as e:
            print(f"Warning: Could not query GPU {gpu_id}: {e}")
            continue
    
    if best_gpu is not None:
        print(f"Selected GPU {best_gpu} with {max_free_memory / 1e9:.2f} GB free memory")
    else:
        print(f"No GPU found with at least {min_free_gb:.1f} GB free memory")
    
    return best_gpu

def get_available_gpus(min_free_gb: float = 1.0) -> List[int]:
    """
    Get list of all GPUs with sufficient free memory.
    
    Args:
        min_free_gb: Minimum free memory in GB required
    
    Returns:
        List of GPU indices with sufficient free memory
    """
    if not torch.cuda.is_available():
        return []
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return []
    
    available_gpus = []
    min_free_bytes = min_free_gb * 1e9
    
    for gpu_id in range(num_gpus):
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            used_memory = total_memory - free_memory
            
            print(f"GPU {gpu_id}: {free_memory / 1e9:.2f} GB free / {total_memory / 1e9:.2f} GB total "
                  f"({used_memory / 1e9:.2f} GB used, {100 * used_memory / total_memory:.1f}% utilized)")
            
            if free_memory >= min_free_bytes:
                available_gpus.append(gpu_id)
        except Exception as e:
            print(f"Warning: Could not query GPU {gpu_id}: {e}")
            continue
    
    return available_gpus

def resolve_model_cls(name: str):
    # Try to resolve a model class by string name.
    # Works for RotatE, ComplEx, TuckER, DistMult, TransE, PairRE, etc.
    if hasattr(pk_models, name):
        return getattr(pk_models, name)
    # Fallback: try case-insensitive match
    for attr in dir(pk_models):
        if attr.lower() == name.lower():
            return getattr(pk_models, attr)
    raise AttributeError(f"PyKEEN model '{name}' not found in pykeen.models (check version or spelling).")

def parse_models(csv_list: str) -> List[str]:
    return [m.strip() for m in csv_list.split(",") if m.strip()]

def delim_from_arg(name: str) -> str:
    return {"tab":"\t", "comma":",", "space":" "}.get(name, "\t")

# ------------- Config -------------
@dataclass
class RunnerConfig:
    # Data
    dataset: Optional[str] = None
    train_path: Optional[str] = None
    valid_path: Optional[str] = None
    test_path: Optional[str] = None
    delimiter: str = "tab"  # tab|comma|space
    create_inverse_triples: bool = False

    # Models & training
    models: List[str] = None  # e.g., ["RotatE","ComplEx"]
    embedding_dim: int = 1000
    epochs: int = 50
    batch_size: int = 4096  # Increased default to match custom runner
    learning_rate: float = 1e-3
    negative_sample_rate: int = 1  # sLCWA default style
    training_loop: str = "sLCWA"  # or "LCWA"
    evaluator: str = "RankBasedEvaluator"
    use_amp: bool = True  # Enable mixed precision by default
    use_early_stopping: bool = False  # Disable early stopping for speed
    eval_frequency: int = 5  # Evaluate validation set every N epochs (<=0 disables)
    eval_batch_size: Optional[int] = None  # Separate batch size for evaluation (default: 4x training)

    # Runtime
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4

    # Output
    output_dir: str = "./pykeen_runs"


class MRREvaluationCallback(TrainingCallback):
    """Periodically evaluate on validation triples and print only MRR."""

    def __init__(
        self,
        *,
        evaluation_triples: torch.Tensor,
        frequency: int,
        evaluator_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        prefix: str = "validation",
        additional_filter_triples: Optional[List[torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.evaluation_triples = evaluation_triples
        self.prefix = prefix
        self.batch_size = batch_size
        self.additional_filter_triples = additional_filter_triples or []
        evaluator_kwargs = evaluator_kwargs or {}
        if batch_size is not None:
            evaluator_kwargs = dict(evaluator_kwargs)
            evaluator_kwargs.setdefault("batch_size", batch_size)
        self.evaluator = RankBasedEvaluator(**evaluator_kwargs)

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D401
        if epoch % self.frequency:
            return

        result = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.evaluation_triples,
            device=self.training_loop.device,
            batch_size=self.evaluator.batch_size or self.batch_size,
            additional_filter_triples=self.additional_filter_triples,
        )

        metrics = result.to_flat_dict()
        possible_keys = [
            "both.realistic.mean_reciprocal_rank",
            "both.realistic.inverse_harmonic_mean_rank",
        ]
        mrr = None
        for key in possible_keys:
            if key in metrics:
                mrr = metrics[key]
                break
        if mrr is None:
            for key, value in metrics.items():
                if key.endswith("mean_reciprocal_rank") or key.endswith("inverse_harmonic_mean_rank"):
                    mrr = value
                    break

        if mrr is not None:
            print(f"[{self.prefix}] epoch={epoch} MRR={mrr:.4f}")

# ------------- Data loading -------------
def load_triples_factory_from_files(cfg: RunnerConfig) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    delim = delim_from_arg(cfg.delimiter)
    
    def read_rows(path: str):
        """Read triples from file. Supports both TSV format and Prolog-style facts."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if it's Prolog-style format: relation(entity1,entity2).
                if '(' in line and ')' in line and line.endswith('.'):
                    # Parse Prolog format: relation(entity1,entity2).
                    try:
                        rel_start = line.index('(')
                        rel_end = line.rindex(')')
                        relation = line[:rel_start].strip()
                        args = line[rel_start+1:rel_end].strip()
                        entities = [e.strip() for e in args.split(',')]
                        if len(entities) == 2:
                            head, tail = entities
                            yield head, relation, tail
                        else:
                            print(f"Warning: Skipping malformed line: {line}")
                    except Exception as e:
                        print(f"Warning: Could not parse line '{line}': {e}")
                else:
                    # Try TSV/CSV format
                    parts = line.split(delim)
                    if len(parts) >= 3:
                        yield parts[0].strip(), parts[1].strip(), parts[2].strip()
                    else:
                        print(f"Warning: Skipping line with insufficient columns: {line}")

    assert cfg.train_path and os.path.isfile(cfg.train_path), f"train_path not found: {cfg.train_path}"

    # Build training factory with its own mappings
    train_triples = list(read_rows(cfg.train_path))
    print(f"Loaded {len(train_triples)} training triples from {cfg.train_path}")
    
    # Convert to numpy array as required by PyKEEN
    train_triples_array = np.array(train_triples, dtype=str)
    
    training = TriplesFactory.from_labeled_triples(
        triples=train_triples_array,
        create_inverse_triples=cfg.create_inverse_triples,
    )

    # Validation/test should be mapped to training IDs; drop OOV
    def map_split(path: Optional[str]) -> Optional[TriplesFactory]:
        if not path or not os.path.isfile(path): 
            return None
        raw_triples = list(read_rows(path))
        # Filter to only include entities and relations seen in training
        triples = [
            (h, r, t) for h, r, t in raw_triples 
            if h in training.entity_to_id and r in training.relation_to_id and t in training.entity_to_id
        ]
        print(f"Loaded {len(triples)}/{len(raw_triples)} valid triples from {path}")
        if not triples:
            return None
        # Convert to numpy array
        triples_array = np.array(triples, dtype=str)
        return TriplesFactory.from_labeled_triples(
            triples=triples_array, 
            entity_to_id=training.entity_to_id, 
            relation_to_id=training.relation_to_id
        )

    validation = map_split(cfg.valid_path)
    testing = map_split(cfg.test_path)

    # If no val/test provided, let PyKEEN create them later
    return training, validation, testing

def load_dataset_or_files(cfg: RunnerConfig):
    """Load data from custom files or PyKEEN built-in dataset."""
    # If dataset name provided but no paths, try to load from files first
    if cfg.dataset and not cfg.train_path:
        base_path = os.path.join("./data", cfg.dataset.lower())
        if os.path.isdir(base_path):
            cfg.train_path = os.path.join(base_path, "train.txt")
            cfg.valid_path = os.path.join(base_path, "valid.txt")
            cfg.test_path = os.path.join(base_path, "test.txt")
            print(f"Loading dataset from: {base_path}")
            return load_triples_factory_from_files(cfg)
        else:
            # If local files don't exist, try PyKEEN built-in dataset
            try:
                print(f"Local path '{base_path}' not found, trying PyKEEN built-in dataset '{cfg.dataset}'...")
                ds = get_dataset(dataset=cfg.dataset)
                return ds.training, ds.validation, ds.testing
            except Exception as e:
                raise ValueError(f"Dataset '{cfg.dataset}' not found locally in '{base_path}' and not available as PyKEEN built-in dataset. Error: {e}")
    
    # If explicit paths provided, use them
    return load_triples_factory_from_files(cfg)

# ------------- Run one model -------------
def run_one_model(cfg: RunnerConfig, model_name: str, train_tf: TriplesFactory, valid_tf: TriplesFactory, test_tf: TriplesFactory) -> Dict:
    run_dir = os.path.join(cfg.output_dir, f"{model_name}_{now_ts()}")
    makedirs(run_dir)
    print(f"\n=== Training {model_name} ===\nArtifacts: {run_dir}")

    # Resolve model class
    try:
        model_cls = resolve_model_cls(model_name)
    except Exception as e:
        print(f"[skip] Could not resolve model '{model_name}': {e}")
        return {"model": model_name, "status": "skipped", "reason": str(e)}

    # Model kwargs: align names for common models
    model_kwargs = {}
    
    # CRITICAL: ComplEx works better with BCEAfterSigmoid loss instead of default SoftplusLoss
    # PyKeen's default SoftplusLoss doesn't work well for ComplEx on many datasets
    # The PyTorch custom implementation uses logsigmoid loss which is equivalent to BCEAfterSigmoid
    if model_name.lower() in {"complex"}:
        model_kwargs["loss"] = "bceaftersigmoid"  # Use BCE loss instead of Softplus for ComplEx
        print(f"Note: Using BCEAfterSigmoid loss for ComplEx (better than default SoftplusLoss)")
    
    if model_name.lower() in {"rotate", "complex", "distmult", "transe", "pairre"}:
        model_kwargs["embedding_dim"] = cfg.embedding_dim
    elif model_name.lower() in {"tucker"}:
        model_kwargs["embedding_dim"] = cfg.embedding_dim  # entity dim
        model_kwargs["relation_dim"] = cfg.embedding_dim // 2
        # Tucker uses three separate dropout parameters for different components
        model_kwargs["dropout_0"] = 0.3  # dropout after first core tensor multiplication
        model_kwargs["dropout_1"] = 0.4  # dropout after batch normalization
        model_kwargs["dropout_2"] = 0.5  # dropout after second core tensor multiplication
        # Disable batch normalization for small datasets to avoid batch size issues
        # Batch norm requires drop_last=True but that can fail with small datasets
        if train_tf.num_triples < cfg.batch_size:
            model_kwargs["apply_batch_normalization"] = False
            print(f"Note: Disabled batch normalization for Tucker (dataset size {train_tf.num_triples} < batch size {cfg.batch_size})")
    elif model_name.lower() in {"conve"}:
        model_kwargs["embedding_dim"] = cfg.embedding_dim
        # ConvE also uses batch normalization by default, which causes issues with small datasets
        # Disable it for small datasets to avoid batch size issues
        if train_tf.num_triples < cfg.batch_size:
            model_kwargs["apply_batch_normalization"] = False
            print(f"Note: Disabled batch normalization for ConvE (dataset size {train_tf.num_triples} < batch size {cfg.batch_size})")
    else:
        # Generic fallback
        model_kwargs["embedding_dim"] = cfg.embedding_dim

    # Pipeline kwargs - build conditionally to support different PyKEEN versions
    # Use larger batch size for evaluation (can be much larger since no backward pass)
    eval_batch = cfg.eval_batch_size if cfg.eval_batch_size is not None else (cfg.batch_size * 4)
    
    # Configure evaluator with batch processing for speed
    # Note: The initial evaluation delay (~50s) is from building filter masks for filtered evaluation.
    # This is unavoidable with PyKEEN's filtered protocol but only happens once.
    # The actual evaluation is much faster (~20-30s) once filters are built.
    auto_mem_supported = True
    try:
        # Try with automatic_memory_optimization if available (newer PyKEEN versions)
        evaluator = RankBasedEvaluator(
            filtered=True,  # Keep filtered for accurate metrics
            batch_size=eval_batch,
            automatic_memory_optimization=True
        )
    except TypeError:
        auto_mem_supported = False
        # Fall back to just batch_size for older versions
        evaluator = RankBasedEvaluator(
            filtered=True,
            batch_size=eval_batch
        )
    
    print(f"Note: Evaluation will build filter masks (one-time ~50s delay), then evaluate quickly.")
    
    # Handle multi-GPU setup
    actual_device = cfg.device
    if cfg.device == "cuda:all":
        # Use all available GPUs via DataParallel
        actual_device = "cuda:0"  # Primary device
        print(f"Using DataParallel across all {torch.cuda.device_count()} GPUs")
    
    # Tucker uses batch normalization, which requires drop_last=False for small datasets
    training_kwargs = dict(
        num_epochs=cfg.epochs,
        use_tqdm_batch=True,
        batch_size=cfg.batch_size,
        label_smoothing=0.0,
        num_workers=cfg.num_workers,  # Add parallel data loading
    )

    callback_instances: List[TrainingCallback] = []

    if valid_tf is not None and cfg.eval_frequency and cfg.eval_frequency > 0:
        print(f"Scheduling validation evaluation every {cfg.eval_frequency} epochs on {valid_tf.num_triples} triples")
        validation_evaluator_kwargs: Dict[str, Any] = dict(
            filtered=True,
        )
        if auto_mem_supported:
            validation_evaluator_kwargs["automatic_memory_optimization"] = True
        callback_instances.append(
            MRREvaluationCallback(
                evaluation_triples=valid_tf.mapped_triples,
                frequency=cfg.eval_frequency,
                evaluator_kwargs=validation_evaluator_kwargs,
                batch_size=eval_batch,
                prefix="validation",
                additional_filter_triples=[train_tf.mapped_triples],
            )
        )

    if callback_instances:
        if len(callback_instances) == 1:
            training_kwargs["callbacks"] = callback_instances[0]
        else:
            training_kwargs["callbacks"] = callback_instances
    
    pipeline_kwargs = dict(
        model=model_cls,
        model_kwargs=model_kwargs,
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        training_kwargs=training_kwargs,
        optimizer="adam",
        optimizer_kwargs=dict(lr=cfg.learning_rate),
        training_loop=cfg.training_loop,
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=cfg.negative_sample_rate),
        evaluator=evaluator,
        evaluation_kwargs=dict(
            batch_size=eval_batch,  # Use larger batch size for evaluation
            slice_size=None,  # Let PyKEEN auto-determine to avoid OOM
            use_tqdm=True,
            additional_filter_triples=[train_tf.mapped_triples],  # Critical: pass training triples for filtered evaluation
        ),
        random_seed=cfg.seed,
        device=actual_device,
    )
    # Add early stopping only if requested (disabled by default for speed)
    if cfg.use_early_stopping:
        pipeline_kwargs["stopper"] = "early"
        pipeline_kwargs["stopper_kwargs"] = dict(
            frequency=cfg.eval_frequency,
            patience=10,
            relative_delta=0.0,
            metric="both.realistic.mean_reciprocal_rank"
        )
    
    # Add use_amp only if supported (PyKEEN >= 1.10)
    if cfg.use_amp:
        try:
            # Test if use_amp is supported
            import inspect
            sig = inspect.signature(pipeline)
            if 'use_amp' in sig.parameters:
                pipeline_kwargs['use_amp'] = True
            else:
                print("[warn] use_amp not supported in this PyKEEN version, ignoring")
        except Exception:
            pass
    
    results = pipeline(**pipeline_kwargs)
    
    # Wrap model with DataParallel if using multiple GPUs
    if cfg.device == "cuda:all" and torch.cuda.device_count() > 1:
        print(f"Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs")
        results.model = torch.nn.DataParallel(results.model)

    # Save artifacts
    try:
        results.save_to_directory(run_dir)
    except Exception as e:
        print(f"[warn] Could not save full directory: {e}")

    # Extract key metrics if available
    out = {
        "model": model_name,
        "status": "ok",
        "run_dir": run_dir,
        "config": asdict(cfg),
    }
    
    try:
        m = results.metric_results.to_dict()
        # Get metrics from realistic (filtered) evaluation
        metrics_dict = m.get("both", {}).get("realistic", {})
        
        # MRR is inverse_harmonic_mean_rank in PyKEEN
        mrr = metrics_dict.get("inverse_harmonic_mean_rank", metrics_dict.get("mean_reciprocal_rank"))
        
        out["metrics"] = {
            "mrr": mrr,
            "mr": metrics_dict.get("arithmetic_mean_rank"),
            "hits@1": metrics_dict.get("hits_at_1"),
            "hits@3": metrics_dict.get("hits_at_3"),
            "hits@10": metrics_dict.get("hits_at_10"),
        }

        if mrr is not None:
            print(f"Test MRR ({model_name}): {mrr:.4f}")
        else:
            print(f"Test MRR ({model_name}): N/A")
    except Exception as e:
        out["metrics_error"] = str(e)
        print(f"\n[warn] Could not extract metrics: {e}\n")

    # Write summary JSON
    try:
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        print(f"[warn] Could not write summary.json: {e}")

    return out

# ------------- Main -------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", default='family', help="Name of dataset (e.g., family, countries_s3, FB15k237, WN18RR). Will auto-detect paths in ./data/<dataset>/")
    p.add_argument("--train_path", default=None, help="Override: path to train.txt")
    p.add_argument("--valid_path", default=None, help="Override: path to valid.txt")
    p.add_argument("--test_path", default=None, help="Override: path to test.txt")
    p.add_argument("--delimiter", choices=["tab","comma","space"], default="tab", help="Delimiter for TSV/CSV files (ignored for Prolog format)")
    p.add_argument("--create_inverse_triples", action="store_true", help="Create inverse triples for relations")
    # Models:  RotatE,ComplEx,TuckER,DistMult,TransE,PairRE,ConvE,QuatE,AutoSF
    p.add_argument("--models", default="complex", help="Comma-separated list of model names (e.g., RotatE,ComplEx)")
    p.add_argument("--embedding_dim", type=int, default=500, help="Embedding dimension")
    p.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--negative_sample_rate", type=int, default=1, help="Number of negative samples per positive")
    p.add_argument("--training_loop", choices=["sLCWA","LCWA"], default="sLCWA", help="Training loop type")
    p.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision if available")
    p.add_argument("--use_early_stopping", action="store_true", help="Enable early stopping (slows down training)")
    p.add_argument("--eval_frequency", type=int, default=5, help="Evaluate validation set every N epochs (<=0 disables)")
    p.add_argument("--eval_batch_size", type=int, default=512, help="Batch size for evaluation (default: 4x training batch size)")
    # Runtime
    p.add_argument("--device", default="cuda:all", choices=["cpu", "cuda:1", "cuda:all"], 
                   help="Device: 'cpu' (use CPU), 'cuda:1' (auto-select best GPU), 'cuda:all' (use all available GPUs)")
    p.add_argument("--min_gpu_memory_gb", type=float, default=2.0, help="Minimum free GPU memory in GB to consider a GPU available")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    # Output
    p.add_argument("--output_dir", default="./kge_pykeen/pykeen_runs")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    
    # Handle device selection based on user choice
    device = args.device
    min_memory_gb = args.min_gpu_memory_gb
    
    if device == "cpu":
        print("\n=== Using CPU ===")
        print("Training will run on CPU (slower but always available)\n")
    
    elif device == "cuda:1":
        print("\n=== Auto-selecting best GPU ===")
        best_gpu = select_best_gpu(min_free_gb=min_memory_gb)
        if best_gpu is not None:
            device = f"cuda:{best_gpu}"
            # Set CUDA_VISIBLE_DEVICES to use only the selected GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"Using device: cuda:{best_gpu}\n")
        else:
            device = "cpu"
            print(f"No GPU with at least {min_memory_gb} GB free memory found.")
            print("Falling back to CPU\n")
    
    elif device == "cuda:all":
        print("\n=== Using all available GPUs ===")
        available_gpus = get_available_gpus(min_free_gb=min_memory_gb)
        
        if len(available_gpus) == 0:
            device = "cpu"
            print(f"No GPUs with at least {min_memory_gb} GB free memory found.")
            print("Falling back to CPU\n")
        elif len(available_gpus) == 1:
            device = f"cuda:{available_gpus[0]}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print(f"Only 1 GPU available with sufficient memory: GPU {available_gpus[0]}")
            print(f"Using device: cuda:{available_gpus[0]}\n")
        else:
            # Multiple GPUs available - use DataParallel
            print(f"Found {len(available_gpus)} GPUs with sufficient memory: {available_gpus}")
            print("Will use DataParallel for multi-GPU training")
            # Set CUDA_VISIBLE_DEVICES to only the available GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
            # After setting CUDA_VISIBLE_DEVICES, the first visible GPU becomes cuda:0
            device = "cuda:all"
            print(f"GPUs will be mapped as: {', '.join([f'GPU {i}->cuda:{idx}' for idx, i in enumerate(available_gpus)])}\n")
    
    cfg = RunnerConfig(
        dataset=args.dataset,
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        delimiter=args.delimiter,
        create_inverse_triples=args.create_inverse_triples,
        models=parse_models(args.models),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        negative_sample_rate=args.negative_sample_rate,
        training_loop=args.training_loop,
        use_amp=args.use_amp,
        use_early_stopping=args.use_early_stopping,
        eval_frequency=args.eval_frequency,
        eval_batch_size=args.eval_batch_size,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    # Prepare data
    train_tf, valid_tf, test_tf = load_dataset_or_files(cfg)

    # Results table
    all_results: List[Dict] = []
    for model_name in cfg.models:
        try:
            out = run_one_model(cfg, model_name, train_tf, valid_tf, test_tf)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Training {model_name} failed with error: {e}")
            traceback.print_exc()
            out = {"model": model_name, "status": "failed", "error": str(e)}
        all_results.append(out)

    # Write CSV scoreboard
    scoreboard_csv = os.path.join(cfg.output_dir, f"scoreboard_{now_ts()}.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(scoreboard_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model","status","mrr","mr","hits@1","hits@3","hits@10","run_dir"])
        for r in all_results:
            metrics = r.get("metrics") or {}
            writer.writerow([
                r.get("model"), r.get("status"),
                metrics.get("mrr"), metrics.get("mr"),
                metrics.get("hits@1"), metrics.get("hits@3"), metrics.get("hits@10"),
                r.get("run_dir"),
            ])
    print(f"\nSaved scoreboard: {scoreboard_csv}")
    print("Done.")

if __name__ == "__main__":
    main()
