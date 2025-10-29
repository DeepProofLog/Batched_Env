#!/usr/bin/env python3
"""
Test script to compare inference speed across TensorFlow, PyTorch, and PyKEEN KGE models.

This script:
1. Samples atoms from the family dataset
2. Runs inference using each of the three implementations
3. Compares inference time and throughput
4. Ensures all implementations use the same batch size for fair comparison

IMPORTANT: TensorFlow and PyTorch can conflict when both use CUDA in the same process.
If you experience segfaults, run the tests separately:
  
  # Test TensorFlow only:
  python tests/test_inference_speed.py --skip_pytorch --skip_pykeen
  
  # Test PyTorch only:
  python tests/test_inference_speed.py --skip_tf --skip_pykeen
  
  # Test PyKEEN only:
  python tests/test_inference_speed.py --skip_tf --skip_pytorch --pykeen_model_dir <path>
"""

import os
import sys

# Set environment variables to help prevent TF/PyTorch conflicts
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import time
import random
import argparse
import tempfile
from typing import List, Tuple
from dataclasses import dataclass

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


@dataclass
class InferenceResult:
    """Results from an inference run."""
    framework: str
    model_path: str
    num_atoms: int
    batch_size: int
    total_time: float
    atoms_per_second: float
    scores_sample: List[float]  # Sample of scores for validation


def load_sample_atoms(dataset_path: str, num_atoms: int = 1000, seed: int = 42) -> List[Tuple[str, str, str]]:
    """
    Load a random sample of atoms from the dataset.
    
    Args:
        dataset_path: Path to the dataset file (e.g., train.txt or test.txt)
        num_atoms: Number of atoms to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of tuples (head, relation, tail)
    """
    random.seed(seed)
    
    atoms = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse Prolog-style format: relation(entity1,entity2).
            if '(' in line and ')' in line and line.endswith('.'):
                try:
                    rel_start = line.index('(')
                    rel_end = line.rindex(')')
                    relation = line[:rel_start].strip()
                    args = line[rel_start+1:rel_end].strip()
                    entities = [e.strip() for e in args.split(',')]
                    if len(entities) == 2:
                        head, tail = entities
                        atoms.append((head, relation, tail))
                except Exception as e:
                    print(f"Warning: Could not parse line '{line}': {e}")
    
    # Sample random atoms
    if len(atoms) > num_atoms:
        atoms = random.sample(atoms, num_atoms)
    
    print(f"Loaded {len(atoms)} atoms from {dataset_path}")
    return atoms


def write_atoms_to_file(atoms: List[Tuple[str, str, str]], output_path: str):
    """Write atoms to file in Prolog format."""
    with open(output_path, 'w') as f:
        for head, relation, tail in atoms:
            f.write(f"{relation}({head},{tail}).\n")


def test_tensorflow_inference(atoms: List[Tuple[str, str, str]], batch_size: int, 
                               checkpoint_dir: str, run_signature: str) -> InferenceResult:
    """Test inference speed using TensorFlow implementation."""
    print("\n" + "="*80)
    print("Testing TensorFlow KGE Inference")
    print("="*80)
    
    from kge_tf.kge_inference_tf import KGEInference
    
    # Initialize inference engine
    inference_engine = KGEInference(
        dataset_name='family',
        base_path='./data',
        checkpoint_dir=checkpoint_dir,
        run_signature=run_signature,
        seed=0,
    )
    
    # Build model (lazy initialization)
    if inference_engine.model is None:
        inference_engine._build_and_load_model()
    
    # Prepare atoms as strings
    atom_strings = [f"{r}({h},{t})" for h, r, t in atoms]
    
    # Warm-up run
    print("Warming up...")
    _ = inference_engine.predict(atom_strings[0])
    
    # Timed inference
    print(f"Running inference on {len(atoms)} atoms with batch_size={batch_size}...")
    start_time = time.time()
    
    scores = []
    for i in range(0, len(atom_strings), batch_size):
        batch_atoms = atom_strings[i:i+batch_size]
        batch_tuples = [inference_engine._atom_str_to_tuple(atom) for atom in batch_atoms]
        
        # Batch prediction
        (model_inputs, _) = inference_engine._prepare_batch(batch_tuples)
        kge_inputs = (model_inputs[0], model_inputs[1])
        atom_outputs, _ = inference_engine.model.kge_model.call(kge_inputs)
        
        batch_scores = [float(score[0].numpy()) for score in atom_outputs]
        scores.extend(batch_scores)
    
    end_time = time.time()
    total_time = end_time - start_time
    atoms_per_second = len(atoms) / total_time
    
    print(f"✓ Completed in {total_time:.2f}s ({atoms_per_second:.2f} atoms/sec)")
    
    return InferenceResult(
        framework="TensorFlow",
        model_path=checkpoint_dir,
        num_atoms=len(atoms),
        batch_size=batch_size,
        total_time=total_time,
        atoms_per_second=atoms_per_second,
        scores_sample=scores[:10]
    )


def test_pytorch_inference(atoms: List[Tuple[str, str, str]], batch_size: int,
                           model_dir: str, temp_dir: str) -> InferenceResult:
    """Test inference speed using PyTorch implementation."""
    print("\n" + "="*80)
    print("Testing PyTorch KGE Inference")
    print("="*80)
    
    from kge_pytorch.kge_inference_torch import PredictConfig, predict
    
    # Write atoms to temporary file
    input_file = os.path.join(temp_dir, "pytorch_input.txt")
    write_atoms_to_file(atoms, input_file)
    
    output_file = os.path.join(temp_dir, "pytorch_output.txt")
    
    # Create config
    cfg = PredictConfig(
        model_dir=model_dir,
        output_path=output_file,
        input_path=input_file,
        batch_size=batch_size,
        amp=False,
        cpu=False
    )
    
    # Warm-up run (just load the model)
    print("Warming up (loading model)...")
    import torch
    
    # Force CUDA context reset if TensorFlow was used
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.synchronize()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    from kge_pytorch.kge_inference_torch import load_model
    model, entity2id, relation2id, _ = load_model(cfg.model_dir, device)
    
    # Timed inference
    print(f"Running inference on {len(atoms)} atoms with batch_size={batch_size}...")
    start_time = time.time()
    
    result = predict(cfg)
    
    end_time = time.time()
    total_time = end_time - start_time
    atoms_per_second = len(atoms) / total_time
    
    # Read scores
    scores = []
    with open(output_file, 'r') as f:
        for line in f:
            if '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        scores.append(float(parts[1]))
                    except ValueError:
                        pass
    
    print(f"✓ Completed in {total_time:.2f}s ({atoms_per_second:.2f} atoms/sec)")
    
    return InferenceResult(
        framework="PyTorch",
        model_path=model_dir,
        num_atoms=result.num_scored,
        batch_size=batch_size,
        total_time=total_time,
        atoms_per_second=atoms_per_second,
        scores_sample=scores[:10]
    )


def test_pykeen_inference(atoms: List[Tuple[str, str, str]], batch_size: int,
                         model_dir: str, temp_dir: str) -> InferenceResult:
    """Test inference speed using PyKEEN implementation."""
    print("\n" + "="*80)
    print("Testing PyKEEN KGE Inference")
    print("="*80)
    
    from kge_pykeen.kge_inference_pykeen import PredictConfig, predict
    
    # Write atoms to temporary file
    input_file = os.path.join(temp_dir, "pykeen_input.txt")
    write_atoms_to_file(atoms, input_file)
    
    output_file = os.path.join(temp_dir, "pykeen_output.txt")
    
    # Create config
    cfg = PredictConfig(
        model_dir=model_dir,
        output_path=output_file,
        input_path=input_file,
        batch_size=batch_size,
        cpu=False
    )
    
    # Warm-up (just load the model)
    print("Warming up (loading model)...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    from kge_pykeen.kge_inference_pykeen import load_model
    model, entity2id, relation2id = load_model(cfg.model_dir, device)
    
    # Timed inference
    print(f"Running inference on {len(atoms)} atoms with batch_size={batch_size}...")
    start_time = time.time()
    
    result = predict(cfg)
    
    end_time = time.time()
    total_time = end_time - start_time
    atoms_per_second = len(atoms) / total_time
    
    # Read scores
    scores = []
    with open(output_file, 'r') as f:
        for line in f:
            if '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        scores.append(float(parts[1]))
                    except ValueError:
                        pass
    
    print(f"✓ Completed in {total_time:.2f}s ({atoms_per_second:.2f} atoms/sec)")
    
    return InferenceResult(
        framework="PyKEEN",
        model_path=model_dir,
        num_atoms=result.num_scored,
        batch_size=batch_size,
        total_time=total_time,
        atoms_per_second=atoms_per_second,
        scores_sample=scores[:10]
    )


def print_comparison_results(results: List[InferenceResult]):
    """Print a comparison table of results."""
    print("\n" + "="*80)
    print("INFERENCE SPEED COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Framework':<15} {'Atoms':<10} {'Batch Size':<12} {'Time (s)':<12} {'Atoms/sec':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result.framework:<15} {result.num_atoms:<10} {result.batch_size:<12} "
              f"{result.total_time:<12.2f} {result.atoms_per_second:<15.2f}")
    
    print("\n" + "="*80)
    
    # Find fastest
    fastest = min(results, key=lambda x: x.total_time)
    print(f"🏆 Fastest: {fastest.framework} ({fastest.atoms_per_second:.2f} atoms/sec)")
    
    # Compute speedup ratios
    print("\nSpeedup ratios (relative to slowest):")
    slowest_time = max(r.total_time for r in results)
    for result in results:
        speedup = slowest_time / result.total_time
        print(f"  {result.framework}: {speedup:.2f}x")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare KGE inference speed across frameworks")
    parser.add_argument("--dataset", default="family", help="Dataset name")
    parser.add_argument("--data_path", default="./data/family/train.txt", 
                       help="Path to dataset file")
    parser.add_argument("--num_atoms", type=int, default=1000, 
                       help="Number of atoms to sample for testing")
    parser.add_argument("--batch_size", type=int, default=2048,
                       help="Batch size for inference (same for all frameworks)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model paths
    parser.add_argument("--tf_checkpoint_dir", 
                       default="/home/castellanoontiv/checkpoints/",
                       help="TensorFlow checkpoint directory")
    parser.add_argument("--tf_run_signature",
                       default="kinship_family-backward_0_1-no_reasoner-complex-True-256-256-4-rules.txt",
                       help="TensorFlow run signature")
    parser.add_argument("--pytorch_model_dir",
                       default="./kge_pytorch/models",
                       help="PyTorch model directory")
    parser.add_argument("--pykeen_model_dir",
                       help="PyKEEN model directory (if not provided, will skip PyKEEN test)")
    
    # Options to skip specific tests
    parser.add_argument("--skip_tf", action="store_true", help="Skip TensorFlow test")
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch test")
    parser.add_argument("--skip_pykeen", action="store_true", help="Skip PyKEEN test")
    
    args = parser.parse_args()
    
    # Load sample atoms
    print("Loading sample atoms from dataset...")
    atoms = load_sample_atoms(args.data_path, args.num_atoms, args.seed)
    
    if not atoms:
        print("Error: No atoms loaded from dataset")
        return 1
    
    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="kge_inference_test_")
    print(f"Using temporary directory: {temp_dir}")
    
    results = []
    
    try:
        # Test TensorFlow
        if not args.skip_tf:
            try:
                tf_result = test_tensorflow_inference(
                    atoms, args.batch_size, 
                    args.tf_checkpoint_dir, args.tf_run_signature
                )
                results.append(tf_result)
                
                # Clean up TensorFlow GPU memory before PyTorch test
                import tensorflow as tf
                import gc
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                print(f"\n❌ TensorFlow test failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test PyTorch
        if not args.skip_pytorch:
            try:
                # Check if PyTorch model exists
                if os.path.exists(args.pytorch_model_dir):
                    pytorch_result = test_pytorch_inference(
                        atoms, args.batch_size, args.pytorch_model_dir, temp_dir
                    )
                    results.append(pytorch_result)
                    
                    # Clean up PyTorch GPU memory
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                else:
                    print(f"\n⚠ PyTorch model directory not found: {args.pytorch_model_dir}")
            except Exception as e:
                print(f"\n❌ PyTorch test failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test PyKEEN
        if not args.skip_pykeen and args.pykeen_model_dir:
            try:
                # Check if PyKEEN model exists
                if os.path.exists(args.pykeen_model_dir):
                    pykeen_result = test_pykeen_inference(
                        atoms, args.batch_size, args.pykeen_model_dir, temp_dir
                    )
                    results.append(pykeen_result)
                else:
                    print(f"\n⚠ PyKEEN model directory not found: {args.pykeen_model_dir}")
            except Exception as e:
                print(f"\n❌ PyKEEN test failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Print comparison
        if results:
            print_comparison_results(results)
        else:
            print("\n❌ No successful tests completed")
            return 1
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
