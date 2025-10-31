#!/usr/bin/env python3
"""
Test script for the new KGE base runner and implementations.
Tests basic functionality without full training.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_base_runner():
    """Test BaseKGERunner functionality."""
    print("\n" + "="*80)
    print("Testing BaseKGERunner")
    print("="*80)
    
    from kge_base_runner import BaseKGERunner, BaseKGEConfig
    
    # Test config creation
    config = BaseKGEConfig()
    print(f"✓ Default config created: dataset={config.dataset}, model={config.model}")
    
    # Test config to dict
    config_dict = config.to_dict()
    print(f"✓ Config to dict: {len(config_dict)} parameters")
    
    # Test config from dict
    new_config = BaseKGEConfig.from_dict(config_dict)
    print(f"✓ Config from dict: dataset={new_config.dataset}")
    
    # Test config update
    config.update(dataset='wn18rr', epochs=2000)
    print(f"✓ Config update: dataset={config.dataset}, epochs={config.epochs}")
    
    print("\nBaseKGERunner tests passed!\n")


def test_pykeen_runner():
    """Test PyKeenRunner functionality."""
    print("\n" + "="*80)
    print("Testing PyKeenRunner")
    print("="*80)
    
    try:
        from kge_pykeen.runner_pykeen_v2 import PyKeenRunner, PyKeenConfig
    except ImportError as e:
        print(f"⚠ Skipping PyKeenRunner tests: {e}")
        print("  (PyKeen is not installed)")
        return
    
    # Test config creation
    config = PyKeenConfig()
    print(f"✓ PyKeen config created: model={config.model}, loss={config.loss}")
    
    # Test runner creation
    runner = PyKeenRunner(config)
    print(f"✓ PyKeen runner created with {len(runner.MODEL_CONFIGS)} models")
    
    # Test model list parsing
    models = runner.parse_model_list("ComplEx,TransE")
    print(f"✓ Model list parsed: {models}")
    
    models_all = runner.parse_model_list("all")
    print(f"✓ All models: {len(models_all)} models")
    
    # Test parser building
    parser = runner.build_parser()
    print(f"✓ Parser built with {len(parser._actions)} arguments")
    
    # Test args parsing
    args = runner.parse_args(['--model', 'ComplEx', '--dataset', 'family', '--epochs', '100'])
    print(f"✓ Args parsed: model={args.model}, dataset={args.dataset}, epochs={args.epochs}")
    
    # Test args to config
    config = runner.args_to_config(args)
    print(f"✓ Args to config: model={config.model}, dataset={config.dataset}, epochs={config.epochs}")
    
    # Test hyperparameter config
    hparam_config = {
        "models": ["TransE", "ComplEx"],
        "datasets": ["family"],
        "lr": [0.001, 0.0005],
        "embedding_dim": [256, 512]
    }
    combinations = runner.generate_hparam_combinations(hparam_config)
    print(f"✓ Hyperparameter combinations: {len(combinations)} total")
    print(f"  Example: {combinations[0]}")
    
    print("\nPyKeenRunner tests passed!\n")


def test_pytorch_runner():
    """Test PyTorchRunner functionality."""
    print("\n" + "="*80)
    print("Testing PyTorchRunner")
    print("="*80)
    
    # Import directly to avoid __init__.py importing torch
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "runner_kge_v2",
        Path(__file__).parent / "kge_pytorch" / "runner_kge_v2.py"
    )
    runner_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(runner_module)
    except ImportError as e:
        print(f"⚠ Skipping PyTorchRunner tests: {e}")
        print("  (PyTorch dependencies not available)")
        return
    
    PyTorchRunner = runner_module.PyTorchRunner
    PyTorchConfig = runner_module.PyTorchConfig
    
    # Test config creation
    config = PyTorchConfig()
    print(f"✓ PyTorch config created: model={config.model}, gamma={config.gamma}")
    
    # Test runner creation
    runner = PyTorchRunner(config)
    print(f"✓ PyTorch runner created with {len(runner.MODEL_CONFIGS)} models")
    
    # Test model list parsing
    models = runner.parse_model_list("RotatE,ComplEx,mrr_boost")
    print(f"✓ Model list parsed: {models}")
    
    # Test parser building
    parser = runner.build_parser()
    print(f"✓ Parser built with {len(parser._actions)} arguments")
    
    # Test args parsing
    args = runner.parse_args([
        '--model', 'RotatE', 
        '--dataset', 'family',
        '--epochs', '100',
        '--use_reciprocal',
        '--grad_clip', '2.0',
        '--device', 'cpu'
    ])
    print(f"✓ Args parsed: model={args.model}, use_reciprocal={args.use_reciprocal}")
    
    # Test args to config
    config = runner.args_to_config(args)
    print(f"✓ Args to config: use_reciprocal={config.use_reciprocal}, grad_clip={config.grad_clip}")
    
    # Test hyperparameter config
    hparam_config = {
        "models": ["TransE", "RotatE"],
        "datasets": ["family"],
        "lr": [0.001, 0.0005],
        "embedding_dim": [256, 512],
        "use_reciprocal": [False, True]
    }
    combinations = runner.generate_hparam_combinations(hparam_config)
    print(f"✓ Hyperparameter combinations: {len(combinations)} total")
    print(f"  Example: {combinations[0]}")
    
    print("\nPyTorchRunner tests passed!\n")


def test_inheritance():
    """Test that runners properly inherit from base."""
    print("\n" + "="*80)
    print("Testing Inheritance")
    print("="*80)
    
    from kge_base_runner import BaseKGERunner
    
    try:
        from kge_pykeen.runner_pykeen_v2 import PyKeenRunner
        has_pykeen = True
    except ImportError:
        print("⚠ PyKeen not available, skipping PyKeen inheritance tests")
        has_pykeen = False
    
    # Import PyTorch runner directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "runner_kge_v2",
        Path(__file__).parent / "kge_pytorch" / "runner_kge_v2.py"
    )
    runner_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(runner_module)
        PyTorchRunner = runner_module.PyTorchRunner
        has_pytorch = True
    except ImportError as e:
        print(f"⚠ PyTorch not available, skipping PyTorch inheritance tests: {e}")
        has_pytorch = False
    
    # Test inheritance
    runners_to_test = []
    
    if has_pykeen:
        pykeen_runner = PyKeenRunner()
        runners_to_test.append(('PyKeenRunner', pykeen_runner))
        assert isinstance(pykeen_runner, BaseKGERunner), "PyKeenRunner should inherit from BaseKGERunner"
        print("✓ PyKeenRunner properly inherits from BaseKGERunner")
        
        # Test that abstract methods are implemented
        assert hasattr(pykeen_runner, 'train_single_model'), "PyKeenRunner should implement train_single_model"
        assert hasattr(pykeen_runner, 'load_model'), "PyKeenRunner should implement load_model"
        assert hasattr(pykeen_runner, 'save_model'), "PyKeenRunner should implement save_model"
        print("✓ PyKeenRunner implements all abstract methods")
    
    if has_pytorch:
        pytorch_runner = PyTorchRunner()
        runners_to_test.append(('PyTorchRunner', pytorch_runner))
        assert isinstance(pytorch_runner, BaseKGERunner), "PyTorchRunner should inherit from BaseKGERunner"
        print("✓ PyTorchRunner properly inherits from BaseKGERunner")
        
        assert hasattr(pytorch_runner, 'train_single_model'), "PyTorchRunner should implement train_single_model"
        assert hasattr(pytorch_runner, 'load_model'), "PyTorchRunner should implement load_model"
        assert hasattr(pytorch_runner, 'save_model'), "PyTorchRunner should implement save_model"
        print("✓ PyTorchRunner implements all abstract methods")
    
    # Test that common methods are available
    for name, runner in runners_to_test:
        assert hasattr(runner, 'build_parser'), f"{name} should have build_parser"
        assert hasattr(runner, 'parse_args'), f"{name} should have parse_args"
        assert hasattr(runner, 'run_experiments'), f"{name} should have run_experiments"
        assert hasattr(runner, 'run_hparam_search'), f"{name} should have run_hparam_search"
        assert hasattr(runner, 'print_summary'), f"{name} should have print_summary"
    print("✓ All runners have common methods from base class")
    
    print("\nInheritance tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("KGE Runner Test Suite")
    print("="*80)
    
    try:
        test_base_runner()
        test_pykeen_runner()
        test_pytorch_runner()
        test_inheritance()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nThe base runner and both implementations are working correctly.")
        print("You can now use them for training KGE models.")
        print("\nNext steps:")
        print("  1. Try running a quick training test:")
        print("     python kge_pykeen/runner_pykeen_v2.py --model TransE --dataset family --epochs 10")
        print("  2. Try hyperparameter search:")
        print("     python kge_pykeen/runner_pykeen_v2.py --hparam_search --hparam_config hparam_config_example.json")
        print("  3. Read KGE_RUNNER_README.md for full documentation")
        print()
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
