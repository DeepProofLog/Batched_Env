#!/usr/bin/env python3
"""Test script to verify KGE engine selection works correctly."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_kge_engine_selection():
    """Test that different KGE backends can be selected."""
    from kge_inference import KGEInference, _get_backend_class
    
    print("Testing KGE backend selection...")
    
    # Test backend selection function
    backends_to_test = ['tf', 'tensorflow', 'pytorch', 'torch', 'pykeen']
    
    for backend in backends_to_test:
        try:
            BackendClass, normalized_name = _get_backend_class(backend)
            print(f"✓ Backend '{backend}' -> '{normalized_name}': {BackendClass.__name__}")
        except Exception as e:
            print(f"✗ Backend '{backend}' failed: {e}")
    
    # Test unknown backend (should fall back to TensorFlow)
    try:
        BackendClass, normalized_name = _get_backend_class('unknown')
        print(f"✓ Unknown backend fallback -> '{normalized_name}': {BackendClass.__name__}")
    except Exception as e:
        print(f"✗ Unknown backend fallback failed: {e}")
    
    print("\nBackend selection test completed!")


def test_config_validation():
    """Test that runner.py config validation works."""
    import argparse
    from runner import DEFAULT_CONFIG
    
    print("\nTesting runner.py config validation...")
    
    # Check that kge_engine is in default config
    if 'kge_engine' in DEFAULT_CONFIG:
        print(f"✓ 'kge_engine' found in DEFAULT_CONFIG with value: {DEFAULT_CONFIG['kge_engine']}")
    else:
        print("✗ 'kge_engine' not found in DEFAULT_CONFIG")
    
    print("\nConfig validation test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("KGE Engine Selection Test Suite")
    print("=" * 60)
    
    test_kge_engine_selection()
    test_config_validation()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
