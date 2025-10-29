#!/usr/bin/env python3
"""Simple test to verify KGE backend parameter configuration."""

def test_backend_parameter():
    """Test that the backend parameter is correctly added to configs."""
    
    print("=" * 60)
    print("KGE Backend Parameter Configuration Test")
    print("=" * 60)
    
    # Test 1: Check runner.py DEFAULT_CONFIG
    print("\nTest 1: Checking runner.py DEFAULT_CONFIG...")
    try:
        with open('runner.py', 'r') as f:
            content = f.read()
            if "'kge_engine': 'tf'" in content:
                print("✓ 'kge_engine' parameter found in runner.py DEFAULT_CONFIG")
            else:
                print("✗ 'kge_engine' parameter NOT found in runner.py DEFAULT_CONFIG")
    except Exception as e:
        print(f"✗ Error reading runner.py: {e}")
    
    # Test 2: Check runner_new.py DEFAULT_CONFIG
    print("\nTest 2: Checking runner_new.py DEFAULT_CONFIG...")
    try:
        with open('runner_new.py', 'r') as f:
            content = f.read()
            if "'kge_engine': 'tf'" in content:
                print("✓ 'kge_engine' parameter found in runner_new.py DEFAULT_CONFIG")
            else:
                print("✗ 'kge_engine' parameter NOT found in runner_new.py DEFAULT_CONFIG")
    except Exception as e:
        print(f"✗ Error reading runner_new.py: {e}")
    
    # Test 3: Check train.py passes backend parameter
    print("\nTest 3: Checking train.py KGE engine initialization...")
    try:
        with open('train.py', 'r') as f:
            content = f.read()
            if 'backend=kge_engine_backend' in content or 'backend=' in content:
                print("✓ train.py passes backend parameter to KGEInference")
            else:
                print("✗ train.py does NOT pass backend parameter to KGEInference")
    except Exception as e:
        print(f"✗ Error reading train.py: {e}")
    
    # Test 4: Check kge_integration.py passes backend parameter
    print("\nTest 4: Checking kge_integration.py KGE engine initialization...")
    try:
        with open('kge_integration.py', 'r') as f:
            content = f.read()
            if 'backend=kge_engine_backend' in content or 'backend=' in content:
                print("✓ kge_integration.py passes backend parameter to KGEInference")
            else:
                print("✗ kge_integration.py does NOT pass backend parameter to KGEInference")
    except Exception as e:
        print(f"✗ Error reading kge_integration.py: {e}")
    
    # Test 5: Check kge_inference.py wrapper accepts backend parameter
    print("\nTest 5: Checking kge_inference.py wrapper...")
    try:
        with open('kge_inference.py', 'r') as f:
            content = f.read()
            if 'backend: str = "tf"' in content:
                print("✓ kge_inference.py KGEInference accepts backend parameter")
            else:
                print("✗ kge_inference.py KGEInference does NOT accept backend parameter")
            
            if '_get_backend_class' in content:
                print("✓ kge_inference.py has _get_backend_class function")
            else:
                print("✗ kge_inference.py does NOT have _get_backend_class function")
    except Exception as e:
        print(f"✗ Error reading kge_inference.py: {e}")
    
    # Test 6: Check wrapper files exist
    print("\nTest 6: Checking KGE backend wrapper files...")
    import os
    
    pytorch_wrapper = 'kge_pytorch/kge_inference_wrapper.py'
    if os.path.exists(pytorch_wrapper):
        print(f"✓ PyTorch wrapper found: {pytorch_wrapper}")
    else:
        print(f"✗ PyTorch wrapper NOT found: {pytorch_wrapper}")
    
    pykeen_wrapper = 'kge_pykeen/kge_inference_wrapper.py'
    if os.path.exists(pykeen_wrapper):
        print(f"✓ PyKEEN wrapper found: {pykeen_wrapper}")
    else:
        print(f"✗ PyKEEN wrapper NOT found: {pykeen_wrapper}")
    
    # Test 7: Check runner.py has backend-specific signatures
    print("\nTest 7: Checking backend-specific KGE run signatures...")
    try:
        with open('runner.py', 'r') as f:
            content = f.read()
            if 'backend = cfg[\'kge_engine\']' in content:
                print("✓ runner.py uses backend to determine KGE run signature")
            else:
                print("✗ runner.py does NOT use backend for KGE run signature")
    except Exception as e:
        print(f"✗ Error reading runner.py: {e}")
    
    print("\n" + "=" * 60)
    print("Configuration test completed!")
    print("=" * 60)


if __name__ == "__main__":
    import os
    os.chdir('/home/castellanoontiv/RL_main/Neural-guided-Grounding')
    test_backend_parameter()
