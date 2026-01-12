"""
Weight mapping utilities for parity testing between tensor and optimized policies.

When parity=True, both policies use nn.Sequential with indexed keys:

Tensor policy (tensor/tensor_model.py):
    - mlp_extractor.shared_network.shared_body.input_transform.0.weight
    - mlp_extractor.shared_network.shared_body.res_blocks.{i}.0.weight
    - mlp_extractor.shared_network.policy_head.out_transform.0.weight
    - mlp_extractor.shared_network.value_head.output_layer.0.weight

Optimized policy (policy.py) with parity=True:
    - mlp_extractor.shared_body.input_transform.0.weight
    - mlp_extractor.shared_body.res_blocks.{i}.0.weight
    - mlp_extractor.policy_head_fused.0.weight
    - mlp_extractor.policy_head_final.weight
    - mlp_extractor.value_head_fused.0.weight
    - mlp_extractor.value_head_final.weight

Both architectures are semantically equivalent when dropout=0.
"""
import re
from typing import Dict
import torch


def map_tensor_to_optimized_state_dict(tensor_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map tensor policy state_dict keys to optimized policy format (parity=True).
    
    This enables loading weights from a tensor policy into an optimized policy
    for parity testing. Both architectures produce identical outputs when
    dropout=0, but use different module structures.
    
    Args:
        tensor_state: State dict from tensor policy (tensor/tensor_model.py)
        
    Returns:
        State dict with keys mapped to optimized policy format (policy.py with parity=True)
    """
    mapped_state = {}
    
    for key, value in tensor_state.items():
        new_key = key
        
        # Handle torch.compile prefix
        new_key = new_key.replace('_orig_mod.', '')
        
        # Handle mlp_extractor.shared_network.shared_body.X -> mlp_extractor.shared_body.X
        new_key = new_key.replace('mlp_extractor.shared_network.shared_body.', 'mlp_extractor.shared_body.')
        
        # === Policy head mappings (tensor -> optimized parity structure) ===
        # policy_head.out_transform.0.weight/bias -> policy_head_fused.0.weight/bias
        new_key = re.sub(r'mlp_extractor\.shared_network\.policy_head\.out_transform\.0\.(weight|bias)', 
                        r'mlp_extractor.policy_head_fused.0.\1', new_key)
        
        # policy_head.out_transform.2.weight/bias -> policy_head_final.weight/bias
        new_key = re.sub(r'mlp_extractor\.shared_network\.policy_head\.out_transform\.2\.(weight|bias)', 
                        r'mlp_extractor.policy_head_final.\1', new_key)
        
        # === Value head mappings (tensor -> optimized parity structure) ===
        # value_head.output_layer.0.weight/bias -> value_head_fused.0.weight/bias
        new_key = re.sub(r'mlp_extractor\.shared_network\.value_head\.output_layer\.0\.(weight|bias)', 
                        r'mlp_extractor.value_head_fused.0.\1', new_key)
        
        # value_head.output_layer.2.weight/bias -> value_head_final.weight/bias
        new_key = re.sub(r'mlp_extractor\.shared_network\.value_head\.output_layer\.2\.(weight|bias)', 
                        r'mlp_extractor.value_head_final.\1', new_key)
        
        mapped_state[new_key] = value
    
    return mapped_state


def map_optimized_to_tensor_state_dict(optimized_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map optimized policy state_dict keys to tensor policy format.
    
    Inverse of map_tensor_to_optimized_state_dict.
    
    Args:
        optimized_state: State dict from optimized policy (policy.py with parity=True)
        
    Returns:
        State dict with keys mapped to tensor policy format (tensor/tensor_model.py)
    """
    mapped_state = {}
    
    for key, value in optimized_state.items():
        new_key = key
        
        # Handle torch.compile prefix
        new_key = new_key.replace('_orig_mod.', '')
        
        # Handle mlp_extractor.shared_body.X -> mlp_extractor.shared_network.shared_body.X
        new_key = new_key.replace('mlp_extractor.shared_body.', 'mlp_extractor.shared_network.shared_body.')
        
        # === Policy head inverse mappings ===
        # policy_head_fused.0.weight/bias -> policy_head.out_transform.0.weight/bias
        new_key = re.sub(r'mlp_extractor\.policy_head_fused\.0\.(weight|bias)', 
                        r'mlp_extractor.shared_network.policy_head.out_transform.0.\1', new_key)
        
        # policy_head_final.weight/bias -> policy_head.out_transform.2.weight/bias
        new_key = re.sub(r'mlp_extractor\.policy_head_final\.(weight|bias)', 
                        r'mlp_extractor.shared_network.policy_head.out_transform.2.\1', new_key)
        
        # === Value head inverse mappings ===
        # value_head_fused.0.weight/bias -> value_head.output_layer.0.weight/bias
        new_key = re.sub(r'mlp_extractor\.value_head_fused\.0\.(weight|bias)', 
                        r'mlp_extractor.shared_network.value_head.output_layer.0.\1', new_key)
        
        # value_head_final.weight/bias -> value_head.output_layer.2.weight/bias
        new_key = re.sub(r'mlp_extractor\.value_head_final\.(weight|bias)', 
                        r'mlp_extractor.shared_network.value_head.output_layer.2.\1', new_key)
        
        mapped_state[new_key] = value
    
    return mapped_state
