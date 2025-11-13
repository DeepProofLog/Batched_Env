"""
Profile test_engine_equivalence to find bottlenecks
"""
import cProfile
import pstats
import io
import random
import torch
from typing import List

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

# String-engine stack
from str_based.str_dataset import DataHandler as StrDataHandler
from str_based.str_index_manager import IndexManager as StrIndexManager
from str_based.str_utils import Term as StrTerm
from str_based.str_unification import get_next_unification_python

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from debug_helper import DebugHelper

DATASET = "family"

def setup_engines():
    """Setup both engines"""
    # Str engine setup
    dh_str = StrDataHandler(
        dataset_name=DATASET,
        base_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )

    im_str = StrIndexManager(
        constants=dh_str.constants,
        predicates=dh_str.predicates,
        max_total_vars=1000000,
        rules=dh_str.rules,
        padding_atoms=20,
        max_arity=dh_str.max_arity,
        device=torch.device('cpu'),
    )
    fact_index_str = im_str.build_fact_index(dh_str.facts)

    rules_by_pred = {}
    for r in dh_str.rules:
        rules_by_pred.setdefault(r.head.predicate, []).append(r)

    # Tensor engine setup
    dh_non = DataHandler(
        dataset_name=DATASET,
        base_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )

    im_non = IndexManager(
        constants=dh_non.constants,
        predicates=dh_non.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=20,
        max_arity=dh_non.max_arity,
        device=torch.device('cpu'),
        rules=dh_non.rules,
    )
    dh_non.materialize_indices(im=im_non, device=torch.device('cpu'))

    engine = UnificationEngine.from_index_manager(im_non, take_ownership=True)

    debug_helper = DebugHelper(
        verbose=0,
        idx2predicate=im_non.idx2predicate,
        idx2constant=im_non.idx2constant,
        idx2template_var=im_non.idx2template_var,
        padding_idx=engine.padding_idx,
        n_constants=im_non.constant_no
    )

    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = rule_max
    next_var_start = max_template_var + 1

    str_engine_data = (dh_str, im_str, fact_index_str, rules_by_pred)
    tensor_engine_data = (dh_non, im_non, engine, debug_helper, next_var_start)
    
    return dh_str, str_engine_data, tensor_engine_data


def test_query(p, h, t, str_engine_data, tensor_engine_data, split='train'):
    """Test a single query - simplified version"""
    dh_str, im_str, fact_index_str, rules_by_pred = str_engine_data
    dh_non, im_non, engine, debug_helper, next_var_start = tensor_engine_data
    
    # Convert facts list to frozenset for O(1) lookup performance
    facts_set_str = frozenset(dh_str.facts)
    
    # Setup for both engines
    q_str = StrTerm(predicate=p, args=(h, t))
    query_tensor = im_non.atom_to_tensor(p, h, t)
    query_padded_tensor = query_tensor.unsqueeze(0).unsqueeze(0)
    padding = torch.zeros(1, 19, 3, dtype=torch.long, device='cpu')
    padding[:, :, 0] = engine.padding_idx
    query_padded = torch.cat([query_padded_tensor, padding], dim=1)
    
    excluded_fact_str = q_str if split == 'train' else None
    excluded_query_tensor = query_padded if split == 'train' else None
    
    # String engine
    state_str = [q_str]
    for step in range(3):  # Just 3 steps
        derived_str, _ = get_next_unification_python(
            state_str, facts_set_str, fact_index_str, rules_by_pred,
            excluded_fact=excluded_fact_str, verbose=0, next_var_index=1
        )
        if not derived_str:
            break
        state_str = derived_str[0]
    
    # Tensor engine
    next_var_tracker = torch.tensor([next_var_start], dtype=torch.long, device='cpu')
    state_tensor = query_padded
    for step in range(3):  # Just 3 steps
        derived, derived_counts, next_var_tracker = engine.get_derived_states(
            state_tensor, next_var_tracker,
            excluded_queries=excluded_query_tensor, verbose=0
        )
        if derived_counts[0] == 0:
            break
        state_tensor = derived[0, 0].unsqueeze(0)


def run_profile():
    """Run profiling on a few queries"""
    print("Setting up engines...")
    dh_str, str_engine_data, tensor_engine_data = setup_engines()
    
    # Get first 10 queries
    queries = []
    for i, q in enumerate(dh_str.train_queries[:10]):
        queries.append(('train', q.predicate, q.args[0], q.args[1]))
    
    print(f"Profiling {len(queries)} queries...")
    
    # Profile the execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    for split, p, h, t in queries:
        test_query(p, h, t, str_engine_data, tensor_engine_data, split)
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (by cumulative time)")
    print("="*80)
    print(s.getvalue())
    
    # Also print by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (by total time)")
    print("="*80)
    print(s.getvalue())


if __name__ == "__main__":
    run_profile()
