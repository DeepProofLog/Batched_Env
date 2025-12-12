"""
this was designed to compile separately the the Policy (~30s) and Trajectory Loop (~30s)
"""
@torch.inference_mode()
def eval_corruptions_fast(
    evaluator: CompiledEvaluator,
    queries: Tensor,
    sampler: Any,
    *,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ("head", "tail"),
    chunk_queries: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fast corruption evaluation using pre-warmed CompiledEvaluator.
    
    Usage:
        # One-time setup (26s)
        evaluator = CompiledEvaluator(env, policy_fn, batch_size=1020, max_steps=20)
        evaluator.warmup(queries[:10])
        
        # Fast evaluation (~2s for 100 queries)
        results = eval_corruptions_fast(evaluator, queries, sampler)
    """
    device = evaluator.env.device
    N = queries.shape[0]
    K = n_corruptions
    
    per_mode_ranks: Dict[str, list] = {m: [] for m in corruption_modes}
    
    for start in range(0, N, chunk_queries):
        end = min(start + chunk_queries, N)
        Q = end - start
        
        if verbose:
            print(f"Processing queries {start}-{end} / {N}")
        
        chunk_queries_tensor = queries[start:end]
        
        for mode in corruption_modes:
            corruptions = sampler.corrupt(
                chunk_queries_tensor, 
                num_negatives=K, 
                mode=mode, 
                device=device
            )
            
            valid_mask = corruptions.sum(dim=-1) != 0
            
            candidates = torch.zeros(Q, 1 + K, 3, dtype=torch.long, device=device)
            candidates[:, 0, :] = chunk_queries_tensor
            candidates[:, 1:, :] = corruptions
            
            flat_candidates = candidates.view(-1, 3)
            
            # Use pre-warmed evaluator
            log_probs, success, lengths, rewards = evaluator(flat_candidates)
            
            log_probs = log_probs.view(Q, 1 + K)
            success = success.view(Q, 1 + K)
            
            full_valid = torch.ones(Q, 1 + K, dtype=torch.bool, device=device)
            full_valid[:, 1:] = valid_mask
            
            pos_log_prob = log_probs[:, 0:1]
            scores = log_probs.clone()
            scores[~full_valid] = float('-inf')
            
            higher = (scores[:, 1:] > pos_log_prob).float() * full_valid[:, 1:].float()
            ties = (scores[:, 1:] == pos_log_prob).float() * full_valid[:, 1:].float()
            ranks = 1 + higher.sum(dim=1) + 0.5 * ties.sum(dim=1)
            
            per_mode_ranks[mode].append(ranks)
    
    results: Dict[str, Any] = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    per_mode_results: Dict[str, Dict[str, float]] = {}
    
    for mode in corruption_modes:
        if per_mode_ranks[mode]:
            all_ranks = torch.cat(per_mode_ranks[mode])
            per_mode_results[mode] = compute_metrics_from_ranks(all_ranks)
        else:
            per_mode_results[mode] = compute_metrics_from_ranks(torch.tensor([], device=device))
    
    for mode in corruption_modes:
        for k, v in per_mode_results[mode].items():
            results[k] += v
    
    n_modes = len(corruption_modes)
    for k in results:
        results[k] /= n_modes if n_modes > 0 else 1.0
    
    results["per_mode"] = per_mode_results
    results["_mrr"] = results["MRR"]
    
    if verbose:
        print(f"\nResults: MRR={results['MRR']:.4f}, Hits@10={results['Hits@10']:.4f}")
    
    return results
