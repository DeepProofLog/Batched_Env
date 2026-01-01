"""
Minimal PPO Evaluation - Raw Tensor Version with Zero-Arg Compiled Step.

Key optimization: The compiled step has ZERO arguments and operates on
persistent buffers via closure. This eliminates _foreach_copy_ overhead.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence

from env_evaluate_raw import EnvEvalRaw


class PPOEvalRaw:
    """Minimal PPO wrapper with zero-argument compiled step.
    
    Key optimizations:
    - Raw tensors (no TensorDict overhead)
    - Zero-argument compiled step (no input copying)
    - Persistent buffers updated in-place via closure
    """
    
    def __init__(
        self,
        policy,
        env: EnvEvalRaw,
        device: torch.device,
        mask_fill_value: float = -3.4e38,
    ):
        self.policy = policy
        self.env = env
        self.device = device
        self.mask_fill_value = mask_fill_value
        self._compiled_step = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Create and compile zero-argument step function.
        
        The step function captures env._obs and env._state via closure,
        so it has ZERO arguments. This eliminates CUDA graph input copying.
        """
        policy = self.policy
        env = self.env
        mask_fill_value = self.mask_fill_value
        
        # Capture persistent buffers via closure
        obs = env._obs
        state = env._state
        
        def zero_arg_step():
            """Zero-argument step - all state captured via closure."""
            # Get logits from policy
            # Policy expects dict-like obs, so we create a minimal wrapper
            logits = policy.get_logits({
                'sub_index': obs.sub_index,
                'derived_sub_indices': obs.derived_sub_indices,
                'action_mask': obs.action_mask,
            })
            
            # Mask and select action
            masked = torch.where(obs.action_mask.bool(), logits, mask_fill_value)
            actions = masked.argmax(dim=-1)
            
            # Step in-place (updates env._state and env._obs)
            env.step_inplace(actions)
            
            # Return just the tracking values we need
            return state.step_dones, state.step_rewards
        
        self._compiled_step = torch.compile(
            zero_arg_step,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[PPOEvalRaw] Compiled zero-arg step (mode={mode})")
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run evaluation with zero-argument compiled step."""
        device = self.device
        batch_size = self.env.batch_size
        max_depth = self.env.max_depth
        total_queries = len(queries)
        total_candidates_per_query = 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() before evaluate()")
        
        # 1. Generate ALL candidates
        all_pools = []
        for mode in corruption_modes:
            negative_candidates = sampler.corrupt(
                queries, num_negatives=n_corruptions, mode=mode, device=device
            )
            all_cands = torch.cat([
                queries.unsqueeze(1),
                negative_candidates
            ], dim=1)
            all_pools.append(all_cands.view(-1, 3))
        
        query_pool = torch.cat(all_pools, dim=0)
        pool_size = query_pool.size(0)
        
        if verbose:
            print(f"Eval: {pool_size} candidates, batch_size={batch_size}")
        
        # 2. Process in batched rounds
        n_rounds = (pool_size + batch_size - 1) // batch_size
        success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)
        
        for round_idx in range(n_rounds):
            start_idx = round_idx * batch_size
            end_idx = min(start_idx + batch_size, pool_size)
            round_size = end_idx - start_idx
            
            # Pad if needed
            round_queries = query_pool[start_idx:end_idx]
            if round_size < batch_size:
                padding = batch_size - round_size
                pad_queries = torch.zeros(padding, 3, dtype=torch.long, device=device)
                round_queries = torch.cat([round_queries, pad_queries], dim=0)
            
            # Reset (OUTSIDE compiled region)
            self.env.reset_into_buffers(round_queries)
            
            # Track completion
            slot_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            slot_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Run max_depth steps
            for step in range(max_depth):
                torch.compiler.cudagraph_mark_step_begin()
                
                # Zero-argument compiled step
                step_dones, step_rewards = self._compiled_step()
                
                # Track first finish
                just_done = step_dones
                step_success = (step_rewards > 0.5)
                
                newly_done = just_done & ~slot_finished
                slot_success = torch.where(newly_done, step_success, slot_success)
                slot_finished = slot_finished | just_done
            
            # Write results
            if round_size < batch_size:
                success_buffer[start_idx:end_idx] = slot_success[:round_size]
            else:
                success_buffer[start_idx:end_idx] = slot_success
        
        # 3. Compute metrics
        results = {}
        offset = 0
        
        for mode in corruption_modes:
            mode_size = total_queries * total_candidates_per_query
            mode_success = success_buffer[offset:offset + mode_size].view(
                total_queries, total_candidates_per_query
            )
            
            scores = torch.where(
                mode_success,
                torch.zeros_like(mode_success, dtype=torch.float32),
                torch.full_like(mode_success, -100.0, dtype=torch.float32)
            )
            
            pos_score = scores[:, 0:1]
            neg_scores = scores[:, 1:]
            
            rnd = torch.rand(total_queries, total_candidates_per_query, device=device)
            better = neg_scores > pos_score
            tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1])
            ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)
            
            mrr = (1.0 / ranks.float()).mean()
            hits1 = (ranks <= 1).float().mean()
            hits3 = (ranks <= 3).float().mean()
            hits10 = (ranks <= 10).float().mean()
            
            results[f'{mode}_mrr'] = mrr.item()
            results[f'{mode}_hits1'] = hits1.item()
            results[f'{mode}_hits3'] = hits3.item()
            results[f'{mode}_hits10'] = hits10.item()
            
            offset += mode_size
        
        n_modes = len(corruption_modes)
        results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
        results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
        results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
        results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes
        
        return results
