"""
Minimal PPO - Pure Functional Zero-Arg Step.

Compiled step is pure functional (returns new tensors).
Copying happens OUTSIDE compiled region.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence, Tuple

from env_evaluate_pure import EnvEvalPure


class PPOEvalPure:
    """PPO with pure functional zero-arg compiled step."""
    
    def __init__(
        self,
        policy,
        env: EnvEvalPure,
        device: torch.device,
        mask_fill_value: float = -3.4e38,
    ):
        self.policy = policy
        self.env = env
        self.device = device
        self.mask_fill_value = mask_fill_value
        self._compiled_step = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Create zero-arg compiled step that is pure functional."""
        policy = self.policy
        env = self.env
        mask_fill_value = self.mask_fill_value
        
        def zero_arg_step() -> Tuple[Tensor, ...]:
            """Pure functional - reads from buffers via closure, returns NEW tensors."""
            # Get logits (policy expects dict-like input)
            logits = policy.get_logits({
                'sub_index': env._current_states.unsqueeze(1),  # [B, 1, A, 3]
                'derived_sub_indices': env._derived_states,  # [B, S, A, 3]
                'action_mask': env._action_mask,  # [B, S]
            })
            
            masked = torch.where(env._action_mask.bool(), logits, mask_fill_value)
            actions = masked.argmax(dim=-1)
            
            # Pure step - returns new tensors, no mutations
            return env.step_pure(actions)
        
        self._compiled_step = torch.compile(
            zero_arg_step,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[PPOEvalPure] Compiled pure zero-arg step (mode={mode})")
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Evaluate with pure functional step + copy-outside pattern."""
        device = self.device
        batch_size = self.env.batch_size
        max_depth = self.env.max_depth
        total_queries = len(queries)
        total_candidates = 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() first")
        
        # Generate candidates
        all_pools = []
        for mode in corruption_modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            all_cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            all_pools.append(all_cands.view(-1, 3))
        
        query_pool = torch.cat(all_pools, dim=0)
        pool_size = query_pool.size(0)
        
        if verbose:
            print(f"Eval: {pool_size} candidates, batch={batch_size}")
        
        n_rounds = (pool_size + batch_size - 1) // batch_size
        success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)
        
        for round_idx in range(n_rounds):
            start = round_idx * batch_size
            end = min(start + batch_size, pool_size)
            size = end - start
            
            round_queries = query_pool[start:end]
            if size < batch_size:
                pad = torch.zeros(batch_size - size, 3, dtype=torch.long, device=device)
                round_queries = torch.cat([round_queries, pad], dim=0)
            
            # Reset (OUTSIDE compiled region)
            self.env.reset(round_queries)
            
            slot_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            slot_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for step in range(max_depth):
                torch.compiler.cudagraph_mark_step_begin()
                
                # Pure compiled step returns new tensors
                results = self._compiled_step()
                
                # Copy OUTSIDE compiled region
                self.env.copy_step_results(results)
                
                # Track completion
                step_dones = self.env._step_dones
                step_success = (self.env._step_rewards > 0.5)
                
                newly_done = step_dones & ~slot_finished
                slot_success = torch.where(newly_done, step_success, slot_success)
                slot_finished = slot_finished | step_dones
            
            if size < batch_size:
                success_buffer[start:end] = slot_success[:size]
            else:
                success_buffer[start:end] = slot_success
        
        # Compute metrics
        results = {}
        offset = 0
        
        for mode in corruption_modes:
            mode_size = total_queries * total_candidates
            mode_success = success_buffer[offset:offset + mode_size].view(total_queries, total_candidates)
            
            scores = torch.where(mode_success,
                                torch.zeros_like(mode_success, dtype=torch.float32),
                                torch.full_like(mode_success, -100.0, dtype=torch.float32))
            
            pos = scores[:, 0:1]
            neg = scores[:, 1:]
            rnd = torch.rand(total_queries, total_candidates, device=device)
            better = neg > pos
            tied = (neg == pos) & (rnd[:, 1:] > rnd[:, 0:1])
            ranks = 1 + better.sum(1) + tied.sum(1)
            
            mrr = (1.0 / ranks.float()).mean().item()
            results[f'{mode}_mrr'] = mrr
            results[f'{mode}_hits1'] = (ranks <= 1).float().mean().item()
            results[f'{mode}_hits3'] = (ranks <= 3).float().mean().item()
            results[f'{mode}_hits10'] = (ranks <= 10).float().mean().item()
            
            offset += mode_size
        
        n_modes = len(corruption_modes)
        results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
        results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
        results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
        results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes
        
        return results
