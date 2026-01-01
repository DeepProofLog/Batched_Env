"""
Minimal Eval V9 - Uses the same pattern as env.py rollout.

Key insight from env.py:
1. Pre-allocate a _state_buffer
2. Copy state TO buffer BEFORE calling compiled function  
3. Compiled function returns NEW tensors (no mutations inside)
4. Copy results FROM buffer AFTER calling compiled function

This keeps mutations OUTSIDE the compiled region.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence, Tuple


class MinimalEvalV9:
    """Eval using the buffer-copy pattern from env.py."""
    
    def __init__(
        self,
        policy,
        engine,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
    ):
        self.policy = policy
        self.engine = engine
        self.B = batch_size
        self.A = padding_atoms
        self.S = padding_states
        self.max_depth = max_depth
        self.device = device
        self.pad = engine.pad
        
        # Constants
        self._arange_B = torch.arange(batch_size, device=device)
        self._arange_S = torch.arange(padding_states, device=device)
        self._mask_fill_val = -3.4e38
        
        # Persistent state (updated outside compiled function)
        self._current = torch.full((batch_size, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._derived = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._mask = torch.zeros(batch_size, padding_states, dtype=torch.bool, device=device)
        self._depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._pool_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Pre-allocated buffer for compiled function (stable addresses)
        self._buf_current = torch.full((batch_size, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._buf_derived = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._buf_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._buf_mask = torch.zeros(batch_size, padding_states, dtype=torch.bool, device=device)
        self._buf_depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._buf_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._buf_pool_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Pool - pre-allocated
        self._max_pool_size = 50000
        self._pool = torch.zeros(self._max_pool_size, 3, dtype=torch.long, device=device)
        self._pool_size = 0
        self._result_buf = torch.zeros(self._max_pool_size, dtype=torch.bool, device=device)
        
        self._compiled_step = None
    
    def _copy_to_buffer(self):
        """Copy state TO buffer before compiled call."""
        torch._foreach_copy_(
            [self._buf_current, self._buf_derived, self._buf_counts, self._buf_mask, self._buf_depths, self._buf_done, self._buf_pool_ptr],
            [self._current, self._derived, self._counts, self._mask, self._depths, self._done, self._pool_ptr]
        )
    
    def _copy_from_results(self, new_current, new_derived, new_counts, new_mask, new_depths, new_done, new_ptr):
        """Copy results FROM compiled call to persistent state."""
        torch._foreach_copy_(
            [self._current, self._derived, self._counts, self._mask, self._depths, self._done, self._pool_ptr],
            [new_current, new_derived, new_counts, new_mask, new_depths, new_done, new_ptr]
        )
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile step function - returns NEW tensors, no mutations inside."""
        policy = self.policy
        engine = self.engine
        B, S, A, pad = self.B, self.S, self.A, self.pad
        max_depth = self.max_depth
        arange_B = self._arange_B
        arange_S = self._arange_S
        mask_fill_val = self._mask_fill_val
        device = self.device
        
        # Capture BUFFER references (stable addresses for CUDA graphs)
        buf_current = self._buf_current
        buf_derived = self._buf_derived
        buf_counts = self._buf_counts
        buf_mask = self._buf_mask
        buf_depths = self._buf_depths
        buf_done = self._buf_done
        buf_pool_ptr = self._buf_pool_ptr
        
        eval_obj = self
        
        def step_fn():
            """
            Pure functional step on buffer tensors.
            Returns new tensors - NO mutations to inputs.
            """
            # Get actions
            obs = {
                'sub_index': buf_current.unsqueeze(1),
                'derived_sub_indices': buf_derived,
                'action_mask': buf_mask.to(torch.uint8),
            }
            logits = policy.get_logits(obs)
            masked_logits = logits.masked_fill(~buf_mask, mask_fill_val)
            actions = masked_logits.argmax(dim=-1)
            
            # Step
            active = ~buf_done
            next_states = buf_derived[arange_B, actions]
            new_current = torch.where(active.view(B, 1, 1), next_states, buf_current)
            new_depths = torch.where(active, buf_depths + 1, buf_depths)
            
            # Check done
            is_terminal = (new_current[:, 0, 0] == pad)
            is_success = is_terminal & active
            truncated = (new_depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            finished_idx = torch.where(newly_done, buf_pool_ptr, torch.full_like(buf_pool_ptr, -1))
            new_ptr = torch.where(newly_done, buf_pool_ptr + B, buf_pool_ptr)
            needs_reset = newly_done & (new_ptr < eval_obj._pool_size)
            
            # Reset - compute derived for new queries
            safe_idx = new_ptr.clamp(0, max(0, eval_obj._pool_size - 1))
            new_queries_raw = eval_obj._pool[safe_idx]
            
            reset_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            reset_queries[:, 0, :] = new_queries_raw
            
            reset_derived, reset_counts = engine.get_derived(reset_queries)
            
            # Merge
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            
            final_current = torch.where(m1, reset_queries, new_current)
            final_derived = torch.where(m3, reset_derived, buf_derived)
            final_counts = torch.where(needs_reset, reset_counts, buf_counts)
            final_depths = torch.where(needs_reset, torch.zeros_like(new_depths), new_depths)
            final_done = torch.where(needs_reset, torch.zeros_like(buf_done), buf_done | newly_done)
            
            exhausted = new_ptr >= eval_obj._pool_size
            final_done = torch.where(exhausted & newly_done, torch.ones_like(final_done), final_done)
            
            new_mask = arange_S.unsqueeze(0) < final_counts.unsqueeze(1)
            
            # Return NEW tensors - no mutations
            return (
                final_current, final_derived, final_counts, new_mask,
                final_depths, final_done, new_ptr,
                newly_done, is_success, finished_idx,
            )
        
        self._compiled_step = torch.compile(step_fn, mode=mode, fullgraph=True, dynamic=False)
        print(f"[MinimalEvalV9] Compiled (mode={mode})")
    
    def _init_from_pool(self):
        B, A, pad = self.B, self.A, self.pad
        
        init_idx = torch.arange(B, device=self.device).clamp(max=max(0, self._pool_size - 1))
        queries_raw = self._pool[init_idx]
        
        queries = torch.full((B, A, 3), pad, dtype=torch.long, device=self.device)
        queries[:, 0, :] = queries_raw
        
        derived, counts = self.engine.get_derived(queries)
        
        self._current.copy_(queries)
        self._derived.copy_(derived)
        self._counts.copy_(counts)
        self._mask.copy_(self._arange_S.unsqueeze(0) < counts.unsqueeze(1))
        self._depths.zero_()
        self._done.zero_()
        self._pool_ptr.copy_(torch.arange(B, device=self.device))
    
    def setup_pool(self, queries: Tensor, sampler, n_corruptions: int, modes: Sequence[str]):
        pools = []
        for mode in modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=self.device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            pools.append(cands.view(-1, 3))
        
        new_pool = torch.cat(pools, dim=0)
        new_size = new_pool.size(0)
        
        if new_size > self._max_pool_size:
            raise ValueError(f"Pool size {new_size} exceeds max {self._max_pool_size}")
        
        self._pool[:new_size].copy_(new_pool)
        self._pool[new_size:].fill_(self.pad)
        self._pool_size = new_size
        self._result_buf.zero_()
        self._init_from_pool()
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        N, K = len(queries), 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() first")
        
        self.setup_pool(queries, sampler, n_corruptions, corruption_modes)
        
        if verbose:
            print(f"Pool: {self._pool_size}, batch: {self.B}")
        
        max_steps = (self._pool_size // self.B + 2) * self.max_depth
        steps = 0
        
        while steps < max_steps:
            torch.compiler.cudagraph_mark_step_begin()
            
            # 1. Copy TO buffer
            self._copy_to_buffer()
            
            # 2. Call compiled step (returns new tensors)
            (
                new_current, new_derived, new_counts, new_mask,
                new_depths, new_done, new_ptr,
                newly_done, success, indices,
            ) = self._compiled_step()
            
            # 3. Copy FROM results to persistent state
            self._copy_from_results(new_current, new_derived, new_counts, new_mask, new_depths, new_done, new_ptr)
            
            # Record results
            valid = (indices >= 0) & (indices < self._pool_size)
            if valid.any():
                safe_idx = torch.where(valid, indices, torch.zeros_like(indices))
                safe_val = torch.where(valid, success, torch.zeros_like(success))
                self._result_buf.scatter_(0, safe_idx, safe_val)
            
            steps += 1
            
            if self._done.all():
                break
        
        if verbose:
            print(f"Steps: {steps}")
        
        results = {}
        offset = 0
        for mode in corruption_modes:
            ms = self._result_buf[offset:offset + N * K].view(N, K)
            scores = torch.where(ms, torch.zeros(N, K, device=self.device),
                               torch.full((N, K), -100.0, device=self.device))
            
            pos, neg = scores[:, 0:1], scores[:, 1:]
            rnd = torch.rand(N, K, device=self.device)
            better = neg > pos
            tied = (neg == pos) & (rnd[:, 1:] > rnd[:, 0:1])
            ranks = 1 + better.sum(1) + tied.sum(1)
            
            results[f'{mode}_mrr'] = (1.0 / ranks.float()).mean().item()
            results[f'{mode}_hits10'] = (ranks <= 10).float().mean().item()
            offset += N * K
        
        nm = len(corruption_modes)
        results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / nm
        results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / nm
        
        return results
