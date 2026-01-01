"""
Minimal Eval V6 - Zero-arg compiled step for proper CUDA graph capture.

Key: Zero-argument step function captures state via closure.
Copy happens by having step write to output buffers, not input mutation.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence, Tuple

from engine_minimal import EngineMinimal


class MinimalEvalV6:
    """Eval with zero-arg step for CUDA graph capture."""
    
    def __init__(
        self,
        policy,
        engine: EngineMinimal,
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
        self._ones_B = torch.ones(batch_size, dtype=torch.long, device=device)
        self._false_state = torch.full((padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._mask_fill = torch.full((batch_size, padding_states), -3.4e38, device=device)
        
        # INPUT buffers (read by step)
        self._current = torch.full((batch_size, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._derived = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._mask = torch.zeros(batch_size, padding_states, dtype=torch.bool, device=device)
        self._depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._pool_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # OUTPUT buffers (written by step)
        self._out_current = torch.full((batch_size, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._out_derived = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._out_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._out_mask = torch.zeros(batch_size, padding_states, dtype=torch.bool, device=device)
        self._out_depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._out_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._out_pool_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._out_newly_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._out_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._out_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Pool
        self._pool = None
        self._pool_size = 0
        self._result_buf = None
        
        self._compiled_step = None
    
    def _init_from_pool(self):
        """Initialize first B slots."""
        B, A, S, pad = self.B, self.A, self.S, self.pad
        
        init_idx = torch.arange(B, device=self.device).clamp(max=max(0, self._pool_size - 1))
        queries_raw = self._pool[init_idx]
        
        queries = torch.full((B, A, 3), pad, dtype=torch.long, device=self.device)
        queries[:, 0, :] = queries_raw
        
        derived, counts = self.engine.get_derived(queries)
        
        self._current.copy_(queries)
        self._derived.copy_(derived)
        self._counts.copy_(counts)
        self._mask = self._arange_S.unsqueeze(0) < counts.unsqueeze(1)
        self._depths.zero_()
        self._done.zero_()
        self._pool_ptr = torch.arange(B, device=self.device)
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile zero-arg step that reads from input buffers, writes to output buffers."""
        policy = self.policy
        engine = self.engine
        B, S, A, pad = self.B, self.S, self.A, self.pad
        max_depth = self.max_depth
        arange_B = self._arange_B
        arange_S = self._arange_S
        ones_B = self._ones_B
        false_state = self._false_state
        mask_fill = self._mask_fill
        device = self.device
        
        # Capture input buffers
        current = self._current
        derived = self._derived
        counts = self._counts
        mask = self._mask
        depths = self._depths
        done = self._done
        pool_ptr = self._pool_ptr
        
        # Capture output buffers
        out_current = self._out_current
        out_derived = self._out_derived
        out_counts = self._out_counts
        out_mask = self._out_mask
        out_depths = self._out_depths
        out_done = self._out_done
        out_pool_ptr = self._out_pool_ptr
        out_newly_done = self._out_newly_done
        out_success = self._out_success
        out_indices = self._out_indices
        
        eval_obj = self
        
        def step_and_reset():
            """Zero-arg step: reads input buffers, writes to output buffers."""
            # 1. Get actions
            logits = policy.get_logits({
                'sub_index': current.unsqueeze(1),
                'derived_sub_indices': derived,
                'action_mask': mask.to(torch.uint8),
            })
            masked_logits = torch.where(mask, logits, mask_fill)
            actions = masked_logits.argmax(dim=-1)
            
            # 2. Step
            active = ~done
            next_states = derived[arange_B, actions]
            new_current = torch.where(active.view(B, 1, 1), next_states, current)
            new_depths = torch.where(active, depths + 1, depths)
            
            # 3. Check done
            is_terminal = (new_current[:, 0, 0] == pad)
            is_success = is_terminal & active
            truncated = (new_depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            finished_idx = torch.where(newly_done, pool_ptr, torch.full_like(pool_ptr, -1))
            new_ptr = torch.where(newly_done, pool_ptr + B, pool_ptr)
            needs_reset = newly_done & (new_ptr < eval_obj._pool_size)
            
            # 4. Reset
            safe_idx = new_ptr.clamp(0, max(0, eval_obj._pool_size - 1))
            new_queries_raw = eval_obj._pool[safe_idx]
            
            reset_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            reset_queries[:, 0, :] = new_queries_raw
            
            reset_derived, reset_counts = engine.get_derived(reset_queries)
            
            # 5. Merge
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            
            final_current = torch.where(m1, reset_queries, new_current)
            final_derived = torch.where(m3, reset_derived, derived)
            final_counts = torch.where(needs_reset, reset_counts, counts)
            final_depths = torch.where(needs_reset, torch.zeros_like(new_depths), new_depths)
            final_done = torch.where(needs_reset, torch.zeros_like(done), done | newly_done)
            
            exhausted = new_ptr >= eval_obj._pool_size
            final_done = torch.where(exhausted & newly_done, torch.ones_like(final_done), final_done)
            
            new_mask = arange_S.unsqueeze(0) < final_counts.unsqueeze(1)
            
            # Write to output buffers
            out_current.copy_(final_current)
            out_derived.copy_(final_derived)
            out_counts.copy_(final_counts)
            out_mask.copy_(new_mask)
            out_depths.copy_(final_depths)
            out_done.copy_(final_done)
            out_pool_ptr.copy_(new_ptr)
            out_newly_done.copy_(newly_done)
            out_success.copy_(is_success)
            out_indices.copy_(finished_idx)
        
        self._compiled_step = torch.compile(
            step_and_reset,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[MinimalEvalV6] Compiled (mode={mode})")
    
    def _swap_buffers(self):
        """Swap output buffers to input buffers."""
        self._current.copy_(self._out_current)
        self._derived.copy_(self._out_derived)
        self._counts.copy_(self._out_counts)
        self._mask.copy_(self._out_mask)
        self._depths.copy_(self._out_depths)
        self._done.copy_(self._out_done)
        self._pool_ptr.copy_(self._out_pool_ptr)
    
    def setup_pool(self, queries: Tensor, sampler, n_corruptions: int, modes: Sequence[str]):
        pools = []
        for mode in modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=self.device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            pools.append(cands.view(-1, 3))
        
        self._pool = torch.cat(pools, dim=0)
        self._pool_size = self._pool.size(0)
        self._result_buf = torch.zeros(self._pool_size, dtype=torch.bool, device=self.device)
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
            
            # Run zero-arg step (reads input buffers, writes output buffers)
            self._compiled_step()
            
            # Record results
            newly_done = self._out_newly_done
            success = self._out_success
            indices = self._out_indices
            
            valid = (indices >= 0) & (indices < self._pool_size)
            if valid.any():
                safe_idx = torch.where(valid, indices, torch.zeros_like(indices))
                safe_val = torch.where(valid, success, torch.zeros_like(success))
                self._result_buf.scatter_(0, safe_idx, safe_val)
            
            # Swap buffers for next iteration
            self._swap_buffers()
            
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
