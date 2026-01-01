"""
Minimal Standalone Evaluation - Optimized Version.

Key optimizations:
1. Reset fused inside compiled step (no Python loop for resets)
2. Buffer-copy pattern for CUDA graph compatibility
3. No tensor mutations inside compiled function
"""
import torch
from torch import Tensor
from typing import Dict, Sequence, Tuple

from unification import UnificationEngineVectorized


class MinimalEval:
    """Optimized evaluation with fused step+reset in compiled function."""
    
    def __init__(
        self,
        policy,
        engine: UnificationEngineVectorized,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
        runtime_var_start_index: int,
    ):
        self.policy = policy
        self.engine = engine
        self.B = batch_size
        self.A = padding_atoms
        self.S = padding_states
        self.max_depth = max_depth
        self.device = device
        self.var_start = runtime_var_start_index
        self.pad = engine.padding_idx
        
        # Pre-allocated constants
        self._arange_B = torch.arange(batch_size, device=device)
        self._arange_S = torch.arange(padding_states, device=device)
        self._ones_B = torch.ones(batch_size, dtype=torch.long, device=device)
        self._false_state = torch.full((padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._mask_fill_val = -3.4e38
        self._var_buf = torch.full((batch_size,), runtime_var_start_index, dtype=torch.long, device=device)
        
        # State buffers (read by compiled step)
        self._current = torch.full((batch_size, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._derived = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._mask = torch.zeros(batch_size, padding_states, dtype=torch.bool, device=device)
        self._depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._pool_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Pool - fixed max size for stable addresses
        self._max_pool_size = 50000
        self._pool = torch.zeros(self._max_pool_size, 3, dtype=torch.long, device=device)
        self._pool_size = torch.tensor(0, dtype=torch.long, device=device)
        self._result_buf = torch.zeros(self._max_pool_size, dtype=torch.bool, device=device)
        
        self._compiled_step = None
    
    def _compute_derived_batch(self, queries: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute derived states for [B, A, 3] queries with static output."""
        B, S, A, pad = self.B, self.S, self.A, self.pad
        
        derived_raw, counts_raw, _ = self.engine.get_derived_states_compiled(
            queries, self._var_buf, None
        )
        
        # Pad to static shape
        derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        derived[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        # Validate
        within = self._arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
        valid = derived[:, :, :, 0] != pad
        ac = valid.sum(2)
        base_valid = within & (ac <= A) & (ac > 0)
        
        derived = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), derived,
                             self._false_state.unsqueeze(0).expand(B,-1,-1,-1))
        counts = base_valid.sum(1)
        counts = torch.where(counts == 0, self._ones_B, counts)
        
        return derived, counts
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile step function with fused reset."""
        policy = self.policy
        engine = self.engine
        B, S, A, pad = self.B, self.S, self.A, self.pad
        max_depth = self.max_depth
        arange_B = self._arange_B
        arange_S = self._arange_S
        ones_B = self._ones_B
        mask_fill_val = self._mask_fill_val
        device = self.device
        max_pool = self._max_pool_size
        false_state = self._false_state
        var_buf = self._var_buf
        
        # Capture state references
        current = self._current
        derived = self._derived
        counts = self._counts
        mask = self._mask
        depths = self._depths
        done = self._done
        pool_ptr = self._pool_ptr
        
        def step_fn(pool: Tensor, pool_size: Tensor):
            """
            Compiled step with fused reset.
            Returns NEW tensors - no mutations inside for CUDA graph compatibility.
            """
            # 1. Policy forward
            obs = {
                'sub_index': current.unsqueeze(1),
                'derived_sub_indices': derived,
                'action_mask': mask.to(torch.uint8),
            }
            logits = policy.get_logits(obs)
            masked_logits = logits.masked_fill(~mask, mask_fill_val)
            actions = masked_logits.argmax(dim=-1)
            
            # 2. Step
            active = ~done
            next_states = derived[arange_B, actions]
            new_current = torch.where(active.view(B, 1, 1), next_states, current)
            new_depths = torch.where(active, depths + 1, depths)
            
            # 3. Terminal check
            first_pred = new_current[:, 0, 0]
            is_terminal = first_pred == pad
            is_success = is_terminal & active
            truncated = (new_depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            # Record finished indices BEFORE incrementing pointer
            finished_idx = torch.where(newly_done, pool_ptr, torch.full_like(pool_ptr, -1))
            
            # 4. Compute new pointer (stride by B)
            new_ptr = torch.where(newly_done, pool_ptr + B, pool_ptr)
            
            # 5. Reset logic - fused inside
            needs_reset = newly_done & (new_ptr < pool_size)
            
            # Get new queries from pool
            safe_idx = new_ptr.clamp(0, max_pool - 1)
            new_queries_raw = pool[safe_idx]
            
            reset_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            reset_queries[:, 0, :] = new_queries_raw
            
            # Compute derived for reset slots
            reset_derived_raw, reset_counts_raw, _ = engine.get_derived_states_compiled(
                reset_queries, var_buf, None
            )
            
            # Static shape processing
            reset_derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=device)
            rK = min(reset_derived_raw.shape[1], S)
            rM = min(reset_derived_raw.shape[2], A)
            reset_derived[:, :rK, :rM, :] = reset_derived_raw[:, :rK, :rM, :]
            
            within_r = arange_S.unsqueeze(0) < reset_counts_raw.unsqueeze(1)
            valid_r = reset_derived[:, :, :, 0] != pad
            ac_r = valid_r.sum(2)
            base_valid_r = within_r & (ac_r <= A) & (ac_r > 0)
            
            reset_derived = torch.where(
                base_valid_r.unsqueeze(-1).unsqueeze(-1),
                reset_derived,
                false_state.unsqueeze(0).expand(B, -1, -1, -1)
            )
            reset_counts = base_valid_r.sum(1)
            reset_counts = torch.where(reset_counts == 0, ones_B, reset_counts)
            
            # 6. Merge reset with continue
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            
            final_current = torch.where(m1, reset_queries, new_current)
            final_derived = torch.where(m3, reset_derived, derived)
            final_counts = torch.where(needs_reset, reset_counts, counts)
            final_depths = torch.where(needs_reset, torch.zeros_like(new_depths), new_depths)
            final_done = torch.where(needs_reset, torch.zeros_like(done), done | newly_done)
            
            # Mark exhausted slots as done
            exhausted = new_ptr >= pool_size
            final_done = torch.where(exhausted & newly_done, torch.ones_like(final_done), final_done)
            
            new_mask = arange_S.unsqueeze(0) < final_counts.unsqueeze(1)
            
            return (final_current, final_derived, final_counts, new_mask, 
                    final_depths, final_done, new_ptr,
                    newly_done, is_success, finished_idx)
        
        self._compiled_step = torch.compile(step_fn, mode=mode, fullgraph=True, dynamic=False)
        print(f"[MinimalEval] Compiled (mode={mode})")
    
    def _init_from_pool(self, pool_size_int: int):
        """Initialize state from pool."""
        B, A, pad = self.B, self.A, self.pad
        
        init_idx = torch.arange(B, device=self.device).clamp(max=max(0, pool_size_int - 1))
        queries_raw = self._pool[init_idx]
        
        queries = torch.full((B, A, 3), pad, dtype=torch.long, device=self.device)
        queries[:, 0, :] = queries_raw
        
        derived, counts = self._compute_derived_batch(queries)
        
        self._current.copy_(queries)
        self._derived.copy_(derived)
        self._counts.copy_(counts)
        self._mask.copy_(self._arange_S.unsqueeze(0) < counts.unsqueeze(1))
        self._depths.zero_()
        self._done.zero_()
        self._pool_ptr.copy_(torch.arange(B, device=self.device))
    
    def setup_pool(self, queries: Tensor, sampler, n_corruptions: int, modes: Sequence[str]):
        """Generate candidate pool."""
        pools = []
        for mode in modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=self.device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)  # [N, K, 3]
            pools.append(cands.view(-1, 3))
        
        new_pool = torch.cat(pools, dim=0)
        new_size = new_pool.size(0)
        
        if new_size > self._max_pool_size:
            raise ValueError(f"Pool size {new_size} exceeds max {self._max_pool_size}")
        
        self._pool[:new_size].copy_(new_pool)
        self._pool[new_size:].fill_(self.pad)
        self._pool_size.fill_(new_size)
        self._result_buf.zero_()
        self._init_from_pool(new_size)
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run evaluation."""
        N, K = len(queries), 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() first")
        
        self.setup_pool(queries, sampler, n_corruptions, corruption_modes)
        pool_size_int = int(self._pool_size.item())
        
        if verbose:
            print(f"Pool: {pool_size_int} candidates, batch: {self.B}")
        
        max_total_steps = (pool_size_int // self.B + 2) * self.max_depth
        steps = 0
        
        while steps < max_total_steps:
            torch.compiler.cudagraph_mark_step_begin()
            
            # Compiled step returns NEW tensors
            (new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
             newly_done, success, indices) = self._compiled_step(self._pool, self._pool_size)
            
            # Copy OUTSIDE compiled region for CUDA graph compat
            torch._foreach_copy_(
                [self._current, self._derived, self._counts, self._mask, 
                 self._depths, self._done, self._pool_ptr],
                [new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr]
            )
            
            # Record results (vectorized, no loop)
            valid = (indices >= 0) & (indices < pool_size_int)
            if valid.any():
                safe_idx = torch.where(valid, indices, torch.zeros_like(indices))
                safe_val = torch.where(valid, success, torch.zeros_like(success))
                self._result_buf.scatter_(0, safe_idx, safe_val)
            
            steps += 1
            
            if self._done.all():
                break
        
        if verbose:
            print(f"Steps: {steps}")
        
        # Compute metrics
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
