"""
Minimal Eval V10 - With multiple engine strategies.

Supports different strategies for handling engine calls:
- 'torch_cond': Uses torch.cond for conditional execution (~21s, breaks CUDA graphs)
- 'fused': Engine inside compiled step, called every step (~28s)
- 'separate': Engine outside compiled step with .any() check (use V11 for this)

Default: 'torch_cond' for best performance in V10.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence, Tuple, Literal

from engine_wrapper import EngineWrapper


class MinimalEvalV10:
    """Evaluation with configurable engine execution strategy."""
    
    def __init__(
        self,
        policy,
        engine: EngineWrapper,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
        strategy: Literal['torch_cond', 'fused'] = 'torch_cond',
    ):
        self.policy = policy
        self.engine = engine
        self.B = batch_size
        self.A = padding_atoms
        self.S = padding_states
        self.max_depth = max_depth
        self.device = device
        self.pad = engine.pad
        self.true_pred_idx = engine.true_pred_idx
        self.false_pred_idx = engine.false_pred_idx
        self.strategy = strategy
        
        # Constants
        self._arange_B = torch.arange(batch_size, device=device)
        self._arange_S = torch.arange(padding_states, device=device)
        self._mask_fill_val = -3.4e38
        
        # State buffers
        self._current = torch.full((batch_size, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._derived = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._mask = torch.zeros(batch_size, padding_states, dtype=torch.bool, device=device)
        self._depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._pool_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Pool
        self._max_pool_size = 50000
        self._pool = torch.zeros(self._max_pool_size, 3, dtype=torch.long, device=device)
        self._pool_size = torch.tensor(0, dtype=torch.long, device=device)
        self._result_buf = torch.zeros(self._max_pool_size, dtype=torch.bool, device=device)
        
        # Stride for pool access
        self._stride = torch.tensor(batch_size, dtype=torch.long, device=device)
        self._N = 0
        self._K = 0
        
        self._compiled_step = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile step function with selected strategy."""
        if self.strategy == 'torch_cond':
            self._compile_torch_cond(mode)
        elif self.strategy == 'fused':
            self._compile_fused(mode)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _compile_torch_cond(self, mode: str):
        """Compile with torch.cond for conditional engine execution."""
        policy = self.policy
        engine = self.engine
        B, S, A, pad = self.B, self.S, self.A, self.pad
        max_depth = self.max_depth
        arange_B = self._arange_B
        arange_S = self._arange_S
        mask_fill_val = self._mask_fill_val
        device = self.device
        max_pool = self._max_pool_size
        true_pred_idx = self.true_pred_idx
        false_pred_idx = self.false_pred_idx
        
        current = self._current
        derived = self._derived
        counts = self._counts
        mask = self._mask
        depths = self._depths
        done = self._done
        pool_ptr = self._pool_ptr
        stride_tensor = self._stride
        
        def step_fn(pool: Tensor, pool_size: Tensor):
            """Step with torch.cond - engine only called when needed."""
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
            is_true = (first_pred == true_pred_idx) if true_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_false_pred = (first_pred == false_pred_idx) if false_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_pad = (first_pred == pad)
            is_terminal = is_true | is_false_pred | is_pad
            is_success = is_true & active
            truncated = (new_depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            finished_idx = torch.where(newly_done, pool_ptr, torch.full_like(pool_ptr, -1))
            new_ptr = torch.where(newly_done, pool_ptr + stride_tensor, pool_ptr)
            needs_reset = newly_done & (new_ptr < pool_size)
            any_needs_reset = needs_reset.any()
            
            def reset_branch(new_current, derived, counts, new_depths, done, newly_done, needs_reset, new_ptr):
                safe_idx = new_ptr.clamp(0, max_pool - 1)
                new_queries_raw = pool[safe_idx]
                reset_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
                reset_queries[:, 0, :] = new_queries_raw
                reset_derived, reset_counts = engine.get_derived(reset_queries)
                m1 = needs_reset.view(B, 1, 1)
                m3 = needs_reset.view(B, 1, 1, 1)
                final_current = torch.where(m1, reset_queries, new_current)
                final_derived = torch.where(m3, reset_derived, derived)
                final_counts = torch.where(needs_reset, reset_counts, counts)
                final_depths = torch.where(needs_reset, torch.zeros_like(new_depths), new_depths)
                final_done = torch.where(needs_reset, torch.zeros_like(done), done | newly_done)
                return final_current, final_derived, final_counts, final_depths, final_done
            
            def no_reset_branch(new_current, derived, counts, new_depths, done, newly_done, needs_reset, new_ptr):
                final_done = done | newly_done
                return new_current.clone(), derived.clone(), counts.clone(), new_depths.clone(), final_done
            
            final_current, final_derived, final_counts, final_depths, final_done = torch.cond(
                any_needs_reset, reset_branch, no_reset_branch,
                (new_current, derived, counts, new_depths, done, newly_done, needs_reset, new_ptr),
            )
            
            exhausted = new_ptr >= pool_size
            final_done = torch.where(exhausted & newly_done, torch.ones_like(final_done), final_done)
            new_mask = arange_S.unsqueeze(0) < final_counts.unsqueeze(1)
            
            return (final_current, final_derived, final_counts, new_mask, 
                    final_depths, final_done, new_ptr, newly_done, is_success, finished_idx)
        
        self._compiled_step = torch.compile(step_fn, mode=mode, fullgraph=True, dynamic=False)
        print(f"[MinimalEvalV10] Compiled with torch.cond (mode={mode})")
    
    def _compile_fused(self, mode: str):
        """Compile with fused engine - called every step."""
        policy = self.policy
        engine = self.engine
        B, S, A, pad = self.B, self.S, self.A, self.pad
        max_depth = self.max_depth
        arange_B = self._arange_B
        arange_S = self._arange_S
        mask_fill_val = self._mask_fill_val
        device = self.device
        max_pool = self._max_pool_size
        true_pred_idx = self.true_pred_idx
        false_pred_idx = self.false_pred_idx
        
        current = self._current
        derived = self._derived
        counts = self._counts
        mask = self._mask
        depths = self._depths
        done = self._done
        pool_ptr = self._pool_ptr
        stride_tensor = self._stride
        
        def step_fn(pool: Tensor, pool_size: Tensor):
            """Step with fused engine - called every step for all slots."""
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
            is_true = (first_pred == true_pred_idx) if true_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_false_pred = (first_pred == false_pred_idx) if false_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_pad = (first_pred == pad)
            is_terminal = is_true | is_false_pred | is_pad
            is_success = is_true & active
            truncated = (new_depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            finished_idx = torch.where(newly_done, pool_ptr, torch.full_like(pool_ptr, -1))
            new_ptr = torch.where(newly_done, pool_ptr + stride_tensor, pool_ptr)
            needs_reset = newly_done & (new_ptr < pool_size)
            
            # 4. Engine call for ALL slots (fused approach)
            safe_idx = new_ptr.clamp(0, max_pool - 1)
            new_queries_raw = pool[safe_idx]
            reset_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            reset_queries[:, 0, :] = new_queries_raw
            reset_derived, reset_counts = engine.get_derived(reset_queries)
            
            # 5. Merge with torch.where
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            final_current = torch.where(m1, reset_queries, new_current)
            final_derived = torch.where(m3, reset_derived, derived)
            final_counts = torch.where(needs_reset, reset_counts, counts)
            final_depths = torch.where(needs_reset, torch.zeros_like(new_depths), new_depths)
            final_done = torch.where(needs_reset, torch.zeros_like(done), done | newly_done)
            
            exhausted = new_ptr >= pool_size
            final_done = torch.where(exhausted & newly_done, torch.ones_like(final_done), final_done)
            new_mask = arange_S.unsqueeze(0) < final_counts.unsqueeze(1)
            
            return (final_current, final_derived, final_counts, new_mask, 
                    final_depths, final_done, new_ptr, newly_done, is_success, finished_idx)
        
        self._compiled_step = torch.compile(step_fn, mode=mode, fullgraph=True, dynamic=False)
        print(f"[MinimalEvalV10] Compiled with fused engine (mode={mode})")
    
    def _init_from_pool(self, pool_size_int: int):
        """Initialize state from pool."""
        B, A, pad = self.B, self.A, self.pad
        N = self._N
        
        init_idx = torch.arange(B, device=self.device).clamp(max=max(0, N - 1))
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
        if N < B:
            self._done[N:] = True
        
        self._pool_ptr.copy_(torch.arange(B, device=self.device))
    
    def setup_pool(self, queries: Tensor, sampler, n_corruptions: int, modes: Sequence[str]):
        """Setup pool with transposed layout."""
        N = queries.shape[0]
        K = 1 + n_corruptions
        
        pools = []
        for mode in modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=self.device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            cands_t = cands.transpose(0, 1).contiguous()
            pools.append(cands_t.view(-1, 3))
        
        new_pool = torch.cat(pools, dim=0)
        new_size = new_pool.size(0)
        
        if new_size > self._max_pool_size:
            raise ValueError(f"Pool size {new_size} exceeds max {self._max_pool_size}")
        
        self._pool[:new_size].copy_(new_pool)
        self._pool[new_size:].fill_(self.pad)
        self._pool_size.fill_(new_size)
        self._result_buf.zero_()
        
        self._N = N
        self._K = K
        self._stride.fill_(N)
        
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
            print(f"Pool: {pool_size_int}, batch: {self.B}, strategy: {self.strategy}")
        
        max_steps = (pool_size_int // self.B + 2) * self.max_depth
        steps = 0
        
        while steps < max_steps:
            torch.compiler.cudagraph_mark_step_begin()
            
            (new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
             newly_done, success, indices) = self._compiled_step(self._pool, self._pool_size)
            
            torch._foreach_copy_(
                [self._current, self._derived, self._counts, self._mask, 
                 self._depths, self._done, self._pool_ptr],
                [new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr]
            )
            
            safe_idx = indices.clamp(min=0, max=pool_size_int - 1)
            valid_mask = (indices >= 0) & (indices < pool_size_int)
            safe_val = success & valid_mask
            self._result_buf.scatter_(0, safe_idx, safe_val)
            
            steps += 1
            
            if self._done.all():
                break
        
        if verbose:
            print(f"Steps: {steps}")
        
        results = {}
        offset = 0
        for mode in corruption_modes:
            ms_t = self._result_buf[offset:offset + N * K].view(K, N)
            ms = ms_t.t().contiguous()
            
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
