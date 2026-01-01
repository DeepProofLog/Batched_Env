"""
Fused Reset+Step Evaluation.

Key: Single compiled function that does reset + all steps.
This keeps reset inside the compiled/CUDA graph region.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence

from unification import UnificationEngineVectorized


class EvalFusedReset:
    """Evaluation with reset+steps fused in one compiled function."""
    
    def __init__(
        self,
        policy,
        vec_engine: UnificationEngineVectorized,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
        runtime_var_start_index: int,
    ):
        self.policy = policy
        self.engine = vec_engine
        self.batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.device = device
        self.runtime_var_start_index = runtime_var_start_index
        self.padding_idx = vec_engine.padding_idx
        
        B, S, A = batch_size, padding_states, padding_atoms
        
        # Pre-allocated constants
        self._arange_B = torch.arange(B, device=device)
        self._arange_S = torch.arange(S, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        self._false_state = torch.full((S, A, 3), self.padding_idx, dtype=torch.long, device=device)
        self._mask_fill = torch.full((B, S), -3.4e38, device=device)
        
        # Input buffer - queries go here BEFORE calling compiled function
        self._query_buffer = torch.zeros(B, 3, dtype=torch.long, device=device)
        
        self._compiled_episode = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile the fused reset+episode function."""
        policy = self.policy
        engine = self.engine
        B, S, A = self.batch_size, self.padding_states, self.padding_atoms
        max_depth = self.max_depth
        pad = self.padding_idx
        arange_B = self._arange_B
        arange_S = self._arange_S
        ones_B = self._ones_B
        false_state = self._false_state
        mask_fill = self._mask_fill
        var_start = self.runtime_var_start_index
        device = self.device
        query_buf = self._query_buffer
        
        def run_episode() -> Tensor:
            """Full episode: reset + all steps. Returns success tensor."""
            
            # ===== RESET PHASE =====
            # Expand queries [B, 3] -> [B, A, 3]
            current = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            current[:, 0, :] = query_buf
            
            # Compute initial derived states
            var_idx = torch.full((B,), var_start, dtype=torch.long, device=device)
            derived_raw, counts_raw, _ = engine.get_derived_states_compiled(current, var_idx, None)
            
            # Pad to static shape
            derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=device)
            K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
            derived[:, :K, :M, :] = derived_raw[:, :K, :M, :]
            
            # Validate
            within = arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
            valid = derived[:, :, :, 0] != pad
            ac = valid.sum(2)
            base_valid = within & (ac <= A) & (ac > 0)
            derived = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), derived, 
                                 false_state.unsqueeze(0).expand(B,-1,-1,-1))
            counts = base_valid.sum(1)
            counts = torch.where(counts == 0, ones_B, counts)
            
            # Build mask
            mask = arange_S.unsqueeze(0) < counts.unsqueeze(1)
            
            # Initialize tracking
            done = torch.zeros(B, dtype=torch.bool, device=device)
            success = torch.zeros(B, dtype=torch.bool, device=device)
            
            # ===== STEP PHASE =====
            for step in range(max_depth):
                active = ~done
                
                # Get action
                logits = policy.get_logits({
                    'sub_index': current.unsqueeze(1),
                    'derived_sub_indices': derived,
                    'action_mask': mask.to(torch.uint8),
                })
                masked_logits = torch.where(mask, logits, mask_fill)
                actions = masked_logits.argmax(dim=-1)
                
                # Take step
                next_states = derived[arange_B, actions]
                current = torch.where(active.view(B, 1, 1), next_states, current)
                
                # Check termination
                is_terminal = (current[:, 0, 0] == pad)
                is_success = is_terminal & active
                done = done | (active & is_terminal)
                success = success | is_success
                
                # Recompute derived for active (simplified - skip for now)
                # In full version, would call engine here for still-active states
            
            return success
        
        self._compiled_episode = torch.compile(
            run_episode,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[EvalFusedReset] Compiled (mode={mode})")
    
    def set_queries(self, queries: Tensor):
        """Copy queries into buffer (called BEFORE compiled episode)."""
        n = min(queries.shape[0], self.batch_size)
        self._query_buffer.zero_()
        self._query_buffer[:n] = queries[:n].to(self.device)
    
    def run(self) -> Tensor:
        """Run compiled episode."""
        return self._compiled_episode()
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Full evaluation."""
        device = self.device
        B = self.batch_size
        N = len(queries)
        K = 1 + n_corruptions
        
        if self._compiled_episode is None:
            raise RuntimeError("Must call compile() first")
        
        # Generate candidates
        pools = []
        for mode in corruption_modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            pools.append(cands.view(-1, 3))
        
        pool = torch.cat(pools, dim=0)
        pool_size = pool.size(0)
        
        if verbose:
            print(f"Eval: {pool_size} candidates, batch={B}")
        
        n_rounds = (pool_size + B - 1) // B
        success_buf = torch.zeros(pool_size, dtype=torch.bool, device=device)
        
        for r in range(n_rounds):
            start, end = r * B, min((r + 1) * B, pool_size)
            size = end - start
            
            q = pool[start:end]
            if size < B:
                q = torch.cat([q, torch.zeros(B - size, 3, dtype=torch.long, device=device)], 0)
            
            # Set queries + run compiled episode
            self.set_queries(q)
            torch.compiler.cudagraph_mark_step_begin()
            success = self.run()
            
            success_buf[start:end] = success[:size] if size < B else success
        
        # Metrics
        results = {}
        offset = 0
        for mode in corruption_modes:
            ms = success_buf[offset:offset + N * K].view(N, K)
            scores = torch.where(ms, torch.zeros(N, K, device=device), torch.full((N, K), -100.0, device=device))
            
            pos, neg = scores[:, 0:1], scores[:, 1:]
            rnd = torch.rand(N, K, device=device)
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
