"""
Manual CUDA Graph Evaluation.

Uses torch.cuda.CUDAGraph directly for full control over memory.
No _foreach_copy_ overhead - graphs replay on exact same memory.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence, Tuple

from unification import UnificationEngineVectorized


class EvalCUDAGraph:
    """Evaluation with manual CUDA graph capture.
    
    Key: Use static buffers and manual graph capture/replay.
    No torch.compile - direct CUDA graph control.
    """
    
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
        
        # Static buffers - NEVER reallocated
        self._current = torch.zeros(B, A, 3, dtype=torch.long, device=device)
        self._derived = torch.zeros(B, S, A, 3, dtype=torch.long, device=device)
        self._counts = torch.zeros(B, dtype=torch.long, device=device)
        self._mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        self._done = torch.zeros(B, dtype=torch.bool, device=device)
        self._success = torch.zeros(B, dtype=torch.bool, device=device)
        self._actions = torch.zeros(B, dtype=torch.long, device=device)
        
        # Constants
        self._arange_B = torch.arange(B, device=device)
        self._arange_S = torch.arange(S, device=device)
        self._mask_fill = torch.tensor(-3.4e38, device=device)
        
        # CUDA graph for step
        self._graph = None
        self._graph_captured = False
    
    def _step_kernel(self):
        """Single step - operates on static buffers IN-PLACE."""
        B = self.batch_size
        pad = self.padding_idx
        
        # Get action from policy (in-place into self._actions)
        with torch.no_grad():
            logits = self.policy.get_logits({
                'sub_index': self._current.unsqueeze(1),
                'derived_sub_indices': self._derived,
                'action_mask': self._mask.to(torch.uint8),
            })
        
        # Mask invalid actions
        masked = torch.where(self._mask, logits, self._mask_fill)
        self._actions.copy_(masked.argmax(dim=-1))
        
        # Take step - update current IN-PLACE
        active = ~self._done
        next_states = self._derived[self._arange_B, self._actions]
        self._current.copy_(torch.where(active.view(B, 1, 1), next_states, self._current))
        
        # Check termination
        is_terminal = (self._current[:, 0, 0] == pad)
        is_success = is_terminal & active
        self._success.copy_(self._success | is_success)
        self._done.copy_(self._done | (active & is_terminal))
    
    def capture_graph(self):
        """Capture CUDA graph for step kernel."""
        # Warm up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_kernel()
        
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step_kernel()
        
        self._graph_captured = True
        print("[EvalCUDAGraph] Graph captured")
    
    def reset(self, queries: Tensor):
        """Reset buffers with new queries."""
        B, A = self.batch_size, self.padding_atoms
        pad = self.padding_idx
        Q = queries.shape[0]
        
        # Create padded queries
        padded = torch.full((B, A, 3), pad, dtype=torch.long, device=self.device)
        
        if queries.ndim == 2:
            # queries is [Q, 3], need to expand to [Q, A, 3]
            n = min(Q, B)
            padded[:n, 0, :] = queries[:n].to(self.device)
        else:
            # queries is already [Q, A, 3]
            n = min(Q, B)
            padded[:n] = queries[:n].to(self.device)
        
        # Compute derived
        var_idx = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=self.device)
        derived, counts = self._compute_derived(padded, var_idx)
        
        # Copy into static buffers
        self._current.copy_(padded)
        self._derived.copy_(derived)
        self._counts.copy_(counts)
        self._mask.copy_(self._arange_S.unsqueeze(0) < counts.unsqueeze(1))
        self._done.zero_()
        self._success.zero_()
    
    def _compute_derived(self, current, var_idx):
        """Compute derived states."""
        B, S, A, pad = self.batch_size, self.padding_states, self.padding_atoms, self.padding_idx
        
        derived_raw, counts_raw, _ = self.engine.get_derived_states_compiled(current, var_idx, None)
        
        buf = torch.full((B, S, A, 3), pad, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        within = self._arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
        valid = buf[:, :, :, 0] != pad
        ac = valid.sum(2)
        base_valid = within & (ac <= A) & (ac > 0)
        
        false_state = torch.full((S, A, 3), pad, dtype=torch.long, device=self.device)
        buf = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), buf, false_state.unsqueeze(0).expand(B,-1,-1,-1))
        counts = base_valid.sum(1)
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        
        return buf, counts
    
    def run_episode(self):
        """Run full episode using graph replay."""
        for _ in range(self.max_depth):
            if self._graph_captured:
                self._graph.replay()
            else:
                self._step_kernel()
        
        return self._success.clone()
    
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
        success = torch.zeros(pool_size, dtype=torch.bool, device=device)
        
        for r in range(n_rounds):
            start, end = r * B, min((r + 1) * B, pool_size)
            size = end - start
            
            q = pool[start:end]
            if size < B:
                q = torch.cat([q, torch.zeros(B - size, 3, dtype=torch.long, device=device)], 0)
            
            self.reset(q)
            result = self.run_episode()
            
            success[start:end] = result[:size] if size < B else result
        
        # Metrics
        results = {}
        offset = 0
        for mode in corruption_modes:
            ms = success[offset:offset + N * K].view(N, K)
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
