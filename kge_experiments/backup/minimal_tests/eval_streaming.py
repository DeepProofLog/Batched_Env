"""
Streaming Evaluation with step_and_maybe_reset.

Key: Single compiled step that auto-resets finished slots to next candidate.
Loop over this step in Python until all candidates evaluated.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence

from unification import UnificationEngineVectorized


class EvalStreaming:
    """Streaming evaluation - slots auto-reset when done."""
    
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
        pad = self.padding_idx
        
        # Constants
        self._arange_B = torch.arange(B, device=device)
        self._arange_S = torch.arange(S, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        self._zeros_B = torch.zeros(B, dtype=torch.long, device=device)
        self._false_state = torch.full((S, A, 3), pad, dtype=torch.long, device=device)
        self._mask_fill = torch.full((B, S), -3.4e38, device=device)
        self._var_start_buf = torch.full((B,), runtime_var_start_index, dtype=torch.long, device=device)
        
        # State buffers
        self._current = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        self._derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=device)
        self._counts = torch.zeros(B, dtype=torch.long, device=device)
        self._mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        self._depths = torch.zeros(B, dtype=torch.long, device=device)
        self._done = torch.zeros(B, dtype=torch.bool, device=device)
        self._success = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Pool buffers (set before evaluation)
        self._pool = None  # [P, 3] all candidates
        self._pool_ptr = torch.zeros(B, dtype=torch.long, device=device)  # current index per slot
        self._pool_size = 0
        self._success_buffer = None  # [P] results
        
        self._compiled_step = None
    
    def _reset_slot(self, slot_idx: int, query: Tensor):
        """Reset a single slot with new query."""
        pad = self.padding_idx
        A = self.padding_atoms
        
        # Pad query
        padded = torch.full((A, 3), pad, dtype=torch.long, device=self.device)
        padded[0, :] = query
        
        self._current[slot_idx] = padded
        self._depths[slot_idx] = 0
        self._done[slot_idx] = False
        self._success[slot_idx] = False
    
    def _reset_derived_for_slots(self, slot_mask: Tensor):
        """Compute derived states for specified slots."""
        if not slot_mask.any():
            return
        
        B, S, A, pad = self.batch_size, self.padding_states, self.padding_atoms, self.padding_idx
        
        # Get derived for all (simpler than masking)
        derived_raw, counts_raw, _ = self.engine.get_derived_states_compiled(
            self._current, self._var_start_buf, None
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
        
        # Only update masked slots
        self._derived = torch.where(slot_mask.view(B,1,1,1), derived, self._derived)
        self._counts = torch.where(slot_mask, counts, self._counts)
        self._mask = self._arange_S.unsqueeze(0) < self._counts.unsqueeze(1)
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile step_and_maybe_reset function."""
        policy = self.policy
        B = self.batch_size
        pad = self.padding_idx
        max_depth = self.max_depth
        arange_B = self._arange_B
        mask_fill = self._mask_fill
        device = self.device
        
        # Capture buffers via closure
        current = self._current
        derived = self._derived
        mask = self._mask
        depths = self._depths
        done = self._done
        success = self._success
        pool = self  # Reference to self for pool access
        
        def step_and_maybe_reset():
            """Take one step, auto-reset done slots."""
            # Get actions
            logits = policy.get_logits({
                'sub_index': current.unsqueeze(1),
                'derived_sub_indices': derived,
                'action_mask': mask.to(torch.uint8),
            })
            masked_logits = torch.where(mask, logits, mask_fill)
            actions = masked_logits.argmax(dim=-1)
            
            # Take step
            active = ~done
            next_states = derived[arange_B, actions]
            current.copy_(torch.where(active.view(B, 1, 1), next_states, current))
            depths.copy_(torch.where(active, depths + 1, depths))
            
            # Check termination
            is_terminal = (current[:, 0, 0] == pad)
            is_success = is_terminal & active
            truncated = (depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            done.copy_(done | newly_done)
            success.copy_(success | is_success)
            
            return newly_done
        
        self._compiled_step = torch.compile(
            step_and_maybe_reset,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[EvalStreaming] Compiled (mode={mode})")
    
    def setup_pool(self, queries: Tensor, sampler, n_corruptions: int, corruption_modes: Sequence[str]):
        """Generate candidate pool."""
        device = self.device
        
        pools = []
        for mode in corruption_modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            pools.append(cands.view(-1, 3))
        
        self._pool = torch.cat(pools, dim=0)
        self._pool_size = self._pool.size(0)
        self._success_buffer = torch.zeros(self._pool_size, dtype=torch.bool, device=device)
        
        # Initialize slots with first B candidates
        B = min(self.batch_size, self._pool_size)
        self._pool_ptr = torch.arange(B, dtype=torch.long, device=device)
        if B < self.batch_size:
            self._pool_ptr = torch.cat([
                self._pool_ptr,
                torch.full((self.batch_size - B,), self._pool_size, dtype=torch.long, device=device)
            ])
        
        # Reset all slots
        for i in range(min(B, self._pool_size)):
            self._reset_slot(i, self._pool[i])
        
        # Mark slots beyond pool as done
        self._done = self._pool_ptr >= self._pool_size
        
        # Compute initial derived
        self._reset_derived_for_slots(~self._done)
    
    def _handle_done_slots(self, newly_done: Tensor):
        """Record results and reset done slots to next candidates."""
        if not newly_done.any():
            return False
        
        done_indices = newly_done.nonzero(as_tuple=True)[0]
        
        for idx in done_indices:
            i = idx.item()
            ptr = self._pool_ptr[i].item()
            
            # Record result
            if ptr < self._pool_size:
                self._success_buffer[ptr] = self._success[i]
            
            # Advance to next candidate
            next_ptr = ptr + self.batch_size
            self._pool_ptr[i] = next_ptr
            
            if next_ptr < self._pool_size:
                # Reset slot with new candidate
                self._reset_slot(i, self._pool[next_ptr])
                self._done[i] = False
                self._success[i] = False
            else:
                # No more candidates for this slot
                self._done[i] = True
        
        # Recompute derived for reset slots
        reset_mask = newly_done & (self._pool_ptr < self._pool_size)
        if reset_mask.any():
            self._reset_derived_for_slots(reset_mask)
        
        return (self._pool_ptr >= self._pool_size).all().item()
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run streaming evaluation."""
        device = self.device
        N = len(queries)
        K = 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() first")
        
        # Setup pool
        self.setup_pool(queries, sampler, n_corruptions, corruption_modes)
        
        if verbose:
            print(f"Eval: {self._pool_size} candidates, batch={self.batch_size}")
        
        # Run until all done
        step_count = 0
        while True:
            torch.compiler.cudagraph_mark_step_begin()
            newly_done = self._compiled_step()
            
            all_done = self._handle_done_slots(newly_done)
            step_count += 1
            
            if all_done or self._done.all():
                break
        
        if verbose:
            print(f"Steps: {step_count}")
        
        # Compute metrics
        results = {}
        offset = 0
        for mode in corruption_modes:
            ms = self._success_buffer[offset:offset + N * K].view(N, K)
            scores = torch.where(ms, torch.zeros(N, K, device=device), 
                                torch.full((N, K), -100.0, device=device))
            
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
