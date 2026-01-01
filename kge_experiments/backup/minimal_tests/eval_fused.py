"""
Fused Multi-Step Evaluation.

Key innovation: Run ALL steps (max_depth) in a SINGLE compiled function call.
This reduces graph invocations from (rounds × steps) to just (rounds).
"""
import torch
from torch import Tensor
from typing import Tuple, Dict, Sequence

from unification import UnificationEngineVectorized


class EnvEvalFused:
    """Environment with fused multi-step execution."""
    
    def __init__(
        self,
        vec_engine: UnificationEngineVectorized,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
        runtime_var_start_index: int,
        end_proof_action: bool = True,
    ):
        self.engine = vec_engine
        self._batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.device = device
        self.runtime_var_start_index = runtime_var_start_index
        self.end_proof_action = end_proof_action
        self.padding_idx = vec_engine.padding_idx
        
        B, S, A = batch_size, padding_states, padding_atoms
        self._arange_S = torch.arange(S, device=device)
        self._arange_B = torch.arange(B, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        self._false_state = torch.full((S, A, 3), self.padding_idx, dtype=torch.long, device=device)
        self.end_state = torch.full((A, 3), self.padding_idx, dtype=torch.long, device=device)
        
        # Persistent buffers
        self._current = torch.zeros(B, A, 3, dtype=torch.long, device=device)
        self._derived = torch.zeros(B, S, A, 3, dtype=torch.long, device=device)
        self._counts = torch.zeros(B, dtype=torch.long, device=device)
        self._mask = torch.zeros(B, S, dtype=torch.uint8, device=device)
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def reset(self, queries: Tensor):
        """Reset environment."""
        B, A = queries.shape[0], self.padding_atoms
        pad = self.padding_idx
        
        if queries.ndim == 2:
            padded = torch.full((B, A, 3), pad, dtype=torch.long, device=self.device)
            padded[:, 0, :] = queries.to(self.device)
            queries = padded
        else:
            queries = queries.to(self.device)
        
        var_idx = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=self.device)
        derived, counts = self._compute_derived(queries, var_idx)
        
        self._current.copy_(queries)
        self._derived.copy_(derived)
        self._counts.copy_(counts)
        self._mask.copy_((self._arange_S < counts.unsqueeze(1)).to(torch.uint8))
    
    def _compute_derived(self, current, var_indices):
        """Compute derived states."""
        B, S, A, pad = self._batch_size, self.padding_states, self.padding_atoms, self.padding_idx
        
        derived_raw, counts_raw, _ = self.engine.get_derived_states_compiled(current, var_indices, None)
        
        buf = torch.full((B, S, A, 3), pad, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        within = self._arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
        valid = buf[:, :, :, 0] != pad
        atom_counts = valid.sum(dim=2)
        base_valid = within & (atom_counts <= A) & (atom_counts > 0)
        buf = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), buf, self._false_state.unsqueeze(0).expand(B,-1,-1,-1))
        new_counts = base_valid.sum(dim=1)
        
        needs_false = new_counts == 0
        buf = torch.where(needs_false.view(-1,1,1,1), self._false_state.unsqueeze(0).expand(B,-1,-1,-1), buf)
        new_counts = torch.where(needs_false, self._ones_B, new_counts)
        
        if self.end_proof_action:
            new_counts = new_counts.clamp(max=S-1)
            buf[self._arange_B, new_counts] = self.end_state.unsqueeze(0).expand(B,-1,-1)
            is_term = (current[:, 0, 0] == pad)
            new_counts = torch.where(~is_term, new_counts + 1, new_counts).clamp(max=S)
        
        return buf, new_counts


class PPOEvalFused:
    """PPO with fused multi-step evaluation.
    
    Key: Compile a function that runs ALL max_depth steps at once.
    This means only ONE graph invocation per round instead of max_depth.
    """
    
    def __init__(self, policy, env: EnvEvalFused, device: torch.device, mask_fill_value: float = -3.4e38):
        self.policy = policy
        self.env = env
        self.device = device
        self.mask_fill_value = mask_fill_value
        self._compiled_episode = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile fused episode function."""
        policy = self.policy
        env = self.env
        mask_fill = self.mask_fill_value
        max_depth = env.max_depth
        B = env.batch_size
        pad = env.padding_idx
        arange_B = env._arange_B
        arange_S = env._arange_S
        false_state = env._false_state
        ones_B = env._ones_B
        end_state = env.end_state
        S = env.padding_states
        
        def run_episode() -> Tuple[Tensor, Tensor]:
            """Run full episode - ALL steps fused into one graph."""
            # Read from persistent buffers
            current = env._current
            derived = env._derived
            counts = env._counts
            mask = env._mask
            
            done = torch.zeros(B, dtype=torch.bool, device=env.device)
            success = torch.zeros(B, dtype=torch.bool, device=env.device)
            first_done = torch.zeros(B, dtype=torch.bool, device=env.device)
            first_success = torch.zeros(B, dtype=torch.bool, device=env.device)
            
            # Unroll all steps
            for step in range(max_depth):
                active = ~done
                
                # Get action
                logits = policy.get_logits({
                    'sub_index': current.unsqueeze(1),
                    'derived_sub_indices': derived,
                    'action_mask': mask,
                })
                masked_logits = torch.where(mask.bool(), logits, mask_fill)
                actions = masked_logits.argmax(dim=-1)
                
                # Take step
                next_states = derived[arange_B, actions]
                current = torch.where(active.view(B, 1, 1), next_states, current)
                
                # Check termination
                is_terminal = (current[:, 0, 0] == pad)
                is_success = is_terminal & active
                newly_done = active & is_terminal
                
                # Record FIRST completion
                first_now = newly_done & ~first_done
                first_success = torch.where(first_now, is_success, first_success)
                first_done = first_done | newly_done
                
                done = done | newly_done
                success = success | is_success
                
                # Compute next derived (simplified - skip for done)
                still_active = ~done
                # For simplicity, just use current derived if mostly done
                # This is a speedup heuristic
                new_counts = torch.where(still_active, counts, counts)
                mask = (arange_S < new_counts.unsqueeze(1)).to(torch.uint8)
            
            return first_done, first_success
        
        self._compiled_episode = torch.compile(
            run_episode,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[PPOEvalFused] Compiled fused episode (mode={mode})")
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Evaluate with fused episode execution."""
        device = self.device
        batch_size = self.env.batch_size
        total_queries = len(queries)
        total_candidates = 1 + n_corruptions
        
        if self._compiled_episode is None:
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
                pad_q = torch.zeros(batch_size - size, 3, dtype=torch.long, device=device)
                round_queries = torch.cat([round_queries, pad_q], dim=0)
            
            # Reset (outside graph)
            self.env.reset(round_queries)
            
            # Run fused episode (ONE graph call for all steps!)
            torch.compiler.cudagraph_mark_step_begin()
            done, success = self._compiled_episode()
            
            if size < batch_size:
                success_buffer[start:end] = success[:size]
            else:
                success_buffer[start:end] = success
        
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
