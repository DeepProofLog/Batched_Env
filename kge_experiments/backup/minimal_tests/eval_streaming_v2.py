"""
Streaming Evaluation V2 - Uses existing EnvVec._step_and_reset_core.

Key insight: Use the EXISTING _step_and_reset_core which is already
vectorized and compiled. Just set up the pool correctly.
"""
import torch
from torch import Tensor
from typing import Dict, Sequence
from time import time


class EvalStreamingV2:
    """Streaming evaluation using existing EnvVec._step_and_reset_core."""
    
    def __init__(
        self,
        ppo,  # Existing PPO instance with compiled EnvVec
        device: torch.device,
    ):
        self.ppo = ppo
        self.env = ppo.env
        self.policy = ppo._uncompiled_policy
        self.device = device
        
        self._compiled_step = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile the fused eval step using existing env._step_and_reset_core."""
        policy = self.policy
        env = self.env
        mask_fill = -3.4e38
        
        def eval_step_and_reset(obs, state):
            """Fused policy + step + auto-reset."""
            # Get action
            logits = policy.get_logits(obs)
            masked = logits.masked_fill(obs['action_mask'] == 0, mask_fill)
            actions = masked.argmax(dim=-1)
            
            # Step + auto-reset
            new_obs, new_state = env._step_and_reset_core(
                state, actions, env._query_pool, state['per_env_ptrs']
            )
            
            return new_obs, new_state
        
        self._compiled_step = torch.compile(
            eval_step_and_reset,
            mode=mode,
            fullgraph=True,
            dynamic=False
        )
        print(f"[EvalStreamingV2] Compiled (mode={mode})")
    
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
        env = self.env
        B = env.batch_size
        N = len(queries)
        K = 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() first")
        
        # Generate candidate pool
        pools = []
        for mode in corruption_modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            pools.append(cands.view(-1, 3))
        
        pool = torch.cat(pools, dim=0)
        pool_size = pool.size(0)
        
        if verbose:
            print(f"Eval: {pool_size} candidates, batch={B}")
        
        # Set up env pool (for ordered iteration through candidates)
        env._query_pool = pool
        env.order = True  # Sequential traversal
        
        # Calculate steps needed
        # Each slot processes pool_size / B candidates on average
        # Each candidate takes max_depth steps
        candidates_per_slot = (pool_size + B - 1) // B
        max_steps = candidates_per_slot * env.max_depth + B
        
        # Success tracking per candidate
        success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)
        completed = torch.zeros(pool_size, dtype=torch.bool, device=device)
        
        # Initial reset with first B candidates
        initial_queries = pool[:B] if pool_size >= B else torch.cat([
            pool, torch.zeros(B - pool_size, 3, dtype=torch.long, device=device)
        ], 0)
        
        # Use env's reset
        obs = env.reset_from_pool(initial_queries)
        state = env._state_buffer
        
        # Track which candidate each slot is working on
        slot_candidate_idx = torch.arange(B, device=device)
        
        steps = 0
        while True:
            torch.compiler.cudagraph_mark_step_begin()
            obs, state = self._compiled_step(obs, state)
            
            # Record results for done slots
            done_slots = state['step_dones'].bool()
            if done_slots.any():
                done_indices = done_slots.nonzero(as_tuple=True)[0]
                for idx in done_indices:
                    i = idx.item()
                    cand_idx = slot_candidate_idx[i].item()
                    if cand_idx < pool_size and not completed[cand_idx]:
                        success_buffer[cand_idx] = state['step_rewards'][i] > 0.5
                        completed[cand_idx] = True
                    
                    # Advance to next candidate (strided)
                    slot_candidate_idx[i] += B
            
            steps += 1
            
            # Check if all candidates completed
            if completed.all() or steps > max_steps:
                break
        
        if verbose:
            print(f"Steps: {steps}, Completed: {completed.sum().item()}/{pool_size}")
        
        # Compute metrics
        results = {}
        offset = 0
        for mode in corruption_modes:
            ms = success_buffer[offset:offset + N * K].view(N, K)
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
