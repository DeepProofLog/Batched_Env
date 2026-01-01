"""
Minimal PPO Evaluation Wrapper.

Stripped-down PPO for pure evaluation benchmarking - no training overhead.
"""
import torch
from torch import Tensor
from typing import Tuple, Dict, Sequence, Optional
from tensordict import TensorDict

from env_evaluate import EnvEval


class PPOEval:
    """Minimal PPO wrapper for evaluation only.
    
    Key simplifications vs PPO:
    - No rollout buffer
    - No collect_rollouts/train
    - No optimizer/scheduler
    - No callbacks
    - Direct evaluation loop with copy-outside pattern for CUDA graphs
    """
    
    def __init__(
        self,
        policy,
        env: EnvEval,
        device: torch.device,
        mask_fill_value: float = -3.4e38,
    ):
        self.policy = policy
        self.env = env
        self.device = device
        self.mask_fill_value = mask_fill_value
        self._compiled_step = None
        
        # Persistent buffers for copy-outside pattern (allocated on first use)
        self._obs_buffer = None
        self._state_buffer = None
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile evaluation step function."""
        policy = self.policy
        env = self.env
        mask_fill_value = self.mask_fill_value
        
        def eval_step(obs: TensorDict, state: TensorDict) -> Tuple[TensorDict, TensorDict]:
            """Pure functional eval step - returns NEW tensors."""
            logits = policy.get_logits(obs)
            masked = torch.where(obs['action_mask'].bool(), logits, mask_fill_value)
            actions = masked.argmax(dim=-1)
            return env.step_core(state, actions)
        
        self._compiled_step = torch.compile(
            eval_step, 
            mode=mode, 
            fullgraph=True, 
            dynamic=False
        )
        print(f"[PPOEval] Compiled eval step (mode={mode})")
    
    def _allocate_buffers(self, obs: TensorDict, state: TensorDict):
        """Allocate persistent buffers for copy-outside pattern."""
        # Observation buffer
        self._obs_buffer = TensorDict({
            'sub_index': torch.zeros_like(obs['sub_index']),
            'derived_sub_indices': torch.zeros_like(obs['derived_sub_indices']),
            'action_mask': torch.zeros_like(obs['action_mask']),
        }, batch_size=obs.batch_size, device=self.device)
        
        # State buffer
        self._state_buffer = TensorDict({
            'current_states': torch.zeros_like(state['current_states']),
            'derived_states': torch.zeros_like(state['derived_states']),
            'derived_counts': torch.zeros_like(state['derived_counts']),
            'original_queries': torch.zeros_like(state['original_queries']),
            'next_var_indices': torch.zeros_like(state['next_var_indices']),
            'depths': torch.zeros_like(state['depths']),
            'done': torch.zeros_like(state['done']),
            'success': torch.zeros_like(state['success']),
            'current_labels': torch.zeros_like(state['current_labels']),
            'history_hashes': torch.zeros_like(state['history_hashes']),
            'history_count': torch.zeros_like(state['history_count']),
            'step_rewards': torch.zeros_like(state['step_rewards']),
            'step_dones': torch.zeros_like(state['step_dones']),
        }, batch_size=state.batch_size, device=self.device)
    
    def _copy_obs(self, new_obs: TensorDict):
        """Copy obs into persistent buffer."""
        self._obs_buffer['sub_index'].copy_(new_obs['sub_index'])
        self._obs_buffer['derived_sub_indices'].copy_(new_obs['derived_sub_indices'])
        self._obs_buffer['action_mask'].copy_(new_obs['action_mask'])
    
    def _copy_state(self, new_state: TensorDict):
        """Copy state into persistent buffer."""
        self._state_buffer['current_states'].copy_(new_state['current_states'])
        self._state_buffer['derived_states'].copy_(new_state['derived_states'])
        self._state_buffer['derived_counts'].copy_(new_state['derived_counts'])
        self._state_buffer['original_queries'].copy_(new_state['original_queries'])
        self._state_buffer['next_var_indices'].copy_(new_state['next_var_indices'])
        self._state_buffer['depths'].copy_(new_state['depths'])
        self._state_buffer['done'].copy_(new_state['done'])
        self._state_buffer['success'].copy_(new_state['success'])
        self._state_buffer['current_labels'].copy_(new_state['current_labels'])
        self._state_buffer['history_hashes'].copy_(new_state['history_hashes'])
        self._state_buffer['history_count'].copy_(new_state['history_count'])
        self._state_buffer['step_rewards'].copy_(new_state['step_rewards'])
        self._state_buffer['step_dones'].copy_(new_state['step_dones'])
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Tensor,
        sampler,
        n_corruptions: int = 100,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run evaluation and compute MRR/Hits@K.
        
        Uses copy-outside pattern for CUDA graph compatibility:
        - Computation happens INSIDE compiled function (gets CUDA graph speedup)
        - Copying into persistent buffers happens OUTSIDE (avoids memory conflicts)
        """
        device = self.device
        batch_size = self.env.batch_size
        max_depth = self.env.max_depth
        total_queries = len(queries)
        total_candidates_per_query = 1 + n_corruptions
        
        if self._compiled_step is None:
            raise RuntimeError("Must call compile() before evaluate()")
        
        # 1. Generate ALL candidates for ALL modes
        all_pools = []
        for mode in corruption_modes:
            negative_candidates = sampler.corrupt(
                queries, num_negatives=n_corruptions, mode=mode, device=device
            )
            all_cands = torch.cat([
                queries.unsqueeze(1),  # [N, 1, 3]
                negative_candidates    # [N, K, 3]
            ], dim=1)
            all_pools.append(all_cands.view(-1, 3))  # [(N*(K+1)), 3]
        
        query_pool = torch.cat(all_pools, dim=0)
        pool_size = query_pool.size(0)
        
        if verbose:
            print(f"Eval: {pool_size} candidates, batch_size={batch_size}")
        
        # 2. Process pool in batched rounds
        n_rounds = (pool_size + batch_size - 1) // batch_size
        success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)
        
        for round_idx in range(n_rounds):
            start_idx = round_idx * batch_size
            end_idx = min(start_idx + batch_size, pool_size)
            round_size = end_idx - start_idx
            
            # Get queries for this round (with padding if needed)
            round_queries = query_pool[start_idx:end_idx]
            if round_size < batch_size:
                padding = batch_size - round_size
                pad_queries = torch.zeros(padding, 3, dtype=torch.long, device=device)
                round_queries = torch.cat([round_queries, pad_queries], dim=0)
            
            # Reset environment (OUTSIDE compiled region)
            obs, state = self.env.reset(round_queries)
            
            # Allocate persistent buffers on first use
            if self._obs_buffer is None:
                self._allocate_buffers(obs, state)
            
            # Track completion
            slot_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            slot_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Run max_depth steps with copy-outside pattern
            for step in range(max_depth):
                torch.compiler.cudagraph_mark_step_begin()
                
                # Pure computation INSIDE CUDA graph (returns NEW tensors)
                new_obs, new_state = self._compiled_step(obs, state)
                
                # Copy OUTSIDE compiled region (avoids mutation detection)
                self._copy_obs(new_obs)
                self._copy_state(new_state)
                
                # Use buffers for next iteration
                obs = self._obs_buffer
                state = self._state_buffer
                
                # Track done/success from buffer
                just_done = state['step_dones'].bool()
                step_success = (state['step_rewards'] > 0.5)
                
                # Record first finish
                newly_done = just_done & ~slot_finished
                slot_success = torch.where(newly_done, step_success, slot_success)
                slot_finished = slot_finished | just_done
            
            # Write results
            if round_size < batch_size:
                success_buffer[start_idx:end_idx] = slot_success[:round_size]
            else:
                success_buffer[start_idx:end_idx] = slot_success
        
        # 3. Compute metrics by mode
        results = {}
        offset = 0
        
        for mode in corruption_modes:
            mode_size = total_queries * total_candidates_per_query
            mode_success = success_buffer[offset:offset + mode_size].view(
                total_queries, total_candidates_per_query
            )
            
            # Rank computation: success -> 0, fail -> -100
            scores = torch.where(
                mode_success,
                torch.zeros_like(mode_success, dtype=torch.float32),
                torch.full_like(mode_success, -100.0, dtype=torch.float32)
            )
            
            pos_score = scores[:, 0:1]
            neg_scores = scores[:, 1:]
            
            # Rank with tie-breaking
            rnd = torch.rand(total_queries, total_candidates_per_query, device=device)
            better = neg_scores > pos_score
            tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1])
            ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)
            
            # Metrics
            mrr = (1.0 / ranks.float()).mean()
            hits1 = (ranks <= 1).float().mean()
            hits3 = (ranks <= 3).float().mean()
            hits10 = (ranks <= 10).float().mean()
            
            results[f'{mode}_mrr'] = mrr.item()
            results[f'{mode}_hits1'] = hits1.item()
            results[f'{mode}_hits3'] = hits3.item()
            results[f'{mode}_hits10'] = hits10.item()
            
            offset += mode_size
        
        # Aggregate
        n_modes = len(corruption_modes)
        results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
        results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
        results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
        results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes
        
        return results
