import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pickle
from typing import Tuple, Optional, Dict, Any
import math

from dataset import DataHandler
from index_manager import IndexManager


class Attention_State(nn.Module):
    """
    Computes state embeddings using multi-head self-attention over atom embeddings.
    Assumes atoms within a state form a set/sequence to attend over.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 dropout_rate: float = 0.0, 
                 regularization: float = 0.0, 
                 device="cpu"):
        """
        Args:
            embed_dim: The embedding dimension of atoms and the final state.
            num_heads: Number of attention heads. Must divide embed_dim.
            dropout_rate: Dropout probability for attention and final output.
            regularization: Coefficient for L2 regularization loss on the output.
            device: Device for computation.
        """
        super(Attention_State, self).__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.device = device

        # Multi-Head Self-Attention Layer
        # batch_first=True expects input shape [Batch, Seq, Feat]
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, 
                                               num_heads=num_heads, 
                                               dropout=dropout_rate, 
                                               batch_first=True) 
                                               
        # Optional: Layer Normalization for stability
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        # Optional: A small FeedForward network after attention pooling
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), # Expand
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Apply dropout within FFN
            nn.Linear(embed_dim * 2, embed_dim)  # Contract
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        if dropout_rate > 0:
            self.output_dropout = nn.Dropout(p=dropout_rate)
            
        self.to(device)

    def add_loss(self, loss_value):
        # Placeholder for your custom regularization loss mechanism
        # In a standard setup, this might be handled by the optimizer directly
        # For now, we'll just store it if needed, or you can integrate later.
        if not hasattr(self, '_custom_losses'):
            self._custom_losses = []
        self._custom_losses.append(loss_value)
        # print(f"Debug: Added loss {loss_value.item()}") # Optional debug


    def forward(self, atom_embeddings: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            atom_embeddings: Tensor of shape [B, n_states, n_atoms, embed_dim].
                             Represents the embeddings of atoms within each state.
            padding_mask: Optional tensor of shape [B, n_states, n_atoms] where True indicates
                          a padded atom that should be ignored by attention. 
                          (Requires adjustment below if used).

        Returns:
            output: Tensor of shape [B, n_states, embed_dim] representing state embeddings.
        """
        B, n_states, n_atoms, embed_dim = atom_embeddings.shape
        
        # Reshape for MultiheadAttention: Combine B and n_states into the batch dim
        # Input shape: [B * n_states, n_atoms, embed_dim]
        flat_atoms = atom_embeddings.reshape(B * n_states, n_atoms, embed_dim)

        # --- Handle Padding Mask (if provided) ---
        # MultiheadAttention expects key_padding_mask of shape [Batch, Seq_len]
        # where True indicates positions to be *ignored*.
        key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.shape != (B, n_states, n_atoms):
                 raise ValueError("padding_mask shape must be [B, n_states, n_atoms]")
            key_padding_mask = padding_mask.reshape(B * n_states, n_atoms)
            # Ensure mask is boolean; True means ignore.
            # If your mask means something else (e.g., 0 for padding), invert it:
            # key_padding_mask = ~key_padding_mask 

        # Apply LayerNorm before attention
        normed_atoms = self.layer_norm1(flat_atoms)

        # Apply self-attention: query, key, value are all the atom embeddings
        # attn_output shape: [B * n_states, n_atoms, embed_dim]
        # attn_weights shape: [B * n_states, n_atoms, n_atoms] (average over heads)
        attn_output, _ = self.attention(query=normed_atoms, 
                                        key=normed_atoms, 
                                        value=normed_atoms,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False) # Don't need weights unless debugging

        # Add residual connection (skip connection)
        res_atoms = flat_atoms + attn_output # Or normed_atoms + attn_output

        # --- Pooling Strategy ---
        # Average pooling across the atoms dimension (n_atoms)
        # Apply mask *before* pooling if available to ignore padding tokens in mean calculation
        if key_padding_mask is not None:
             # Expand mask for broadcasting: [B*n_states, n_atoms, 1]
             mask_expanded = (~key_padding_mask).unsqueeze(-1).float() 
             # Element-wise multiply features by mask (zeros out padded positions)
             masked_res_atoms = res_atoms * mask_expanded 
             # Summing masked features and dividing by the count of non-padded items
             pooled_state = masked_res_atoms.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
             # Simple mean pooling if no mask provided
             pooled_state = res_atoms.mean(dim=1) # Shape: [B * n_states, embed_dim]
        
        # Apply LayerNorm and FeedForward network
        normed_pooled = self.layer_norm2(pooled_state)
        ffn_output = self.ffn(normed_pooled)
        
        # Add second residual connection
        final_pooled_state = pooled_state + ffn_output # Or normed_pooled + ffn_output

        # Reshape back to original batch/state structure
        # Shape: [B, n_states, embed_dim]
        state_emb = final_pooled_state.reshape(B, n_states, embed_dim)

        # Apply final dropout
        if self.dropout_rate > 0:
            state_emb = self.output_dropout(state_emb)

        # Apply regularization loss (using the custom method)
        if self.regularization > 0:
            # Calculate L2 norm across the embedding dimension for each state
            reg_loss = self.regularization * torch.linalg.vector_norm(state_emb, ord=2, dim=-1).mean() 
            self.add_loss(reg_loss)
            
        return state_emb

def read_embeddings(file_c:str, file_p:str, constant_str2idx:dict, predicate_str2idx:dict)-> Tuple[dict, dict]:
    '''Read embeddings from a file'''
    with open(file_c, 'rb') as f:
        constant_embeddings = pickle.load(f)
    with open(file_p, 'rb') as f:
        predicate_embeddings = pickle.load(f)
    # in cte embeddings the key is the domain (we ignore it) and the value is a dict, whose key is the constant and the value is the embedding
    constant_embeddings = {
        constant: emb
        for domain, domain_dict in constant_embeddings.items()
        for constant, emb in domain_dict.items()
    }
    # in pred embeddings as key take the first str until ( and the value is the embedding
    predicate_embeddings = {
        pred.split('(')[0]: emb
        for pred, emb in predicate_embeddings.items()
    }
    # using the str2idx dicts, create the idx2emb dicts
    constant_idx2emb = {constant_str2idx[constant]: emb for constant, emb in constant_embeddings.items()}
    predicate_idx2emb = {predicate_str2idx[predicate]: emb for predicate, emb in predicate_embeddings.items()}

    # order the embeddings by index
    constant_idx2emb = {idx: constant_idx2emb[idx] for idx in sorted(constant_idx2emb)}
    predicate_idx2emb = {idx: predicate_idx2emb[idx] for idx in sorted(predicate_idx2emb)}
    return constant_idx2emb, predicate_idx2emb


def create_embed_tables(constant_idx2emb:dict, predicate_idx2emb:dict, var_no:int)-> Tuple[dict, dict]:
    '''Create embedding tables for constants + variables and predicates'''
    # embeddings ndarray to tensor
    constant_idx2emb = torch.tensor(np.stack([constant_idx2emb[key] for key in constant_idx2emb.keys()]), dtype=torch.float32)
    predicate_idx2emb = torch.tensor(np.stack([predicate_idx2emb[key] for key in predicate_idx2emb.keys()]), dtype=torch.float32)
    # TODO: better ways to do variable and T/F embeddings?
    # random embeddings for True, False and variables
    embed_dim = constant_idx2emb.size(1)
    for i in range(2):
        predicate_idx2emb = torch.cat([predicate_idx2emb, torch.rand(1, embed_dim)], dim=0)
    for i in range(var_no):
        constant_idx2emb= torch.cat([constant_idx2emb, torch.rand(1, embed_dim)], dim=0)
    return constant_idx2emb, predicate_idx2emb



def transE_embedding(predicate_embeddings: torch.Tensor, constant_embeddings: torch.Tensor) -> torch.Tensor:
    """
    TransE function to compute atom embeddings.
    
    Arguments:
    - predicate_embeddings: Tensor of shape (batch_size,..., embedding_dim)
    - constant_embeddings: Tensor of shape (batch_size,..., 2, embedding_dim)
    
    Returns:
    - atom_embeddings: Tensor of shape (batch_size,..., embedding_dim)
    """
    # Separate the constants
    assert constant_embeddings.size(-2) == 2, "The second dimension of constant_embeddings should be 2 (arity)"
    assert predicate_embeddings.size(-2) == 1, "The second dimension of predicate_embeddings should be 1"
    predicate_embeddings = predicate_embeddings.squeeze(-2)
    constant_1 = constant_embeddings[..., 0, :]  # Shape: (256, 64)
    constant_2 = constant_embeddings[..., 1, :]  # Shape: (256, 64)
    # Compute the atom embedding using TransE formula
    atom_embeddings = predicate_embeddings + (constant_1 - constant_2)
    return atom_embeddings


        
class EmbedderNonLearnable:
    """
    Embedder using pre-trained, non-learnable embeddings.
    Handles batch processing where the input tensor's leading dimensions
    represent the batch structure (e.g., [bs] or [bs, n_states]).
    """
    def __init__(self, constant_idx2emb: torch.Tensor, predicate_idx2emb: torch.Tensor, device="cpu"):
        """
        Initialize the embedding function.
        Args:
            constant_idx2emb: Tensor of pre-trained constant/variable embeddings.
            predicate_idx2emb: Tensor of pre-trained predicate embeddings.
            device: Torch device.
        """
        self.embed_dim = constant_idx2emb.size(1)
        self.constant_idx2emb = constant_idx2emb.to(device)
        self.predicate_idx2emb = predicate_idx2emb.to(device)
        # Ensure padding embedding (index 0) is zeros
        if self.constant_idx2emb.shape[0] > 0:
             self.constant_idx2emb[0] = 0
        if self.predicate_idx2emb.shape[0] > 0:
             self.predicate_idx2emb[0] = 0
        self.device = device # Store device

    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """
        Get state embeddings for a batch of sub-indices using vectorized lookup and TransE.

        Args:
            sub_indices: Tensor containing sub-indices (P, S, O) for atoms.
                         Expected shape: (bs, ..., n_atoms, 3)
                         Examples:
                           - Current state: (bs, n_atoms, 3)
                           - Derived states (actions): (bs, n_states, n_atoms, 3)

        Returns:
            Tensor of state embeddings.
            Expected shape: (bs, ..., embed_dim)
            Examples:
              - Current state output: (bs, embed_dim)
              - Derived states output: (bs, n_states, embed_dim)
        """
        # Ensure indices are long type
        sub_indices = sub_indices.long()

        # Look up embeddings
        predicate_indices = sub_indices[..., 0].unsqueeze(-1) # Add feature dim: (bs, ..., n_atoms, 1)
        constant_indices = sub_indices[..., 1:]             # (bs, ..., n_atoms, 2)

        predicate_embeddings = F.embedding(predicate_indices, self.predicate_idx2emb, padding_idx=0) # (bs, ..., n_atoms, 1, embed_dim)
        constant_embeddings = F.embedding(constant_indices, self.constant_idx2emb, padding_idx=0)   # (bs, ..., n_atoms, 2, embed_dim)

        # Calculate atom embeddings using TransE (or other chosen method)
        # transE_embedding expects pred shape (..., embed_dim) and const shape (..., 2, embed_dim)
        atom_embeddings = transE_embedding(predicate_embeddings, constant_embeddings) # (bs, ..., n_atoms, embed_dim)

        # Aggregate atom embeddings to get state embeddings (sum over n_atoms dimension)
        # The dimension to sum over is the second to last (-2)
        state_embeddings = atom_embeddings.sum(dim=-2) # (bs, ..., embed_dim)

        return state_embeddings

    def to(self, device):
        """Move embedding tables to specified device."""
        self.device = device
        self.constant_idx2emb = self.constant_idx2emb.to(device)
        self.predicate_idx2emb = self.predicate_idx2emb.to(device)
        return self






class RNN(nn.Module):
    """RNN-based layer for computing atom embeddings with specific input shapes.

    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]
    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu"
    ):
        super(RNN, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regularization = regularization
        self.device = device
        
        # GRU expects inputs of shape (seq_len, batch, input_size)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=1)
        self.to(device)
    
    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
                           where the fourth dimension is the arity.
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        # Apply dropout
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)
        
        # Remove the arity dimension from predicate_emb (which is 1)
        # New shape: [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(-2)
        
        # Prepare initial hidden state h0 for the GRU.
        # GRU expects h0 shape: (num_layers, batch, embed_dim)
        # Here, we combine batch, n_states, and n_atoms into one dimension.
        B, n_states, n_atoms, embed_dim = predicate_emb.shape
        h0 = predicate_emb.reshape(B * n_states * n_atoms, embed_dim)  # [B*n_states*n_atoms, embed_dim]
        h0 = h0.unsqueeze(0)  # [1, B*n_states*n_atoms, embed_dim]
        
        # Prepare the constant embeddings as a sequence for the GRU.
        # constant_embs has shape: [B, n_states, n_atoms, 2, embed_dim]
        # We want to create a tensor of shape: [seq_len, B*n_states*n_atoms, embed_dim] with seq_len=2.
        seq = constant_embs.permute(3, 0, 1, 2, 4)  # now shape: [2, B, n_states, n_atoms, embed_dim]
        seq = seq.reshape(2, B * n_states * n_atoms, embed_dim)  # [2, B*n_states*n_atoms, embed_dim]
        
        # (Optional debugging prints)
        # print('constant_embs:', constant_embs.shape)
        # print('predicate_emb:', predicate_emb.shape)
        # print('seq:', seq.shape)
        # print('h0:', h0.shape)
        
        # Run GRU: we only need the final hidden state.
        _, hidden = self.gru(seq, h0)  # hidden shape: [1, B*n_states*n_atoms, embed_dim]
        output = hidden.squeeze(0)  # shape: [B*n_states*n_atoms, embed_dim]
        
        # Reshape output back to [B, n_states, n_atoms, embed_dim]
        output = output.reshape(B, n_states, n_atoms, embed_dim)
        
        # (Optional) Add regularization loss.
        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))
        
        return output

    def add_loss(self, loss: torch.Tensor):
        # This function should be connected to your overall loss management.
        # For example, you could store the loss in an attribute or add it to a list.
        pass

class RNN_state(nn.Module):
    def __init__(self, embed_dim: int, dropout_rate: float = 0.0, regularization: float = 0.0, device="cpu"):
        """
        Args:
            embed_dim: The embedding dimension.
            dropout_rate: Dropout probability.
            regularization: Coefficient for L2 regularization loss.
            device: Device for computation.
        """
        super(RNN_state, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True)
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_embeddings: Tensor of shape [B, n_states, n_atoms, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, embed_dim]
        """

        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)

        B, n_states, n_atoms, embed_dim = atom_embeddings.shape
        # Flatten batch and state dimensions: shape becomes [B*n_states, n_atoms, embed_dim]
        flat_atoms = atom_embeddings.reshape(B * n_states, n_atoms, embed_dim)
        
        # Process the atom sequence with GRU.
        # h_n has shape [1, B*n_states, embed_dim]; we use its squeezed version as the state embedding.
        _, h_n = self.gru(flat_atoms)
        state_emb = h_n.squeeze(0)  # shape: [B*n_states, embed_dim]
        
        # Reshape back to [B, n_states, embed_dim]
        state_emb = state_emb.reshape(B, n_states, embed_dim)
        
        if self.regularization > 0:
            self.add_loss(self.regularization * state_emb.norm(p=2))
            
        return state_emb

    def add_loss(self, loss: torch.Tensor):
        # This function should be integrated with your overall loss management.
        pass


class Transformer_state(nn.Module):
    def __init__(self, embed_dim: int, dropout_rate: float = 0.0, regularization: float = 0.0, device="cpu", num_heads: int = 1):
        """
        Args:
            embed_dim: The embedding dimension.
            dropout_rate: Dropout probability.
            regularization: Coefficient for L2 regularization loss.
            device: Device for computation.
            num_heads: Number of attention heads.
        """
        super(Transformer_state, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        
        # A learned query vector for aggregating the atom embeddings.
        # It will be broadcast to each state instance.
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        # MultiheadAttention expects inputs as (seq_len, batch, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_embeddings: Tensor of shape [B, n_states, n_atoms, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, embed_dim]
        """
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        
        B, n_states, n_atoms, embed_dim = atom_embeddings.shape
        # Flatten batch and state dimensions to process each state's atoms independently.
        # New shape: [B*n_states, n_atoms, embed_dim]
        flat_atoms = atom_embeddings.reshape(B * n_states, n_atoms, embed_dim)
        
        # Rearrange to (seq_len, batch, embed_dim) for multi-head attention.
        flat_atoms = flat_atoms.transpose(0, 1)  # shape: [n_atoms, B*n_states, embed_dim]
        
        # Prepare the query for each state instance.
        # Expand the learned query to shape [1, B*n_states, embed_dim]
        query = self.query.expand(1, B * n_states, embed_dim)
        
        # Run multi-head attention: query attends to the sequence of atom embeddings.
        attn_output, _ = self.attention(query, flat_atoms, flat_atoms)
        # attn_output is of shape [1, B*n_states, embed_dim]; remove the sequence dimension.
        state_emb = attn_output.squeeze(0)  # shape: [B*n_states, embed_dim]
        
        # Reshape back to [B, n_states, embed_dim]
        state_emb = state_emb.reshape(B, n_states, embed_dim)
        
        if self.regularization > 0:
            self.add_loss(self.regularization * state_emb.norm(p=2))
        
        return state_emb

    def add_loss(self, loss: torch.Tensor):
        # Integrate with your overall loss handling here.
        pass

class Transformer(nn.Module):
    """Transformer-based layer for computing atom embeddings.
    
    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]
      
    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
        num_heads: int = 1
    ):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.regularization = regularization
        self.device = device
        
        # nn.MultiheadAttention expects input of shape (seq_len, batch, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.to(device)
    
    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        # Apply dropout
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)
        
        # Remove the singleton arity dimension from predicate_emb
        # New shape: [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(3)
        B, n_states, n_atoms, embed_dim = predicate_emb.shape
        flat_batch = B * n_states * n_atoms
        
        # Prepare the query: reshape to [flat_batch, embed_dim] then add a sequence dim → [1, flat_batch, embed_dim]
        query = predicate_emb.reshape(flat_batch, embed_dim).unsqueeze(0)
        
        # Prepare key and value from constant_embs:
        # constant_embs: [B, n_states, n_atoms, 2, embed_dim] → [flat_batch, 2, embed_dim]
        const_seq = constant_embs.reshape(flat_batch, 2, embed_dim)
        # Transpose to (seq_len, batch, embed_dim): [2, flat_batch, embed_dim]
        key = const_seq.transpose(0, 1)
        value = key  # same as key
        
        # Run multi-head attention
        attn_output, _ = self.attention(query, key, value)
        # attn_output is [1, flat_batch, embed_dim] → remove sequence dim and reshape
        output = attn_output.squeeze(0).reshape(B, n_states, n_atoms, embed_dim)
        
        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))
        
        return output

    def add_loss(self, loss: torch.Tensor):
        # Integrate with your overall loss handling
        pass

class MultiHeadAttention(nn.Module):
    """Multi-head attention for computing atom embeddings with scaled dot-product attention.
    
    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]
      
    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        device: str = "cpu"
    ):
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        
        # Linear projections for query, key and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final projection after concatenating heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        # Move tensors to the specified device
        predicate_emb = predicate_emb.to(self.device)
        constant_embs = constant_embs.to(self.device)
        
        # Apply dropout
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)
        
        # Remove the singleton dimension from predicate_emb:
        # Now shape: [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(3)
        
        # Linear projections
        # For predicate: [B, n_states, n_atoms, embed_dim]
        # For constants: [B, n_states, n_atoms, 2, embed_dim]
        Q = self.q_proj(predicate_emb)  # [B, n_states, n_atoms, embed_dim]
        K = self.k_proj(constant_embs)    # [B, n_states, n_atoms, 2, embed_dim]
        V = self.v_proj(constant_embs)    # [B, n_states, n_atoms, 2, embed_dim]
        
        # Reshape for multi-head attention:
        # Q: reshape to [B, n_states, n_atoms, num_heads, head_dim] then add a singleton constant dim
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.head_dim).unsqueeze(3)  # [B, n_states, n_atoms, 1, num_heads, head_dim]
        
        # K and V: reshape to [B, n_states, n_atoms, 2, num_heads, head_dim]
        K = K.view(*K.shape[:-1], self.num_heads, self.head_dim)
        V = V.view(*V.shape[:-1], self.num_heads, self.head_dim)
        
        # Transpose to bring num_heads before the constant dimension:
        # Q becomes [B, n_states, n_atoms, num_heads, 1, head_dim]
        Q = Q.transpose(3, 4)
        # K and V become [B, n_states, n_atoms, num_heads, 2, head_dim]
        K = K.transpose(3, 4)
        V = V.transpose(3, 4)
        
        # Scaled dot-product attention:
        # Compute scores: Q @ K^T along last two dimensions.
        # K.transpose(-2, -1) changes K to [B, n_states, n_atoms, num_heads, head_dim, 2]
        # Resulting scores shape: [B, n_states, n_atoms, num_heads, 1, 2]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores.squeeze(-2)  # Now [B, n_states, n_atoms, num_heads, 2]
        scores = scores / math.sqrt(self.head_dim)
        
        # Compute attention weights with softmax over the constant dimension (last dim)
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_states, n_atoms, num_heads, 2]
        
        # Expand weights for weighted sum: [B, n_states, n_atoms, num_heads, 2, 1]
        attn_weights = attn_weights.unsqueeze(-1)
        # Weighted sum of V: multiply and sum over constant dimension -> [B, n_states, n_atoms, num_heads, head_dim]
        weighted_sum = (attn_weights * V).sum(dim=-2)
        
        # Concatenate heads: reshape from [B, n_states, n_atoms, num_heads, head_dim] to [B, n_states, n_atoms, embed_dim]
        concat = weighted_sum.view(*weighted_sum.shape[:-2], self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(concat)
        
        return output

class Attention(nn.Module):
    """Attention-based layer for computing atom embeddings using dot-product attention.
    
    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]
      
    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu"
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.regularization = regularization
        self.device = device
        self.to(device)
    
    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        # Apply dropout
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)
        
        # Remove the singleton arity dimension from predicate_emb
        # New shape: [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(3)
        
        # Compute dot-product attention scores.
        # Expand predicate_emb to [B, n_states, n_atoms, 1, embed_dim] (if not already) 
        # then multiply element-wise with constant_embs and sum over embed_dim.
        scores = (predicate_emb.unsqueeze(3) * constant_embs).sum(dim=-1)  # [B, n_states, n_atoms, 2]
        
        # Compute attention weights with softmax along the arity dimension (dim=-1).
        attn_weights = torch.softmax(scores, dim=-1)  # [B, n_states, n_atoms, 2]
        
        # Weighted sum of the constant embeddings using the attention weights.
        # Multiply weights (expanded to have embed_dim) and sum over the arity dimension.
        output = (attn_weights.unsqueeze(-1) * constant_embs).sum(dim=3)  # [B, n_states, n_atoms, embed_dim]
        
        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))
        
        return output
    
    def add_loss(self, loss: torch.Tensor):
        # Integrate with your overall loss handling
        pass

class TransE(nn.Module):
    """TransE layer for computing atom embeddings."""
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
        super(TransE, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regularization = regularization
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        predicate_emb = predicate_emb.squeeze(-2)
        head, tail = constant_embs[..., 0, :], constant_embs[..., 1, :]
        predicate_emb = self.dropout(predicate_emb)
        head = self.dropout(head)
        tail = self.dropout(tail)
        embeddings = predicate_emb + head - tail
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings


class ComplEx(nn.Module):
    """
    PyTorch implementation of the ComplEx layer.
    
    Expects:
      - predicate_emb: tensor of shape (batch_size,..., 2 * embedding_size)
      - constant_embs: tensor of shape (batch_size,..., 2, 2 * embedding_size)
        where constant_embs[:, 0, :] is the head and constant_embs[:, 1, :] is the tail.
    """
    def __init__(self, dropout_rate: float = 0.0, 
                 regularization: float = 0.0, 
                 regularization_n3: float = 0.0,
                 device="cpu"):
        super(ComplEx, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)
    
    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        predicate_emb = predicate_emb.squeeze(-2)
        # Reset any accumulated regularization loss
        self.reg_loss = 0.0

        # Apply dropout to both predicate and constant embeddings.
        if self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)

        # Split predicate embedding into real and imaginary parts.
        # Expect predicate_emb shape: (batch_size,..., 2 * embedding_size)
        Rr, Ri = torch.chunk(predicate_emb, 2, dim=-1)  # each: (batch_size,..., embedding_size)
        # Extract head and tail embeddings.
        # Expect constant_embs shape: (batch_size,..., 2, 2 * embedding_size)
        head, tail = constant_embs[..., 0, :], constant_embs[..., 1, :]

        # Split head and tail into their real and imaginary parts.
        h_r, h_i = torch.chunk(head, 2, dim=-1)
        t_r, t_i = torch.chunk(tail, 2, dim=-1)

        # Compute ComplEx score:
        # e1 = h_r * t_r * Rr, e2 = h_i * t_i * Rr, e3 = h_r * t_i * Ri, e4 = h_i * t_r * Ri.
        # Final score: e1 + e2 + e3 - e4.
        e1 = h_r * t_r * Rr
        e2 = h_i * t_i * Rr
        e3 = h_r * t_i * Ri
        e4 = h_i * t_r * Ri
        embeddings = e1 + e2 + e3 - e4

        # L2 regularization loss on the relation embeddings.
        if self.regularization > 0:
            self.reg_loss += self.regularization * (Rr.norm(p=2) + Ri.norm(p=2))
        # third-order regularization loss.
        if self.regularization_n3 > 0:
            abs_head = head.abs()
            abs_tail = tail.abs()
            abs_R = torch.cat([Rr.abs(), Ri.abs()], dim=1)
            self.reg_loss += self.regularization_n3 * (
                torch.sum(abs_head ** 3) +
                torch.sum(abs_tail ** 3) +
                torch.sum(abs_R ** 3)
            )
        return embeddings


class RotatE(nn.Module):
    """
    PyTorch implementation of the RotatE layer.
    
    Implements the relation as a rotation in the complex space.
    
    Expects:
      - predicate_emb: tensor of shape (batch_size,..., embedding_size)
          (each value is interpreted as an angle, later scaled by norm_factor)
      - constant_embs: tensor of shape (batch_size,..., 2, 2 * atom_embedding_size)
          where constant_embs[:, 0, :] is the head and constant_embs[:, 1, :] is the tail,
          with head/tail split into real and imaginary parts.
    """
    margin = 6.0  # also called gamma
    epsilon = 0.5

    def __init__(self, 
                 atom_embedding_size: int, 
                 dropout_rate: float = 0.0, 
                 regularization: float = 0.0, 
                 regularization_n3: float = 0.0,
                 device="cpu"):
        super(RotatE, self).__init__()
        self.atom_embedding_size = atom_embedding_size
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.device = device
        self.to(device)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Determine the scaling factor.
        self.embedding_range = (self.margin + self.epsilon) / atom_embedding_size
        self.norm_factor = math.pi / self.embedding_range

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        predicate_emb = predicate_emb.squeeze(-2)

        # Reset any accumulated regularization loss.
        self.reg_loss = 0.0

        # Apply dropout.
        if  self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)

        # Extract head and tail embeddings.
        head, tail = constant_embs[..., 0, :], constant_embs[..., 1, :]

        # Handle empty batch edge case.
        if head.size(0) == 0 or tail.size(0) == 0:
            return torch.zeros((0, self.atom_embedding_size), device=self.device)

        # Split head and tail into real and imaginary parts.
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Scale the predicate embedding (interpreted as a phase/angle).
        phase_relation = predicate_emb * self.norm_factor
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Rotate the tail entity and subtract the head entity.
        re_score = re_relation * re_tail + im_relation * im_tail - re_head
        im_score = re_relation * im_tail - im_relation * re_tail - im_head

        # Compute the L2 norm per embedding dimension.
        # (Using clamp to avoid numerical issues.)
        score = torch.sqrt(torch.clamp(re_score ** 2 + im_score ** 2, min=1e-9))

        if self.regularization > 0:
            self.reg_loss += self.regularization * predicate_emb.norm(p=2)
        if self.regularization_n3 > 0:
            self.reg_loss += self.regularization_n3 * (
                torch.sum(torch.abs(head) ** 3) +
                torch.sum(torch.abs(tail) ** 3)
            )
        return score

class Concat_Atoms(nn.Module):
    """Concat the predicate and constant embeddings."""
    def __init__(self, 
                atom_embedding_dim: int, 
                max_arity: int,
                dropout_rate: float=0.0, 
                regularization: float=0.0, 
                device="cpu"):
        super(Concat_Atoms, self).__init__()
        self.max_arity = max_arity
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if regularization > 0:
            self.regularization = regularization
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        predicate_emb = predicate_emb.squeeze(-2)  # Remove unnecessary dimension if present
        if self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)  
            
        # Determine the number of constants in the constant_embs
        num_constants = constant_embs.size(-2)
        # Pad the embeddings with zeros to reach 10 constants
        if num_constants < self.max_arity:
            padding_tensor = torch.zeros(*constant_embs.shape[:-2], self.max_arity - num_constants, constant_embs.size(-1), device=self.device)
            constant_embs = torch.cat([constant_embs, padding_tensor], dim=-2)
        # Concatenate constant embeddings along the last dimension
        constant_embs = constant_embs.view(*constant_embs.shape[:-2], -1)
        embeddings = torch.cat([predicate_emb, constant_embs], dim=-1)

        
        
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
            
        return embeddings

class Concat_States(nn.Module):
    """Concat the atom embeddings."""
    def __init__(self, 
                padding_atoms: int,
                dropout_rate: float=0.0, 
                regularization: float=0.0, 
                device="cpu"):
        super(Concat_States, self).__init__()
        self.padding_atoms = padding_atoms
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if regularization > 0:
            self.regularization = regularization
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        if self.regularization > 0:
            self.add_loss(self.regularization * atom_embeddings.norm(p=2))

        # Concatenate constant embeddings along the last dimension
        atom_embeddings = atom_embeddings.view(*atom_embeddings.shape[:-2], -1)
        return atom_embeddings


class Sum_state(nn.Module):
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
        super(Sum_state, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        if self.regularization > 0:
            self.add_loss(self.regularization * atom_embeddings.norm(p=2))
        return atom_embeddings.sum(dim=-2)

class Mean_state(nn.Module):
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
        super(Mean_state, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        if self.regularization > 0:
            self.add_loss(self.regularization * atom_embeddings.norm(p=2))
        return atom_embeddings.mean(dim=-2)

    # def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:

    #     # 1. Sum atom embeddings along the atom dimension (-2) to get state embeddings
    #     # Shape changes from [1, n, 5, 64] -> [1, n, 64]
    #     state_embeddings_summed = atom_embeddings.sum(dim=-2)

    #     # 2. Normalize the resulting summed state embeddings
    #     # Normalization is done along the embedding dimension (-1)
    #     # Shape remains [1, n, 64]
    #     normalized_state_embeddings = F.normalize(state_embeddings_summed, p=2, dim=-1)

    #     return normalized_state_embeddings
    
class Sum_atom(nn.Module):
    """For atom or state embeddings, simply sum the embeddings of the constants&predicates or atoms."""
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
        super(Sum_atom, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:

        predicate_emb = predicate_emb.squeeze(-2)  # Remove unnecessary dimension if present
        if self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)  # Apply dropout to all constants

        embeddings = predicate_emb - constant_embs.sum(dim=-2)
                
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
            
        return embeddings



class HybridConstantEmbedder(nn.Module):
    def __init__(self, num_regular_constants, num_image_constants, image_data, embedding_dim, device):
        super().__init__()
        self.num_regular = num_regular_constants # Number of regular constants, i.e., symbols with normal embeddings
        self.num_image = num_image_constants
        self.embedding_dim = embedding_dim
        
        # Regular constants use standard embeddings
        self.regular_embedder = nn.Embedding(num_regular_constants, embedding_dim)
        
        # CNN for image constants
        self.image_embedder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embedding_dim)
        )
        
        # Store pre-loaded image data with proper indexing
        self.register_buffer('image_data', image_data.to(device))

    def forward(self, indices):
        # Initialize output tensor
        embeddings = torch.zeros(*indices.shape, self.embedding_dim, 
                               device=indices.device)
        
        # Create masks using valid index ranges
        image_mask = (indices >= 0) & (indices < self.num_image)
        regular_mask = ~image_mask
        
        # Process image indices
        if image_mask.any():
            # Get valid image indices without filtering
            image_indices = indices[image_mask]
            
            # Directly index using valid positions
            selected_images = self.image_data[image_indices.long()]
            
            # Add channel dimension if missing (for single images)
            if selected_images.dim() == 3:
                selected_images = selected_images.unsqueeze(1)
                
            # Process and store embeddings
            image_embeds = self.image_embedder(selected_images)
            embeddings[image_mask] = image_embeds.view(-1, self.embedding_dim)
        
        # Process regular indices
        if regular_mask.any():
            regular_indices = indices[regular_mask] - self.num_image
            embeddings[regular_mask] = self.regular_embedder(regular_indices.long())
            
        return embeddings


class ConstantEmbeddings(nn.Module):
    """Module to handle constant embeddings per domain."""
    def __init__(self, num_constants: int, embedding_dim: int, regularization=0.0, device="cpu"): 
        super(ConstantEmbeddings, self).__init__()
        self.embedder = nn.Embedding(num_constants+1, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device
        self.to(device)

        # torch.nn.init.xavier_uniform_(self.embedder.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder(indices)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class PredicateEmbeddings(nn.Module):
    """Module to handle predicate embeddings."""
    def __init__(self, num_predicates: int, embedding_dim: int, regularization=0.0, device="cpu"):
        super(PredicateEmbeddings, self).__init__()
        #+1 for True, +1 for False, +1 for padding
        self.embedder = nn.Embedding(num_predicates+3, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device
        self.to(device)

        # torch.nn.init.xavier_uniform_(self.embedder.weight)


    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder(indices)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def to(self, device):
        super().to(device)
        self.device = device
        return self

      

def Emb_Atom_Factory(name: str='transe', 
            embedding_dim: int=-1, 
            max_arity: int=2,
            regularization: float=0.0,
            dropout_rate: float=0.0,
            device="cpu") -> nn.Module:
    
    if name.casefold() == 'transe':
        return TransE(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'complex':
        return ComplEx(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'rotate':
        return RotatE(embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'concat':
        return Concat_Atoms(embedding_dim, max_arity, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'sum':
        return Sum_atom(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'transformer':
        return Transformer(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'rnn':
        return RNN(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'attention':
        # return Attention(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
        return MultiHeadAttention(embed_dim=embedding_dim, dropout_rate=dropout_rate, device=device)
    else:
        raise ValueError(f"Unknown KGE model: {name}")


def Emb_State_Factory(name: str='transe', 
            embedding_dim: int=-1, 
            padding_atoms: int=10,
            regularization: float=0.0,
            dropout_rate: float=0.0,
            device="cpu") -> nn.Module:
    
    if name.casefold() == 'concat':
        return Concat_States(padding_atoms, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'sum':
        return Sum_state(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'mean':
        return Mean_state(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'rnn':
        return RNN_state(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'transformer':
        return Transformer_state(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    else:
        raise ValueError(f"Unknown KGE model: {name}")

class EmbedderLearnable(nn.Module):
    """
    Learnable embedder combining constant/predicate embeddings,
    atom composition, and state aggregation. Handles batch processing.
    """
    def __init__(self,
                 n_constants: int = 0,
                 n_predicates: int = 0,
                 n_vars: int = 0,
                 max_arity: int = 2,
                 padding_atoms: int = 10,
                 atom_embedder: str = 'transe',
                 state_embedder: str = 'sum',
                 constant_embedding_size: int = 64,
                 predicate_embedding_size: int = 64,
                 atom_embedding_size: int = 64, # Often same as const/pred for simple models
                 state_embedding_size: int = 64, # Final output dimension
                 kge_regularization: float = 0.0,
                 kge_dropout_rate: float = 0.0,
                 device: str = "cpu",
                 n_image_constants: int = 0,
                 image_dict: Optional[Dict[str, torch.Tensor]] = None, # Use Dict for clarity
                 n_body_constants: Optional[int] = None): # Keep n_body_constants if used elsewhere

        super(EmbedderLearnable, self).__init__()
        self.device = device
        self.embed_dim = state_embedding_size # Store the final output dimension

        # --- Constant Embedder ---
        image_data_tensor = None
        if image_dict and n_image_constants > 0:
            # Process image data (assuming previous logic is correct)
            image_data_list = []
            for k in sorted(image_dict.keys()): # Ensure consistent order
                 pair = image_dict[k]
                 # Assuming pair structure needs careful handling based on dataset
                 # This part needs verification based on how image_dict is structured
                 # Example: image_data_list.extend([img_tensor1, img_tensor2, ...])
                 image_data_list.extend([pair[0][0], pair[0][1], pair[1][0], pair[1][1]]) # Example based on provided code

            if image_data_list:
                 image_data_tensor = torch.stack(image_data_list).unsqueeze(1) # [N, 1, H, W]
                 image_data_tensor = image_data_tensor.to(device)
            else:
                 n_image_constants = 0 # No images found, revert

        self.n_image_constants = n_image_constants
        num_regular_constants = n_constants + n_vars - n_image_constants

        if self.n_image_constants > 0 and image_data_tensor is not None:
             self.constant_embedder = HybridConstantEmbedder(
                 num_regular_constants=num_regular_constants,
                 num_image_constants=self.n_image_constants,
                 image_data=image_data_tensor,
                 embedding_dim=constant_embedding_size,
                 device=device
             )
        else:
             # Note: num_constants includes padding (0), constants (1..N), variables (N+1..)
             total_const_var_count = n_constants + n_vars + 1 # +1 for padding idx 0
             self.constant_embedder = ConstantEmbeddings(
                 num_constants=total_const_var_count, # Pass total count
                 embedding_dim=constant_embedding_size,
                 regularization=kge_regularization,
                 device=device
             )

        # --- Predicate Embedder ---
        # num_predicates includes padding (0), preds (1..M), True (M+1), False (M+2), End (M+3) if used
        total_predicate_count = n_predicates + 3 + 1 # +1 for padding_idx 0
        self.predicate_embedder = PredicateEmbeddings(
            num_predicates=total_predicate_count, # Pass total count
            embedding_dim=predicate_embedding_size,
            regularization=kge_regularization,
            device=device
        )

        # --- Atom Composition ---
        self.atom_embedder = Emb_Atom_Factory(
            name=atom_embedder,
            embedding_dim=atom_embedding_size, # Input dim often constant/pred size, output is atom_embedding_size
            max_arity=max_arity,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device
        )

        # --- State Aggregation ---
        self.state_embedder = Emb_State_Factory(
            name=state_embedder,
            embedding_dim=state_embedding_size, # Input is atom_embedding_size, output is state_embedding_size
            padding_atoms=padding_atoms,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device
        )

        self.to(device)


    def get_embeddings_batch(self, sub_indices: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Get state embeddings for a batch of sub-indices using learnable components.

        Args:
            sub_indices: Tensor containing sub-indices (P, S, O) for atoms.
                         Expected shape: (bs, ..., n_atoms, 3)
                         Examples:
                           - Current state: (bs, n_atoms, 3)
                           - Derived states (actions): (bs, n_states, n_atoms, 3)
           verbose: If True, print intermediate shapes.

        Returns:
            Tensor of state embeddings.
            Expected shape: (bs, ..., embed_dim) where embed_dim is state_embedding_size
            Examples:
              - Current state output: (bs, embed_dim)
              - Derived states output: (bs, n_states, embed_dim)
        """
        if verbose: print('\n--- Learnable Embedder ---')
        if verbose: print(f'Input sub_indices: {sub_indices.shape}') # e.g., (bs, n_atoms, 3) or (bs, n_states, n_atoms, 3)

        # Ensure indices are long type
        sub_indices = sub_indices.long()

        # 1. Get Constant/Variable Embeddings
        constant_indices = sub_indices[..., 1:] # (bs, ..., n_atoms, 2)
        if verbose: print(f'Constant indices: {constant_indices.shape}')
        constant_embeddings = self.constant_embedder(constant_indices) # (bs, ..., n_atoms, 2, const_embed_dim)
        if verbose: print(f'Constant embeddings: {constant_embeddings.shape}')

        # 2. Get Predicate Embeddings
        predicate_indices = sub_indices[..., 0].unsqueeze(-1) # (bs, ..., n_atoms, 1)
        if verbose: print(f'Predicate indices: {predicate_indices.shape}')
        predicate_embeddings = self.predicate_embedder(predicate_indices) # (bs, ..., n_atoms, 1, pred_embed_dim)
        if verbose: print(f'Predicate embeddings: {predicate_embeddings.shape}')

        # 3. Compute Atom Embeddings
        # Atom embedder combines predicate and constant embeddings
        atom_embeddings = self.atom_embedder(predicate_embeddings, constant_embeddings) # (bs, ..., n_atoms, atom_embed_dim)
        if verbose: print(f'Atom embeddings: {atom_embeddings.shape}')

        # 4. Compute State Embeddings
        # State embedder aggregates atom embeddings
        # Create padding mask based on predicate index (0 means padding atom)
        # Shape: (bs, ..., n_atoms)
        padding_atom_mask = (sub_indices[..., 0] == 0) # PADDING_VALUE is 0

        # Pass mask to state embedder if it accepts it (like Attention_State)
        if isinstance(self.state_embedder, Attention_State):
             state_embeddings = self.state_embedder(atom_embeddings, padding_mask=padding_atom_mask)
        else:
             # For Sum, Mean, RNN, Transformer state embedders, mask is implicitly handled
             # or needs manual application if averaging/summing over non-padded items.
             # Sum/Mean might need adjustment to only sum/average non-padded atoms if not handled internally.
             # Example for manual masking with sum/mean:
             # if isinstance(self.state_embedder, (Sum_state, Mean_state)):
             #    mask_expanded = (~padding_atom_mask).unsqueeze(-1).float() # (bs, ..., n_atoms, 1)
             #    masked_atoms = atom_embeddings * mask_expanded
             #    if isinstance(self.state_embedder, Sum_state):
             #        state_embeddings = masked_atoms.sum(dim=-2) # Sum over n_atoms
             #    else: # Mean_state
             #        non_pad_count = mask_expanded.sum(dim=-2).clamp(min=1e-8)
             #        state_embeddings = masked_atoms.sum(dim=-2) / non_pad_count
             # else: # Assume RNN/Transformer handle sequence length or use all
             state_embeddings = self.state_embedder(atom_embeddings)

        # Final state embedding shape: (bs, ..., state_embed_dim)
        if verbose: print(f'State embeddings: {state_embeddings.shape}')
        if verbose: print('--- End Embedder ---')

        # Squeeze the n_states dimension if the input was just (bs, n_atoms, 3)
        # This ensures output is (bs, embed_dim) for current state obs
        # And (bs, n_states, embed_dim) for derived state obs
        if sub_indices.dim() == 3: # Input was (bs, n_atoms, 3)
             if state_embeddings.dim() == 3 and state_embeddings.shape[1] == 1:
                 # Example: Input (bs, 10, 3), Output might be (bs, 1, 64) -> squeeze to (bs, 64)
                 state_embeddings = state_embeddings.squeeze(1)
             elif state_embeddings.dim() != 2: # Should be (bs, embed_dim)
                 print(f"Warning: Unexpected state embedding shape {state_embeddings.shape} for input {sub_indices.shape}")
        # If input was (bs, n_states, n_atoms, 3), output should be (bs, n_states, embed_dim), no squeeze needed.

        return state_embeddings

    def forward(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """Directly call get_embeddings_batch."""
        return self.get_embeddings_batch(sub_indices)

    def to(self, device):
        """Move all components to the specified device."""
        super().to(device)
        self.device = device
        # Ensure sub-modules are moved (redundant if super().to() works, but safe)
        self.constant_embedder.to(device)
        self.predicate_embedder.to(device)
        self.atom_embedder.to(device)
        self.state_embedder.to(device)
        return self
    



class get_embedder():
    def __init__(self, 
                args: dict,
                data_handler: DataHandler, 
                index_manager: IndexManager, 
                device: str):
        
        self.embedder = self._create_embedder(args, data_handler, index_manager, device)

    def _create_embedder(self, args, data_handler, index_manager, device):

        if args.learn_embeddings:

            return EmbedderLearnable(
                n_constants=data_handler.constant_no,
                n_predicates=data_handler.predicate_no if not args.end_proof_action else data_handler.predicate_no + 1,
                n_vars=data_handler.variable_no if args.rule_depend_var else args.variable_no,
                max_arity=data_handler.max_arity,
                padding_atoms = args.padding_atoms,
                atom_embedder=args.atom_embedder,
                state_embedder=args.state_embedder,
                constant_embedding_size=args.constant_embedding_size,
                predicate_embedding_size=args.predicate_embedding_size,
                atom_embedding_size=args.atom_embedding_size,
                device=device,
                n_image_constants=data_handler.constant_images_no if args.dataset_name == 'mnist_addition' else 0,
                image_dict=data_handler.images if args.dataset_name == 'mnist_addition' else None
            )
        else:

            constant_str2idx, predicate_str2idx = index_manager.constant_str2idx, index_manager.predicate_str2idx
            constant_idx2emb, predicate_idx2emb = read_embeddings(args.constant_emb_file, args.predicate_emb_file, constant_str2idx, predicate_str2idx)
            if args.rule_depend_var:
                constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, data_handler.variable_no)
            else:
                constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, args.variable_no)
            embedding_function = EmbedderNonLearnable(constant_idx2emb, predicate_idx2emb, device=device)
            return embedding_function