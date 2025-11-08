import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pickle
from typing import Tuple, Optional, Dict
import math

from data_handler import DataHandler


# ------------------ Load embeddings functions------------------

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
    def __init__(self, constant_idx2emb: torch.Tensor, predicate_idx2emb: torch.Tensor, device="cpu"):
        """
        Initialize the embedding function.
        """
        self.embed_dim = constant_idx2emb.size(1)
        self.constant_idx2emb = constant_idx2emb.to(device)
        self.predicate_idx2emb = predicate_idx2emb.to(device)
        # Set padding embedding (index 0) to zeros
        self.constant_idx2emb[0] = 0
        self.predicate_idx2emb[0] = 0
        
    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of indices using vectorized operations.
        
        Args:
            indices: Tensor containing indices
            
        Returns:
            Tensor of embeddings with shape [..., embedding_dim]
        """
        # Not consider cache atom embedding for now
        # unique_indices = atom_indices.unique()
        # indices_in_dict = torch.tensor([idx in self.atom_idx2emb for idx in unique_indices])
        # precomputed_embeddings = torch.stack(
        #     [self.atom_idx2emb[idx.item()] for idx in unique_indices[indices_in_dict]]
        # )
        # computed_embeddings = torch.stack(
        #     [compute_embedding(idx.item()) for idx in unique_indices[~indices_in_dict]]
        # )

        # IF I GET "index out of range in self" ERROR, IT IS BECAUSE THE INDICES ARE OUT OF RANGE. ONCE I IMPLEMENT THE LOCAL INDICES, THIS WILL BE SOLVED
        # Look up the embeddings of predicates and args.
        # pred_arg_embeddings = F.embedding(sub_indices, self.embedding_table, padding_idx=0)
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_indices = sub_indices[..., 1:]
        predicate_embeddings = F.embedding(predicate_indices, self.predicate_idx2emb, padding_idx=0)
        constant_embeddings = F.embedding(constant_indices, self.constant_idx2emb, padding_idx=0)
        atom_embeddings = transE_embedding(predicate_embeddings, constant_embeddings)

        # # Sum pred & args embeddings to get atom embeddings.
        # pred_arg_embeddings = torch.cat([predicate_embeddings, constant_embeddings], dim=-2)
        # atom_embeddings = pred_arg_embeddings.sum(dim=-2)
        # Sum atom embeddings to get state embeddings.
        state_embeddings = atom_embeddings.sum(dim=-2)

        return state_embeddings
    
    def to(self, device):
        """Move embedding table to specified device."""
        self.device = device
        self.constant_idx2emb = self.constant_idx2emb.to(device)
        self.predicate_idx2emb = self.predicate_idx2emb.to(device)
        return self

# ------------------ End Load embeddings functions------------------



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
        # # Move tensors to the specified device
        # predicate_emb = predicate_emb.to(self.device)
        # constant_embs = constant_embs.to(self.device)
        
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
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
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
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.device = device
    
    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        predicate_emb = predicate_emb.squeeze(-2)
        # Reset any accumulated regularization loss
        self.reg_loss = 0.0

        # Apply dropout to both predicate and constant embeddings.
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
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if regularization > 0:
            self.regularization = regularization
        self.device = device

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



class ConstantEmbeddings(nn.Module):
    """Module to handle constant embeddings per domain."""
    def __init__(self, num_constants: int, embedding_dim: int, regularization=0.0, device="cpu"): 
        super(ConstantEmbeddings, self).__init__()
        # num_constants is the count, but indices go from 0 to num_constants
        # so we need num_constants+1 entries
        self.embedder = nn.Embedding(num_constants+1, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device
        # Move to device immediately
        self.to(device)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # nn.Embedding automatically handles device placement
        embeddings = self.embedder(indices)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings


class PredicateEmbeddings(nn.Module):
    """Module to handle predicate embeddings."""
    def __init__(self, num_predicates: int, embedding_dim: int, regularization=0.0, device="cpu"):
        super(PredicateEmbeddings, self).__init__()
        # num_predicates is the count, but indices go from 0 to num_predicates
        # so we need num_predicates+1 entries
        self.embedder = nn.Embedding(num_predicates+1, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device
        # Move to device immediately
        self.to(device)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # nn.Embedding automatically handles device placement
        embeddings = self.embedder(indices)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings
      

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
                 atom_embedding_size: int = 64, 
                 kge_regularization: float = 0.0,
                 kge_dropout_rate: float = 0.0, 
                 device: str = "cpu",
                 n_image_constants: int = 0,
                 image_dict: Optional[dict[str, torch.Tensor]] = None
                 ):
        
        super(EmbedderLearnable, self).__init__()
        
        # Store device for later use
        self.device = device
        self.embedding_dim = atom_embedding_size
        self.atom_embedding_size = atom_embedding_size

        # Process image data correctly
        image_data = []
        if image_dict:
            # Flatten dictionary values while preserving order
            for k in sorted(image_dict.keys()):
                pair = image_dict[k]
                image_data.extend([pair[0][0], pair[0][1], pair[1][0], pair[1][1]])
                
            image_data = torch.stack(image_data)
            print(f"Final image data shape: {image_data.shape}")  # Should be [N, 28, 28]
            
            # Add channel dimension
            image_data = image_data.unsqueeze(1)  # Shape becomes [N, 1, 28, 28]
        
        # Handle image constants
        self.n_image_constants = n_image_constants
        num_regular_constants = n_constants + n_vars - n_image_constants
        
        # Calculate total vocabulary size for embedding table
        # Need to accommodate: padding (0), constants (1..n_constants), 
        # template vars (n_constants+1..n_constants+template_var_no),
        # runtime vars (runtime_var_start..runtime_var_end)
        # The embedding table should be sized to the max index + 1
        total_constant_vocab = n_constants + n_vars + 1  # +1 for padding at index 0
        
        # Initialize embedder
        self.constant_embedder = ConstantEmbeddings(
            num_constants=total_constant_vocab,
            embedding_dim=constant_embedding_size,
            regularization=kge_regularization,
            device=device
        )
        
        self.predicate_embedder = PredicateEmbeddings(
            num_predicates=n_predicates,
            embedding_dim=predicate_embedding_size,
            regularization=kge_regularization,
            device=device
        )

        self.atom_embedder = Emb_Atom_Factory(
            name=atom_embedder, #if n_image_constants == 0 else 'concat',
            embedding_dim=atom_embedding_size,
            max_arity=max_arity,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device
        )

        self.state_embedder = Emb_State_Factory(
            name=state_embedder,
            embedding_dim=atom_embedding_size,
            padding_atoms=padding_atoms,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device
        )
        
        # Move entire module to device
        self.to(device)   

    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for a batch of sub-indices.
        Args:
            sub_indices: torch.Tensor of shape (batch_size=n_envs,n_states,n_atoms,3)
            2nd dim is to match the shape of derived_sub_indices
        Returns:
            torch.Tensor of shape (batch_size=n_envs,n_states,embedding_dim)
        """
        # Extract indices - use long() for embedding indices
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_indices = sub_indices[..., 1:]
        
        # Get embeddings - nn.Embedding handles device management
        constant_embeddings = self.constant_embedder(constant_indices)
        predicate_embeddings = self.predicate_embedder(predicate_indices)
        
        atom_embeddings = self.atom_embedder(predicate_embeddings, constant_embeddings)
        state_embeddings = self.state_embedder(atom_embeddings)
        
        return state_embeddings

    def forward(self, sub_indices: torch.Tensor) -> torch.Tensor:
        return self.get_embeddings_batch(sub_indices)
    



class get_embedder():
    """Factory class to create the appropriate embedder based on the configuration."""
    def __init__(self, 
                args: dict,
                data_handler: DataHandler, 
                constant_no: int,
                predicate_no: int,
                runtime_var_end_index: int,
                constant_str2idx: Dict[str, int],
                predicate_str2idx: Dict[str, int],
                constant_images_no: int = 0,
                device: str = "cpu"):
        """
        Initialize embedder factory.
        
        Args:
            args: Configuration arguments
            data_handler: DataHandler instance
            constant_no: Number of constants (without variables)
            predicate_no: Number of predicates
            runtime_var_end_index: Maximum index for variables
            constant_str2idx: Constant string to index mapping
            predicate_str2idx: Predicate string to index mapping
            constant_images_no: Number of image constants (0 if not using images)
            device: Device for embeddings
        """
        self.embedder = self._create_embedder(
            args, data_handler, constant_no, predicate_no, 
            runtime_var_end_index, constant_str2idx, predicate_str2idx,
            constant_images_no, device
        )

    def _create_embedder(self, args, data_handler, constant_no, predicate_no, 
                        runtime_var_end_index, constant_str2idx, predicate_str2idx,
                        constant_images_no, device):

        if args.learn_embeddings:
            # Calculate the actual maximum indices to size embedding tables correctly
            max_constant_idx = max(constant_str2idx.values()) if constant_str2idx else constant_no
            max_predicate_idx = max(predicate_str2idx.values()) if predicate_str2idx else predicate_no
            max_var_idx = runtime_var_end_index
            
            # Total constant vocabulary includes constants + all variables
            total_constant_vocab = max(max_constant_idx, max_var_idx) + 1
            total_predicate_vocab = max_predicate_idx + 1
            
            return EmbedderLearnable(
                n_constants=total_constant_vocab - 1,  # -1 because ConstantEmbeddings adds +1
                n_predicates=total_predicate_vocab - 1,  # -1 because PredicateEmbeddings adds +1
                n_vars=0,  # Already included in total_constant_vocab
                max_arity=data_handler.max_arity,
                padding_atoms = args.padding_atoms,
                atom_embedder=args.atom_embedder,
                state_embedder=args.state_embedder,
                constant_embedding_size=args.constant_embedding_size,
                predicate_embedding_size=args.predicate_embedding_size,
                atom_embedding_size=args.atom_embedding_size,
                device=device,
                n_image_constants=constant_images_no if args.dataset_name == 'mnist_addition' else 0,
                image_dict=data_handler.images if args.dataset_name == 'mnist_addition' else None
            )
        
        else:
            constant_idx2emb, predicate_idx2emb = read_embeddings(args.constant_emb_file, args.predicate_emb_file, constant_str2idx, predicate_str2idx)
            if args.rule_depend_var:
                constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, data_handler.variable_no)
            else:
                constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, args.variable_no)
            return EmbedderNonLearnable(constant_idx2emb, predicate_idx2emb, device=device)

