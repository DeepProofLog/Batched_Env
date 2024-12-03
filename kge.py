import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pickle
from typing import Tuple


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
    - predicate_embeddings: Tensor of shape (batch_size, embedding_dim)
    - constant_embeddings: Tensor of shape (batch_size, 2, embedding_dim)
    
    Returns:
    - atom_embeddings: Tensor of shape (batch_size, embedding_dim)
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


        
class EmbeddingFunction:
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
        
    # @property
    # def weight(self):
    #     """Return the embedding table weights."""
    #     return self.embedding_table






class ConstantEmbeddings(nn.Module):
    """Module to handle constant embeddings per domain."""
    def __init__(self, num_constants: int, embedding_dim: int, regularization=0.0, device="cpu"): 
        super(ConstantEmbeddings, self).__init__()
        self.embedder = nn.Embedding(num_constants+1, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device
        self.to(device)

        torch.nn.init.xavier_uniform_(self.embedder.weight)

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
        self.embedder = nn.Embedding(num_predicates+1+2, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device
        self.to(device)

        torch.nn.init.xavier_uniform_(self.embedder.weight)


    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder(indices)
        # print('first 2 embeddings:',self.embedder.weight.shape,self.embedder.weight[:5,:10])
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def to(self, device):
        super().to(device)
        self.device = device
        return self

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

# Factory function (like KGEFactory)
def KGEFactory(name, embedding_dim: int, regularization: float=0.0, dropout_rate: float=0.0, device="cpu") -> nn.Module:
    if name.casefold() == 'transe':
        return TransE(dropout_rate, regularization, device)
    else:
        raise ValueError(f"Unknown KGE model: {name}")



class KGEModel(nn.Module):
    def __init__(self, n_constants, n_predicates, n_vars, kge: str, constant_embedding_size: int, predicate_embedding_size: int, 
                 atom_embedding_size: int, kge_regularization: float=0, kge_dropout_rate: float=0, device="cpu"):
        super(KGEModel, self).__init__()
        self.embed_dim = atom_embedding_size
        self.constant_embedder = ConstantEmbeddings(
            num_constants=n_constants+n_vars, 
            embedding_dim=constant_embedding_size,
            regularization=kge_regularization,device=device)

        self.predicate_embedder = PredicateEmbeddings(
            num_predicates=n_predicates,
            embedding_dim=predicate_embedding_size,
            regularization=kge_regularization,device=device)

        self.kge_embedder = KGEFactory(
            name=kge,
            embedding_dim=atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,device=device)
    
    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of indices using vectorized operations.
        
        Args:
            indices: Tensor containing indices
            
        Returns:
            Tensor of embeddings with shape [..., embedding_dim]
        """
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_indices = sub_indices[..., 1:]
        constant_embeddings = self.constant_embedder(constant_indices)
        predicate_embeddings = self.predicate_embedder(predicate_indices)
        atom_embeddings = self.kge_embedder(predicate_embeddings, constant_embeddings)
        state_embeddings = atom_embeddings.sum(dim=-2)
        return state_embeddings

    def forward(self,sub_indices):
        """Forward pass to calculate atom scores."""
        return self.get_embeddings_batch(sub_indices)