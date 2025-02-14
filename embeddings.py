import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pickle
from typing import Tuple, Optional
import math

from dataset import DataHandler
from environments.env_logic_gym import IndexManager

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
        
    # @property
    # def weight(self):
    #     """Return the embedding table weights."""
    #     return self.embedding_table
















class TransE(nn.Module):
    """TransE layer for computing atom embeddings."""
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
        super(TransE, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regularization = regularization
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        # print('predicate_emb:',predicate_emb.shape)
        # print('constant_embs:',constant_embs.shape)
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

class Concat(nn.Module):
    """TransE layer for computing atom embeddings."""
    def __init__(self, 
                 atom_embedding_dim: int, 
                 dropout_rate: float=0.0, 
                 regularization: float=0.0, 
                 device="cpu"):
        super(Concat, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.linear1 = nn.Linear(2*atom_embedding_dim, atom_embedding_dim)
        self.linear2 = nn.Linear(2*atom_embedding_dim, atom_embedding_dim)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if regularization > 0:
            self.regularization = regularization
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)  
            
        n = constant_embs.shape[-2]  # Get the number of constants
        # reduce the constants to a single embedding by applying a linear layer. After that, combine it with the predicate embedding
        for i in range(n):
            constant_emb = constant_embs[..., i, :]
            if i == 0:
                embeddings = constant_emb
            else:
                embeddings = self.linear1(torch.cat([embeddings, constant_emb], dim=-1))
        # apply a linear layer to the combined embeddings
        embeddings = self.linear2(torch.cat([embeddings, predicate_emb.squeeze(-2)], dim=-1))

        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
            
        return embeddings




class Sum(nn.Module):
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
        super(Sum, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        # print('atom_embeddings:',atom_embeddings.shape)
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        if self.regularization > 0:
            self.add_loss(self.regularization * atom_embeddings.norm(p=2))
        return atom_embeddings.sum(dim=-2)
    
# class Sum_atoms(nn.Module):
    # """For atom or state embeddings, simply sum the embeddings of the constants&predicates or atoms."""
    # def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"): 
    #     super(Sum, self).__init__()
    #     self.dropout_rate = dropout_rate
    #     self.regularization = regularization
    #     if dropout_rate > 0:
    #         self.dropout = nn.Dropout(p=dropout_rate)
    #     self.device = device
    #     self.to(device)

    # def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:

    #     predicate_emb = predicate_emb.squeeze(-2)  # Remove unnecessary dimension if present
    #     if self.dropout_rate > 0:
    #         predicate_emb = self.dropout(predicate_emb)
    #         constant_embs = self.dropout(constant_embs)  # Apply dropout to all constants

    #     n = constant_embs.shape[-2]  # Get the number of constants
    #     # Initialize the combined embedding with the predicate embedding
    #     embeddings = predicate_emb

    #     # Iterate through the constants and apply the operations
    #     for i in range(n):
    #         constant_emb = constant_embs[..., i, :]
    #         embeddings = embeddings - constant_emb
                
    #     if self.regularization > 0:
    #         self.add_loss(self.regularization * embeddings.norm(p=2))
            
    #     return embeddings



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
        self.embedder = nn.Embedding(num_predicates+3, embedding_dim, padding_idx=0)
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

      

def EmbFactory(name: str='transe', 
            embedding_dim: int=-1, 
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
        return Concat(embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'sum':
        return Sum(dropout_rate=dropout_rate, regularization=regularization, device=device)
        # return Sum_states(dropout_rate=dropout_rate, regularization=regularization, device=device)
    else:
        raise ValueError(f"Unknown KGE model: {name}")


class EmbedderLearnable(nn.Module):
    def __init__(self, 
                 n_constants: int = 0, 
                 n_predicates: int = 0, 
                 n_vars: int = 0, 
                 atom_embedder: str = 'transe', 
                 state_embedder: str = 'sum',
                 constant_embedding_size: int = 64, 
                 predicate_embedding_size: int = 64,
                 atom_embedding_size: int = 64, 
                 kge_regularization: float = 0.0,
                 kge_dropout_rate: float = 0.0, 
                 device: str = "cpu",
                 n_image_constants: int = 0,
                 image_dict: Optional[dict[str, torch.Tensor]] = None,
                 n_body_constants: Optional[int] = None):
        
        super(EmbedderLearnable, self).__init__()

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
        
        # Initialize embedder
        self.constant_embedder = HybridConstantEmbedder(
            num_regular_constants=num_regular_constants,
            num_image_constants=n_image_constants,
            image_data=image_data,
            embedding_dim=constant_embedding_size,
            device=device
        ) if n_image_constants > 0 else ConstantEmbeddings(
            num_constants=n_constants + n_vars,
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

        self.atom_embedder = EmbFactory(
            name=atom_embedder, #if n_image_constants == 0 else 'concat',
            embedding_dim=atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device
        )

        self.state_embedder = EmbFactory(
            name=state_embedder,
            # embedding_dim=state_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device
        )   

    # Keep existing methods unchanged
    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_indices = sub_indices[..., 1:]
        
        constant_embeddings = self.constant_embedder(constant_indices)
        predicate_embeddings = self.predicate_embedder(predicate_indices)
        
        atom_embeddings = self.atom_embedder(predicate_embeddings, constant_embeddings)
        state_embeddings = self.state_embedder(atom_embeddings)
        
        return state_embeddings

    def forward(self, sub_indices: torch.Tensor) -> torch.Tensor:
        return self.get_embeddings_batch(sub_indices)
    



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