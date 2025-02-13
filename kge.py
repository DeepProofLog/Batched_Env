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
        self.embedder = nn.Embedding(num_predicates+3, embedding_dim, padding_idx=0)
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

    # def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:

    #     predicate_emb = predicate_emb.squeeze(-2)  # Remove unnecessary dimension if present
    #     predicate_emb = self.dropout(predicate_emb)
    #     constant_embs = self.dropout(constant_embs)  # Apply dropout to all constants

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

class Concat(nn.Module):
    """TransE layer for computing atom embeddings."""
    def __init__(self, n,embedding_dim: int, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"):
        super(Concat, self).__init__()
        self.linear1 = nn.Linear(2*embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(2*embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regularization = regularization
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:

        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)  
        print('predicate_emb:',predicate_emb.shape)
        print('constant_embs:',constant_embs.shape)
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
        print('embeddings:',embeddings.shape)

        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
            
        return embeddings


def KGEFactory(name, embedding_dim: int, n_body_constants=None, regularization: float=0.0, dropout_rate: float=0.0, device="cpu") -> nn.Module:
    if name.casefold() == 'transe':
        return TransE(dropout_rate, regularization, device)
    elif name.casefold() == 'concat':
        return Concat(n_body_constants, dropout_rate, regularization, embedding_dim, device)
    else:
        raise ValueError(f"Unknown KGE model: {name}")




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

    

from typing import Optional
class KGEModel(nn.Module):
    def __init__(self, n_constants, n_predicates, n_vars, kge: str, 
                 constant_embedding_size: int, predicate_embedding_size: int,
                 atom_embedding_size: int, kge_regularization: float = 0,
                 kge_dropout_rate: float = 0, device="cpu",
                 n_image_constants: int = 0,
                 image_dict: dict[str, torch.Tensor] = None,
                 n_body_constants: Optional[int] = None):
        
        super(KGEModel, self).__init__()
        self.embed_dim = atom_embedding_size
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
        
        self.kge_embedder = KGEFactory(
            name=kge, #if n_image_constants == 0 else 'concat',
            embedding_dim=atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device,
            n_body_constants=n_body_constants
        )

    # Keep existing methods unchanged
    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_indices = sub_indices[..., 1:]
        
        constant_embeddings = self.constant_embedder(constant_indices)
        predicate_embeddings = self.predicate_embedder(predicate_indices)
        
        atom_embeddings = self.kge_embedder(predicate_embeddings, constant_embeddings)
        state_embeddings = atom_embeddings.sum(dim=-2)
        
        return state_embeddings

    def forward(self, sub_indices):
        return self.get_embeddings_batch(sub_indices)
    


from dataset import DataHandler
from environments.env_logic_gym import IndexManager
class get_kge():
    def __init__(self, 
                args: dict,
                data_handler: DataHandler, 
                index_manager: IndexManager, 
                device: str, 
                n_body_constants: Optional[int] = None,
                end_proof_action: bool = False):
        
        self.n_body_constants = n_body_constants
        self.end_proof_action = end_proof_action
        self.kge = self._create_kge(args, data_handler, index_manager, device)
    def _create_kge(self, args, data_handler, index_manager, device):
        if args.learn_embeddings:
            kge_model = KGEModel(data_handler.constant_no,
                                 data_handler.predicate_no if not self.end_proof_action else data_handler.predicate_no + 1,
                                 data_handler.variable_no if args.rule_depend_var else args.variable_no,
                                 args.kge,
                                 constant_embedding_size=args.constant_embedding_size,
                                 predicate_embedding_size=args.predicate_embedding_size,
                                 atom_embedding_size=args.atom_embedding_size,
                                 device=device,
                                 n_image_constants=data_handler.constant_images_no if args.dataset_name == 'mnist_addition' else 0,
                                 image_dict=data_handler.images if args.dataset_name == 'mnist_addition' else None,
                                 n_body_constants=self.n_body_constants)
            return kge_model
        
        else:
            constant_str2idx, predicate_str2idx = index_manager.constant_str2idx, index_manager.predicate_str2idx
            constant_idx2emb, predicate_idx2emb = read_embeddings(args.constant_emb_file, args.predicate_emb_file, constant_str2idx, predicate_str2idx)
            if args.rule_depend_var:
                constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, data_handler.variable_no)
            else:
                constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, args.variable_no)
            embedding_function = EmbeddingFunction(constant_idx2emb, predicate_idx2emb, device=device)
            return embedding_function