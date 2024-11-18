import torch
import torch.nn as nn

# Set the seed for reproducibility
torch.manual_seed(42)

# Step 1: Generate random embeddings for predicates and constants
batch_size = 256
embedding_dim = 64

# Predicates embedding: Shape (256, 64)
predicate_embeddings = torch.randn(batch_size, embedding_dim)

# Constants embedding: Shape (256, 2, 64)
constant_embeddings = torch.randn(batch_size, 2, embedding_dim)

# Step 2: Define a TransE-like function to create atom embeddings
def transE_embedding(predicate_embeddings, constant_embeddings):
    """
    TransE function to compute atom embeddings.
    
    Arguments:
    - predicate_embeddings: Tensor of shape (batch_size, embedding_dim)
    - constant_embeddings: Tensor of shape (batch_size, 2, embedding_dim)
    
    Returns:
    - atom_embeddings: Tensor of shape (batch_size, embedding_dim)
    """
    # Separate the constants
    constant_1 = constant_embeddings[:, 0, :]  # Shape: (256, 64)
    constant_2 = constant_embeddings[:, 1, :]  # Shape: (256, 64)
    
    # Compute the atom embedding using TransE formula
    atom_embeddings = predicate_embeddings + (constant_1 - constant_2)
    return atom_embeddings

# Step 3: Generate atom embeddings
atom_embeddings = transE_embedding(predicate_embeddings, constant_embeddings)

# Print the shapes to verify
print(f"Predicate embeddings shape: {predicate_embeddings.shape}")
print(f"Constant embeddings shape: {constant_embeddings.shape}")
print(f"Atom embeddings shape: {atom_embeddings.shape}")

# Optionally, you can print a sample of the resulting embeddings
print("\nSample atom embeddings:\n", atom_embeddings[:5])












class TransE(nn.Module):
    def __init__(self, atom_embedding_size, regularization=0.0, dropout_rate=0.0):
        super(TransE, self).__init__()
        self.atom_embedding_size = atom_embedding_size
        self.regularization = regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, predicate_embeddings, constant_embeddings):
        """
        Forward pass for TransE.
        
        Args:
        - predicate_embeddings (Tensor): Shape (batch_size, embedding_dim)
        - constant_embeddings (Tensor): Shape (batch_size, 2, embedding_dim)
        
        Returns:
        - atom_embeddings (Tensor): Shape (batch_size, embedding_dim)
        """
        # Apply dropout to embeddings
        predicate_embeddings = self.dropout(predicate_embeddings)
        constant_embeddings = self.dropout(constant_embeddings)
        
        # Split constants into head and tail embeddings
        head = constant_embeddings[:, 0, :]  # Shape: (batch_size, embedding_dim)
        tail = constant_embeddings[:, 1, :]  # Shape: (batch_size, embedding_dim)
        
        # Compute atom embeddings using the TransE formula
        atom_embeddings = predicate_embeddings + (head - tail)
        
        # Apply L2 regularization if specified
        if self.regularization > 0.0:
            self.add_regularization_loss(predicate_embeddings, head, tail)
        
        return atom_embeddings

    def add_regularization_loss(self, *embeddings):
        """
        Adds L2 regularization loss for embeddings.
        """
        for embedding in embeddings:
            self.regularization_loss = self.regularization * torch.sum(embedding ** 2)
    
    @staticmethod
    def output_layer():
        """
        Output layer for TransE. Applies a sigmoid activation.
        """
        return nn.Sigmoid()



def KGEFactory(name: str,
               atom_embedding_size: int,
               regularization: float,
               dropout_rate: float,
               relation_embedding_size: int = None):
    """
    Factory function to create Knowledge Graph Embedding layers.
    
    Args:
    - name (str): Name of the KGE model ('transe', 'complex', etc.)
    - atom_embedding_size (int): Size of atom embeddings
    - regularization (float): L2 regularization coefficient
    - dropout_rate (float): Dropout rate
    - relation_embedding_size (int, optional): Size of relation embeddings
    
    Returns:
    - A tuple of (KGE layer, output layer)
    """
    relation_embedding_size = relation_embedding_size or atom_embedding_size
    
    if name.casefold() == 'transe':
        return TransE(atom_embedding_size, regularization, dropout_rate), TransE.output_layer()
    
    # Add other models like ComplEx, DistMult, etc. as needed
    else:
        raise ValueError(f"Unknown KGE model: {name}")


class KGEModule(nn.Module):
    def __init__(self, kge, atom_embedding_size, regularization, dropout_rate):
        super(KGEModule, self).__init__()
        self.kge_embedder, self.output_layer = KGEFactory(
            name=kge,
            atom_embedding_size=atom_embedding_size,
            relation_embedding_size=atom_embedding_size,
            regularization=regularization,
            dropout_rate=dropout_rate
        )
    
    def forward(self, predicate_embeddings, constant_embeddings):
        # Get the atom embeddings using the KGE layer
        atom_embeddings = self.kge_embedder(predicate_embeddings, constant_embeddings)
        
        # Pass through the output layer (Sigmoid)
        outputs = self.output_layer()(atom_embeddings)
        return outputs

# Example usage
if __name__ == "__main__":
    batch_size = 256
    embedding_dim = 64
    predicate_embeddings = torch.randn(batch_size, embedding_dim)
    constant_embeddings = torch.randn(batch_size, 2, embedding_dim)
    
    model = KGEModule(kge='transe', atom_embedding_size=embedding_dim, regularization=0.01, dropout_rate=0.1)
    outputs = model(predicate_embeddings, constant_embeddings)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Sample outputs:\n{outputs[:5]}")