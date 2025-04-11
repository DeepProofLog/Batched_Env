import torch
import logging
from typing import List, Optional, Tuple
# Assuming these are imported correctly from your project structure
from index_manager_v2 import IndexManager # Use the refactored version
from pykeen.sampling import BasicNegativeSampler # Or your specific sampler class
# from utils import Term # Assuming Term is defined as above or imported

# Placeholder for Term if not imported
class Term:
    def __init__(self, predicate: str, args: List[any]):
        self.predicate = predicate
        self.args = args
    def __str__(self): return f"{self.predicate}({','.join(map(str, self.args))})"
    def __repr__(self): return str(self)
    def __eq__(self, other): return isinstance(other, Term) and self.predicate == other.predicate and self.args == other.args
    def __hash__(self):
        try: return hash((self.predicate, tuple(self.args)))
        except TypeError: return hash((self.predicate, tuple(map(str, self.args))))


def generate_dynamic_negatives(
    positive_terms: List[Term],
    index_manager: IndexManager,
    sampler: BasicNegativeSampler, # Expecting a PyKeen-like sampler
    device: torch.device,
) -> List[Optional[Term]]:
    """
    Generates one negative Term for each positive Term using dynamic corruption.

    Args:
        positive_terms: A list of positive Term objects to corrupt.
        index_manager: The IndexManager instance for mapping.
        sampler: The negative sampler instance (e.g., BasicNegativeSamplerCustom).
        device: The torch device.

    Returns:
        A list of the same length as positive_terms, containing either a
        negative Term or None if corruption failed for a specific term.
    """
    negative_terms_result: List[Optional[Term]] = [None] * len(positive_terms)
    if not positive_terms:
        return negative_terms_result

    # 1. Map positive Terms to tensor indices
    positive_triples_list = []
    valid_pos_indices_map = {} # Map original index to index in positive_triples_list

    for i, term in enumerate(positive_terms):
        indices = index_manager.map_term_to_triple_indices(term)
        if indices:
            valid_index = len(positive_triples_list) # Index within the tensor to be created
            positive_triples_list.append(indices)
            valid_pos_indices_map[valid_index] = i # Map tensor index back to original list index
        else:
            logging.warning(f"Could not map positive term {term} to indices for dynamic corruption. Skipping.")

    if not positive_triples_list:
        logging.warning("Dynamic corruption: No positive terms could be mapped to indices.")
        return negative_terms_result # Return list of Nones

    positive_batch_tensor = torch.tensor(positive_triples_list, dtype=torch.long, device=device)

    # 2. Corrupt the batch using the sampler
    try:
        # corrupt_batch returns shape (batch_shape, num_negs_per_pos, 3)
        # We want one negative per positive. Assume num_negs=1 or take first.
        negative_batch_tensor = sampler.corrupt_batch(positive_batch_tensor)

        # Ensure we take only one negative per positive if sampler returns more
        if negative_batch_tensor.dim() > 2:
            if negative_batch_tensor.shape[1] > 0: # Check if sampler returned any negatives
                 negative_batch_tensor = negative_batch_tensor[:, 0, :] # Take first negative
            else:
                 logging.warning("Sampler returned 0 negatives per positive.")
                 return negative_terms_result # Return list of Nones

        if negative_batch_tensor.shape[0] != len(positive_triples_list):
             logging.error(f"Sampler returned unexpected number of negatives. Expected {len(positive_triples_list)}, got {negative_batch_tensor.shape[0]}")
             return negative_terms_result # Return list of Nones

        # 3. Convert negative tensor indices back to Terms
        for i in range(negative_batch_tensor.shape[0]):
            original_pos_idx = valid_pos_indices_map[i] # Get original index in positive_terms list
            h, r, t = negative_batch_tensor[i].tolist()
            neg_term = index_manager.map_indices_to_term(h, r, t)
            if neg_term:
                negative_terms_result[original_pos_idx] = neg_term
            else:
                logging.warning(f"Could not map negative indices ({h},{r},{t}) back to term for original term: {positive_terms[original_pos_idx]}.")
                # Keep None in the result list for this term

    except Exception as e:
        logging.error(f"Error during dynamic negative sampling: {e}", exc_info=True)
        # Return list of Nones on error

    return negative_terms_result
