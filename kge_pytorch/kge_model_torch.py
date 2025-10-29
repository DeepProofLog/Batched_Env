"""KGE models: RotatE, ComplEx, TuckER (PyTorch)."""
from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
from torch import nn


# ------------------------- RotatE -------------------------
class RotatE(nn.Module):
    """Complex-valued RotatE embedding model.
    Uses phase embeddings for relations and complex embeddings for entities.
    Score = gamma - || (h ∘ r) - t ||_p summed over components.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma: float = 12.0,
        p_norm: int = 1,
    ):
        super().__init__()
        assert dim % 2 == 0, "RotatE requires even dim (re,im)"
        self.dim = dim // 2
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.get_default_dtype()), requires_grad=False)
        self.p = p_norm

        self.ent_re = nn.Embedding(num_entities, self.dim)
        self.ent_im = nn.Embedding(num_entities, self.dim)
        self.rel_phase = nn.Embedding(num_relations, self.dim)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_re.weight, -bound, bound)
        nn.init.uniform_(self.ent_im.weight, -bound, bound)
        nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)
        self.project_entity_modulus_()

    @torch.no_grad()
    def project_entity_modulus_(self):
        re = self.ent_re.weight.data
        im = self.ent_im.weight.data
        mod = torch.clamp(torch.sqrt(re * re + im * im), min=1e-6)
        factor = torch.clamp(1.0 / mod, max=1.0)
        self.ent_re.weight.data = re * factor
        self.ent_im.weight.data = im * factor

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_re = self.ent_re(h); h_im = self.ent_im(h)
        t_re = self.ent_re(t); t_im = self.ent_im(t)
        phase = torch.remainder(self.rel_phase(r), 2 * math.pi)
        c, s = torch.cos(phase), torch.sin(phase)
        hr_re = h_re * c - h_im * s
        hr_im = h_re * s + h_im * c
        if self.p == 1:
            dist = (hr_re - t_re).abs() + (hr_im - t_im).abs()
        else:
            dist = torch.sqrt(((hr_re - t_re) ** 2 + (hr_im - t_im) ** 2) + 1e-9)
        dist = dist.sum(dim=-1)
        return self.gamma - dist

    def forward(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_h, p_r, p_t = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        n_h, n_r, n_t = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        pos_scores = self.score_triples(p_h, p_r, p_t)
        neg_scores = self.score_triples(n_h, n_r, n_t)
        return pos_scores, neg_scores


# ------------------------- ComplEx -------------------------
class ComplEx(nn.Module):
    """ComplEx bilinear model.
    Entities and relations are complex; score = Re( < h, r, conj(t) > ).
    Higher scores indicate more plausible triples.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        assert dim % 2 == 0, "ComplEx requires even dim (re,im)"
        self.dim = dim // 2

        # entity
        self.ent_re = nn.Embedding(num_entities, self.dim)
        self.ent_im = nn.Embedding(num_entities, self.dim)
        # relation
        self.rel_re = nn.Embedding(num_relations, self.dim)
        self.rel_im = nn.Embedding(num_relations, self.dim)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_re.weight, -bound, bound)
        nn.init.uniform_(self.ent_im.weight, -bound, bound)
        nn.init.uniform_(self.rel_re.weight, -bound, bound)
        nn.init.uniform_(self.rel_im.weight, -bound, bound)

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_re = self.ent_re(h); h_im = self.ent_im(h)
        t_re = self.ent_re(t); t_im = self.ent_im(t)
        r_re = self.rel_re(r); r_im = self.rel_im(r)

        # Re(<h, r, conj(t)>) = (h_re*r_re*t_re) + (h_im*r_re*t_im) + (h_re*r_im*t_im) - (h_im*r_im*t_re)
        s = h_re * r_re * t_re
        s = s + h_im * r_re * t_im
        s = s + h_re * r_im * t_im
        s = s - h_im * r_im * t_re
        return s.sum(dim=-1)

    def forward(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_h, p_r, p_t = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        n_h, n_r, n_t = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        pos_scores = self.score_triples(p_h, p_r, p_t)
        neg_scores = self.score_triples(n_h, n_r, n_t)
        return pos_scores, neg_scores


# ------------------------- TuckER -------------------------
class TuckER(nn.Module):
    """TuckER (Balažević et al. 2019): core tensor over entity/relation spaces.
    Score(h,r,t) = W ×_1 e_h ×_2 r ×_3 e_t
    - entity_dim = d_e
    - relation_dim = d_r
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int,
        relation_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_e = entity_dim
        d_r = relation_dim if relation_dim is not None else entity_dim

        self.entity_dim = d_e
        self.relation_dim = d_r

        self.ent = nn.Embedding(num_entities, d_e)
        self.rel = nn.Embedding(num_relations, d_r)

        # Core tensor W in R^{d_r x d_e x d_e}
        self.W = nn.Parameter(torch.empty(d_r, d_e, d_e))

        self.dropout_e = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_r = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier-like for embeddings and core
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.xavier_uniform_(self.W)

    def _score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e_h = self.dropout_e(self.ent(h))     # [B, d_e]
        e_t = self.dropout_e(self.ent(t))     # [B, d_e]
        e_r = self.dropout_r(self.rel(r))     # [B, d_r]

        # Compute W x2 r -> [B, d_e, d_e]
        # tensordot: (B,d_r) x (d_r,d_e,d_e) -> (B,d_e,d_e)
        Wr = torch.tensordot(e_r, self.W, dims=([1], [0]))
        # (B, d_e) x (B, d_e, d_e) -> (B, d_e)
        x = torch.bmm(e_h.unsqueeze(1), Wr).squeeze(1)
        # final bilinear with e_t -> (B,)
        scores = (x * e_t).sum(dim=-1)
        return scores

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._score(h, r, t)

    def forward(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_h, p_r, p_t = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        n_h, n_r, n_t = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        pos_scores = self._score(p_h, p_r, p_t)
        neg_scores = self._score(n_h, n_r, n_t)
        return pos_scores, neg_scores


# ------------------------- Factory -------------------------
def build_model(
    name: str,
    num_entities: int,
    num_relations: int,
    dim: int,
    gamma: float = 12.0,
    p_norm: int = 1,
    relation_dim: Optional[int] = None,
    dropout: float = 0.0,
):
    name = name.lower()
    if name in {"rotate", "rotatE"}:
        return RotatE(num_entities, num_relations, dim=dim, gamma=gamma, p_norm=p_norm)
    if name in {"complex", "complEx"}:
        return ComplEx(num_entities, num_relations, dim=dim)
    if name in {"tucker"}:
        return TuckER(num_entities, num_relations, entity_dim=dim, relation_dim=relation_dim or dim, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")