## State and Atom Structure

A state `s` is represented as a finite set of atoms

\[
s = \{a_0, a_1, \dots, a_{A-1}\}
\]

where each atom is a triple

\[
a_i = (r_i, h_i, t_i)
\]

The order of atoms is not meaningful, **except for `a_0`**, which is a distinguished *query atom* selected by the engine to be proven. The remaining atoms

\[
C = \{a_1, \dots, a_{A-1}\}
\]

form an unordered context set.

---

## Atom Embedding

Let \( e_r \in \mathbb{R}^{d_r} \), \( e_h \in \mathbb{R}^{d_e} \), \( e_t \in \mathbb{R}^{d_e} \) be learned embeddings for relations and entities, and let \( e_q \in \mathbb{R}^{d_q} \) be a learned embedding indicating whether an atom is the query atom.

### Role-aware Atom Encoder

Each atom is embedded via a role-aware encoder

\[
z_i = \phi(a_i) =
\mathrm{MLP}\!\left(
\left[
e_{r_i} \;\|\; e_{h_i} \;\|\; e_{t_i} \;\|\; e_{q_i}
\right]
\right),
\qquad z_i \in \mathbb{R}^d
\]

where `||` denotes concatenation and \( e_{q_i} \) encodes whether \( i = 0 \).

A cheaper linear alternative is

\[
z_i = \sigma\!\left(
W_r e_{r_i} +
W_h e_{h_i} +
W_t e_{t_i} +
W_q e_{q_i} +
b
\right)
\]

---

## Query-conditioned State Factorization

Let

\[
z_q = z_0
\]

be the embedding of the query atom.

### Query-conditioned Context Encoding

Each context atom \( a_i \in C \) is conditioned on the query via

\[
u_i = g(z_i, z_q)
\]

for example

\[
u_i = \sigma\!\left(
W_c [z_i \;\|\; z_q]
\right)
\]

---

### Permutation-invariant Pooling

Let \( m_i \in \{0,1\} \) be a padding mask.

**Sum pooling**
\[
c = \sum_{i=1}^{A-1} m_i \, u_i
\]

**Attention pooling**
\[
\alpha_i =
\frac{\exp(w^\top u_i)}
{\sum_{j=1}^{A-1} \exp(w^\top u_j)},
\qquad
c = \sum_{i=1}^{A-1} \alpha_i \, u_i
\]

---

### State Embedding

The final state embedding is

\[
s = \rho\!\left(
[z_q \;\|\; c]
\right),
\qquad s \in \mathbb{R}^d
\]

---

## Factorized Value Function

The value function is decomposed over atoms

\[
V(s) = v_q + \sum_{i=1}^{A-1} m_i \, v_i
\]

where

\[
v_q = \psi_q(z_q),
\qquad
v_i = \psi(u_i)
\]

This preserves permutation invariance over context atoms while allowing the query atom to dominate the value estimate.

---

## Policy and Value Sharing

A shared backbone computes atom embeddings \( \{z_i\} \), conditioned embeddings \( \{u_i\} \), and the state embedding \( s \).

- **Policy**: scores candidate successor states using the state embedding \( s \) (or embeddings \( s' \) for each candidate).
- **Value**: outputs \( V(s) \) using the factorized decomposition above.

---

## Optional Extensions

### Pairwise Interactions

For small \( A \), pairwise interactions may be added

\[
V(s) = \sum_i v_i + \sum_{i<j} v_2(a_i, a_j)
\]

optionally conditioned on the query atom.

---

### Set Transformer Variant

Treat \( z_q \) as a query token and apply cross-attention to the context set

\[
z_q' = \mathrm{Attn}(z_q, \{z_i\}_{i=1}^{A-1}),
\qquad
s = z_q'
\]

---

## Summary

The proposed design:

- preserves permutation invariance over context atoms  
- treats the query atom explicitly and asymmetrically  
- supports factorized value estimation  
- avoids information loss from naive mean/sum pooling  
- remains compatible with shared policy–value backbones  
