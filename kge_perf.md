# KGE Performance Report: Provable vs Non-Provable Queries

This report analyzes KGE (Knowledge Graph Embedding) model performance on queries
segmented by provability. A query is **provable** if the RL agent can derive it
through logical reasoning (depth >= 1), and **non-provable** otherwise (depth = -1).

Understanding KGE performance on these subsets helps identify the potential for
RL to improve link prediction through hybrid approaches.

## Summary

| Dataset | Split | Total | Provable | Non-Provable | Overall MRR | Provable MRR | Non-Provable MRR | Delta |
|---------|-------|-------|----------|--------------|-------------|--------------|------------------|-------|
| family | valid | 2799 | 2148 | 651 | 0.9481 | 0.9658 | 0.8896 | +0.0762 |
| family | test | 5626 | 4395 | 1231 | 0.9527 | 0.9694 | 0.8933 | +0.0760 |
| nations | valid | 199 | 21 | 178 | 0.4765 | 0.4460 | 0.4801 | -0.0341 |
| nations | test | 201 | 16 | 185 | 0.4993 | 0.5800 | 0.4924 | +0.0876 |
| wn18rr | valid | 2824 | 1480 | 1344 | 0.8356 | 0.9513 | 0.7082 | +0.2431 |
| wn18rr | test | 2924 | 1500 | 1424 | 0.8316 | 0.9511 | 0.7056 | +0.2455 |
| umls | valid | 1300 | 1089 | 211 | 0.3823 | 0.3773 | 0.4079 | -0.0306 |
| umls | test | 3245 | 2704 | 541 | 0.5119 | 0.5233 | 0.4548 | +0.0685 |
| pharmkg_full | valid | 46918 | 9945 | 36973 | 0.4065 | 0.3960 | 0.4094 | -0.0134 |
| pharmkg_full | test | 47408 | 9950 | 37458 | 0.4080 | 0.3976 | 0.4107 | -0.0131 |
| fb15k237 | valid | 17526 | 6964 | 10562 | 0.6498 | 0.6897 | 0.6236 | +0.0661 |
| fb15k237 | test | 20438 | 8139 | 12299 | 0.6456 | 0.6872 | 0.6181 | +0.0691 |

## Detailed Results

### family

#### Valid Split

- **Total queries**: 2799
- **Provable queries**: 2148 (76.7%)
- **Non-provable queries**: 651 (23.3%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.9481 | 0.9658 | 0.8896 |
| Hits@1 | 0.9182 | 0.9404 | 0.8449 |
| Hits@3 | 0.9757 | 0.9905 | 0.9270 |
| Hits@10 | 0.9841 | 0.9951 | 0.9478 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 651 | 0.8896 |
| 1 | 164 | 0.9102 |
| 2 | 1939 | 0.9702 |
| 4 | 45 | 0.9778 |

#### Test Split

- **Total queries**: 5626
- **Provable queries**: 4395 (78.1%)
- **Non-provable queries**: 1231 (21.9%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.9527 | 0.9694 | 0.8933 |
| Hits@1 | 0.9277 | 0.9465 | 0.8603 |
| Hits@3 | 0.9771 | 0.9926 | 0.9216 |
| Hits@10 | 0.9843 | 0.9973 | 0.9379 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 1231 | 0.8933 |
| 1 | 4297 | 0.9696 |
| 2 | 92 | 0.9541 |
| 3 | 6 | 1.0000 |

### nations

#### Valid Split

- **Total queries**: 199
- **Provable queries**: 21 (10.6%)
- **Non-provable queries**: 178 (89.4%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.4765 | 0.4460 | 0.4801 |
| Hits@1 | 0.2538 | 0.1667 | 0.2640 |
| Hits@3 | 0.6106 | 0.6905 | 0.6011 |
| Hits@10 | 0.9799 | 1.0000 | 0.9775 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 178 | 0.4801 |
| 1 | 21 | 0.4460 |

#### Test Split

- **Total queries**: 201
- **Provable queries**: 16 (8.0%)
- **Non-provable queries**: 185 (92.0%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.4993 | 0.5800 | 0.4924 |
| Hits@1 | 0.2886 | 0.3750 | 0.2811 |
| Hits@3 | 0.6219 | 0.7188 | 0.6135 |
| Hits@10 | 0.9726 | 1.0000 | 0.9703 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 185 | 0.4924 |
| 1 | 16 | 0.5800 |

### wn18rr

#### Valid Split

- **Total queries**: 2824
- **Provable queries**: 1480 (52.4%)
- **Non-provable queries**: 1344 (47.6%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.8356 | 0.9513 | 0.7082 |
| Hits@1 | 0.7842 | 0.9314 | 0.6220 |
| Hits@3 | 0.8695 | 0.9655 | 0.7638 |
| Hits@10 | 0.9205 | 0.9834 | 0.8512 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 1344 | 0.7082 |
| 1 | 1178 | 0.9898 |
| 2 | 100 | 0.8994 |
| 3 | 110 | 0.8467 |
| 4 | 92 | 0.6394 |

#### Test Split

- **Total queries**: 2924
- **Provable queries**: 1500 (51.3%)
- **Non-provable queries**: 1424 (48.7%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.8316 | 0.9511 | 0.7056 |
| Hits@1 | 0.7803 | 0.9317 | 0.6208 |
| Hits@3 | 0.8649 | 0.9650 | 0.7595 |
| Hits@10 | 0.9179 | 0.9823 | 0.8501 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 1424 | 0.7056 |
| 1 | 1199 | 0.9912 |
| 2 | 78 | 0.9242 |
| 3 | 134 | 0.8415 |
| 4 | 89 | 0.5993 |

### umls

#### Valid Split

- **Total queries**: 1300
- **Provable queries**: 1089 (83.8%)
- **Non-provable queries**: 211 (16.2%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.3823 | 0.3773 | 0.4079 |
| Hits@1 | 0.1912 | 0.1758 | 0.2701 |
| Hits@3 | 0.4592 | 0.4564 | 0.4739 |
| Hits@10 | 0.8265 | 0.8567 | 0.6706 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 211 | 0.4079 |
| 1 | 66 | 0.3078 |
| 2 | 358 | 0.3414 |
| 3 | 210 | 0.3265 |
| 4 | 30 | 0.3458 |
| 5 | 385 | 0.4608 |
| 6 | 3 | 0.2880 |
| 7 | 12 | 0.2376 |
| 8 | 4 | 0.2134 |
| 9 | 1 | 0.6667 |
| 10 | 20 | 0.3369 |

#### Test Split

- **Total queries**: 3245
- **Provable queries**: 2704 (83.3%)
- **Non-provable queries**: 541 (16.7%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.5119 | 0.5233 | 0.4548 |
| Hits@1 | 0.3143 | 0.3127 | 0.3226 |
| Hits@3 | 0.6357 | 0.6607 | 0.5111 |
| Hits@10 | 0.9111 | 0.9501 | 0.7163 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 541 | 0.4548 |
| 1 | 163 | 0.4284 |
| 2 | 866 | 0.4877 |
| 3 | 503 | 0.4573 |
| 4 | 90 | 0.4943 |
| 5 | 988 | 0.6091 |
| 6 | 4 | 0.4554 |
| 7 | 19 | 0.4408 |
| 8 | 15 | 0.3771 |
| 9 | 1 | 0.7500 |
| 10 | 55 | 0.5422 |

### pharmkg_full

#### Valid Split

- **Total queries**: 46918
- **Provable queries**: 9945 (21.2%)
- **Non-provable queries**: 36973 (78.8%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.4065 | 0.3960 | 0.4094 |
| Hits@1 | 0.2487 | 0.2398 | 0.2511 |
| Hits@3 | 0.4689 | 0.4541 | 0.4729 |
| Hits@10 | 0.7541 | 0.7383 | 0.7584 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 36973 | 0.4094 |
| 1 | 9916 | 0.3955 |
| 2 | 26 | 0.5343 |
| 4 | 1 | 1.0000 |
| 7 | 2 | 0.7500 |

#### Test Split

- **Total queries**: 47408
- **Provable queries**: 9950 (21.0%)
- **Non-provable queries**: 37458 (79.0%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.4080 | 0.3976 | 0.4107 |
| Hits@1 | 0.2505 | 0.2422 | 0.2527 |
| Hits@3 | 0.4715 | 0.4556 | 0.4757 |
| Hits@10 | 0.7537 | 0.7429 | 0.7566 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 37458 | 0.4107 |
| 1 | 9931 | 0.3973 |
| 2 | 16 | 0.5842 |
| 5 | 2 | 0.2647 |
| 7 | 1 | 1.0000 |

### fb15k237

#### Valid Split

- **Total queries**: 17526
- **Provable queries**: 6964 (39.7%)
- **Non-provable queries**: 10562 (60.3%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.6498 | 0.6897 | 0.6236 |
| Hits@1 | 0.5328 | 0.5732 | 0.5062 |
| Hits@3 | 0.7134 | 0.7611 | 0.6820 |
| Hits@10 | 0.8903 | 0.9194 | 0.8711 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 10562 | 0.6236 |
| 2 | 5889 | 0.7033 |
| 3 | 261 | 0.5961 |
| 4 | 814 | 0.6214 |

#### Test Split

- **Total queries**: 20438
- **Provable queries**: 8139 (39.8%)
- **Non-provable queries**: 12299 (60.2%)

**Metrics Comparison:**

| Metric | Overall | Provable | Non-Provable |
|--------|---------|----------|--------------|
| MRR | 0.6456 | 0.6872 | 0.6181 |
| Hits@1 | 0.5268 | 0.5692 | 0.4988 |
| Hits@3 | 0.7088 | 0.7583 | 0.6760 |
| Hits@10 | 0.8925 | 0.9259 | 0.8704 |

**MRR by Proof Depth:**

| Depth | Count | MRR |
|-------|-------|-----|
| Non-provable | 12299 | 0.6181 |
| 2 | 6908 | 0.6983 |
| 3 | 285 | 0.5801 |
| 4 | 946 | 0.6384 |

## Analysis

### Key Observations

- **family/valid**: KGE performs 0.0762 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **family/test**: KGE performs 0.0760 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **nations/valid**: KGE performance is similar on both subsets (delta=-0.0341).
- **nations/test**: KGE performs 0.0876 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **wn18rr/valid**: KGE performs 0.2431 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **wn18rr/test**: KGE performs 0.2455 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **umls/valid**: KGE performance is similar on both subsets (delta=-0.0306).
- **umls/test**: KGE performs 0.0685 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **pharmkg_full/valid**: KGE performance is similar on both subsets (delta=-0.0134).
- **pharmkg_full/test**: KGE performance is similar on both subsets (delta=-0.0131).
- **fb15k237/valid**: KGE performs 0.0661 MRR better on provable queries, suggesting these queries have more learnable patterns.
- **fb15k237/test**: KGE performs 0.0691 MRR better on provable queries, suggesting these queries have more learnable patterns.

### Implications for RL Hybrid Approach

The RL agent can prove queries where `depth >= 1`. The potential improvement from RL
depends on:

1. **Provable query proportion**: Higher proportion means more queries can benefit from RL.
2. **KGE performance gap**: If KGE underperforms on provable queries, RL can complement.
3. **Depth distribution**: Shallow proofs (depth 1-2) are easier to find than deep ones.

For hybrid scoring, consider:
- Weighting RL proofs based on proof depth (shallower = more confident)
- Using KGE as fallback for non-provable queries
- Adjusting `kge_eval_rl_weight` based on dataset characteristics
