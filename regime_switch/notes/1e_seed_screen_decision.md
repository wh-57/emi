# 1E Seed Stability Screen — Decision Record

**Date:** 2026-03-31  
**Script:** 07b_seed_screen.py  
**Output files:** seed_screen_jaccard.csv, seed_screen_results.csv

## Jaccard Results

| Seed pair   | Jaccard |
|-------------|---------|
| 42 vs 123   | 0.111   |
| 42 vs 456   | 0.176   |
| 123 vs 456  | 0.176   |
| **Mean**    | **0.154** |

## Decision

Mean Jaccard = 0.154 < 0.20 threshold.  
**Primary unit of analysis: SAE features (Phase 2D)**  
**Neuron-level analysis (2B/2C): secondary/diagnostic only**

## Notable stable neurons

`layer2_4` appears in top-10 for seeds 42 (rank 1, t=−7.40) and 456 (rank 1, t=−5.05).  
`layer0_5` appears in top-10 for all three seeds.  
These are the most seed-stable individual neurons despite low overall Jaccard.

## Interpretation

Low Jaccard is consistent with polysemantic neurons — the network's representational
geometry varies across random initializations, which motivates SAE decomposition as
the more stable unit of analysis. This is the expected failure mode described in
flexmap v2 and does not indicate a problem with the replication.