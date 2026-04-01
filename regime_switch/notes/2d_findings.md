# Phase 2D Findings — 2026-03-31

## Script
`regime_switch/src/09_2d_sae_real.py` (v3 FINAL)

## Key Results

### PCA Rank Diagnostic
- PC1: 84.3% of variance
- PC2: 3.8%, PC3: 3.2%
- Effective rank @ 90%: 3
- Effective rank @ 95%: 6
- Effective rank @ 99%: 12
- Participation ratio: 1.40 (PR≈1 = rank-1; PR≈16 = full rank)
- Subperiod stability: rank_95 spread = 2 (1987-99: 6, 2000-09: 6, 2010-23: 8) → STABLE

### Interpretation
**Case B outcome.** The GKX-style MLP compresses 94 characteristics into a
near-rank-1 latent space at layer 1. This is the headline empirical finding.

### SAE Results
- Dictionary size: 32 (adaptive, from rank_95=6)
- Lambda: 0.6
- Dead features: 5/32 (16%)
- Mean active per input: 14.1
- Alive features: 27/32
- **Crisis features (strict Q1/Q3 definition): 0**

### Top Recession-Elevated Features
| Feature | Ratio | Top characteristics |
|---------|-------|---------------------|
| 6  | 1.98x | age, roaq, nincr, chtx |
| 25 | 1.62x | roaq, retvol, idiovol, baspread, std_turn |
| 27 | 1.54x | baspread, retvol, idiovol, indmom, cash |
| 28 | 1.51x | age, mom12m, sp, roaq, bm |
| 23 | 1.47x | idiovol, retvol, maxret, baspread, std_dolvol |

Monosemanticity scores: 0.037-0.051 (low, consistent with PR=1.40)

### Economic Interpretation
The network does not switch circuits in recessions. It amplifies the same
dominant direction with higher intensity. This is **regime-dependent amplification**
not circuit-switching. Consistent with:
- ICAPM: risk price changes, not the factor itself
- Intermediary asset pricing: funding constraints → amplified vol/illiq pricing
- Limits to arbitrage: vol/illiq more pricing-relevant when arbitrageurs constrained

Feature 6 loads on (age, roaq, nincr, chtx) — non-mechanical, non-trivially
recession-elevated characteristics. The volatility/illiquidity cluster is
economically meaningful, not just a mechanical reflection.

### Stability
Filtered cosine stability: **pending** — stability reruns were still running
at session end. Update when that completes.

## What This Changes
1. Research question reframed: from circuit-switching to compression structure
2. 2B probing needs PC1-residual pass (not just raw probing)
3. 2C patching needs 3-experiment decomposition (full, PC1-only, residual)
4. Paper framing: PR=1.40 is the headline result, not a diagnostic
5. PR=1.40 is an independent publishable contribution even if patching fails

## Next Steps
1. Run `10_2b_linear_probe.py` (linear probing with PC1-residual pass)
2. Write and run 2C patching script (3-experiment design)
3. Check Miola (2026) thesis (cited by DeepSeek as closest adjacent work)
