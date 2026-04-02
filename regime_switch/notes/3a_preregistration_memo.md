# Phase 3A Pre-Registration Memo
## EMI Project — Representation vs. Computation in ML Asset Pricing Networks
## Date: [FILL IN — must be before running any Phase 3 scripts]
## Author: William Huang

---

## Purpose

This memo pre-registers the circuit definitions, economic labels, and exact
hypotheses for Phase 3 BEFORE looking at any Phase 3 results. Per flexmap v5,
this is required for credibility. Do not modify after running any Phase 3 script.

---

## Phase 2 Empirical Record (inputs to this memo)

### Replication baseline (Phase 1, complete)
- OOS R² = 0.5853%, L/S Sharpe = 1.609, FM t = 13.60, FF5+Mom alpha t = 8.30

### Compression structure (script 12, complete)
- Progressive compression: layer0 PR=4.08, layer1 PR=1.40, layer2 PR=1.28
- Layer1 PC1 explains 84.3% of activation variance
- Rank stable across subperiods (spread=2)

### 2C Activation patching (script 11 v2, complete)
- Exp 1 full shift:     +0.311  CI [+0.251, +0.363]
- Exp 2 PC1-only:       +0.039  CI [+0.021, +0.058]
- Exp 3 residual PC2+:  +0.390  CI [+0.316, +0.458]
- Exp 7 placebo ortho:  +0.033  (CI crosses zero)
- Effect measure: mean(pred_patched − pred_orig) / std(pred_orig)

### Pre-memo diagnostics (script 14, complete)
- Angle between PC1 and causal residual: 90.0° (cosine = 0.000)
- Residual FM t-stat (expansion months):  −0.25 (not significant)
- Residual FM t-stat (recession months):  +3.86 (significant)
- PC1 FM t-stat (expansion):              −1.17
- PC1 FM t-stat (recession):              +2.18

### Economic content (script 13, complete)
- PC1 top characteristics: idiovol, retvol, baspread, maxret, std_dolvol, roaq, mvel1
- Residual top characteristics: baspread, idiovol, retvol, std_dolvol, mvel1
- Additional residual-specific loadings: rsup (16.49x ratio), lgr (42.03x ratio)
- Overlap: 4 of top-5 characteristics are shared
- Magnitude: residual loadings are ~5x smaller than PC1 loadings on shared chars

---

## Circuit Definitions

### Circuit R — Representation Direction (PC1)
**Definition:** The first principal component of expansion-month layer-1
activations (pc1_loading in 2c_shift_decomposition.csv).

**Geometric properties:**
- Explains 84.3% of layer-1 activation variance
- Cosine similarity with causal residual: 0.000 (perfectly orthogonal)

**Economic content:**
- Dominant loadings: idiovol (+0.34), retvol (+0.34), baspread (+0.33),
  maxret (+0.30), std_dolvol (+0.27), roaq (−0.27), mvel1 (−0.23)
- Economic label: baseline volatility/illiquidity/distress ranking direction

**Causal role:** NOT the primary causal output driver.
- Patching PC1-only: Exp 2 = +0.039 (small)
- FM t-stat pooled: −0.47 (not significant)
- FM t-stat recession: +2.18 (secondary signal)

**Interpretation:** Circuit R encodes cross-sectional levels of
volatility/illiquidity characteristics. It is the network's dominant
representational direction but a secondary computational direction.

---

### Circuit C — Computation Direction (Residual)
**Definition:** The direction in layer-1 activation space equal to the
recession-minus-expansion mean activation shift after removing the PC1
component (shift_residual in 2c_shift_decomposition.csv).

**Geometric properties:**
- Explains ~16% of activation variance (by construction, orthogonal to PC1)
- Angle from PC1: 90.0°
- Decomposed across PCs: PC3−PC8 collectively (no single PC dominates)

**Economic content:**
- Shared with PC1: baspread (+0.060), idiovol (+0.060), retvol (+0.059),
  std_dolvol (+0.058), mvel1 (−0.055)
- Residual-specific (ratio >> PC1): rsup (16.49x), lgr (42.03x),
  quick (1.83x), salerec (1.82x)
- Economic label: regime-sensitive volatility/illiquidity computation
  direction, with additional loading on revenue surprise and growth

**Causal role:** PRIMARY causal output driver.
- Patching residual only: Exp 3 = +0.390 (dominant)
- vs. placebo same norm: Exp 7 = +0.033 (direction-specific, not magnitude)
- FM t-stat expansion: −0.25 (not significant — functionally silent)
- FM t-stat recession: +3.86 (significant — functionally active)

**Interpretation:** Circuit C encodes the regime-dependent modulation
of the volatility/illiquidity signal. It is geometrically orthogonal to
Circuit R (90°), economically similar in top characteristics but
distinct in secondary loadings, and functionally active only in
recessions. The network uses Circuit C to adjust predictions for
macroeconomic conditions while using Circuit R for baseline cross-
sectional ranking.

---

## Core Finding Statement

The GKX-style MLP dissociates its representation geometry from its
computation geometry within the same economic family:

- Circuit R (PC1) packs 84.3% of activation variance into the
  volatility/illiquidity cluster. It is the dominant representational
  direction but a weak computational driver (Exp 2 = +0.039).

- Circuit C (residual) is geometrically orthogonal (90°) and carries
  far less activation variance, yet drives the dominant causal output
  shift (Exp 3 = +0.390). It predicts returns strongly in recessions
  (FM t = 3.86) but not in expansions (FM t = −0.25).

The network does not switch economic factors in recessions. It maintains
the same economic inputs (volatility/illiquidity) but routes regime-
dependent computation through a geometrically separate, low-variance
orthogonal direction.

---

## Pre-Registered Hypotheses for Phase 3

### H1 — Ablation asymmetry: Circuit C (PRIMARY)
**Hypothesis:** Ablating Circuit C from layer-1 activations reduces
recession-month L/S Sharpe more than expansion-month Sharpe.

**Exact test:**
```
h1_ablated = h1 - (h1 @ unit_C) * unit_C
where unit_C = shift_residual / ||shift_residual||
```
Ablation ratio = ΔSharpe_NBER / ΔSharpe_expansion

**Pre-registered threshold:** ratio > 2 = evidence; ratio > 3 = strong evidence
**Prediction direction:** POSITIVE (recession Sharpe drops more)
**Motivation:** Exp 3 patching (+0.390) >> Exp 2 (+0.039); FM t=3.86 in
recession vs −0.25 in expansion. The causal and predictive evidence
both point to regime-specific use of Circuit C.

---

### H2 — Ablation: Circuit R (SECONDARY / EXPLORATORY)
**Hypothesis:** Ablating Circuit R (PC1) has a smaller or more symmetric
regime effect than ablating Circuit C.

**Exact test:**
```
h1_ablated = h1 - (h1 @ pc1) * pc1
```
**Pre-registered threshold:** ablation ratio < H1 ratio (Circuit R effect
is weaker than Circuit C effect)
**Note:** PC1 FM t=2.18 in recession suggests some regime sensitivity.
Do NOT pre-register a direction for this test — report exploratory.

---

### H3 — Placebo check
**Hypothesis:** Ablating a random direction orthogonal to PC1 (same norm
as Circuit C) has no regime-differential effect.

Ablation ratio for placebo should be < 1.5 and CI should include 1.0.

---

### H4 — Structural break (corroborating, low power)
**Hypothesis:** Circuit C activation amplitude rises before NBER recession
onset. Track monthly mean projection onto unit_C. Compare months [−3, 0]
relative to NBER peak vs expansion mean. Bootstrap 95% CI across ~8 events.
Report as corroborating evidence only — do not overweight given power constraint.

---

### H5 — Factor pricing
**Hypothesis:** A long-short portfolio sorted on Circuit C activation score
has significant alpha in recession months (FM t > 2) but not in expansion
months (FM t < 1). Motivation: FM t=3.86 in recession (script 14).

---

## What Would Change the Interpretation

| Outcome | Interpretation | Venue |
|---------|---------------|-------|
| H1 ratio > 3, H5 passes | Full dissociation + causal + economic | JF possible |
| H1 ratio > 2, H5 mixed | Causal dissociation confirmed | RFS |
| H1 ratio < 1.5 | Circuit C not causally necessary for regime performance | RFS/MS (weaker claim) |
| H1 and H3 both show ratio > 2 | Ablation is magnitude-driven, not direction-specific | Reframe |

---

## Phase 3 Script Specifications (locked)

**Ablation method — Circuit C:**
```python
unit_C = shift_residual / np.linalg.norm(shift_residual)
h1_ablated = h1 - (h1 @ unit_C[:, None]) * unit_C[None, :]
```

**Ablation method — Circuit R (PC1):**
```python
h1_ablated = h1 - (h1 @ pc1[:, None]) * pc1[None, :]
```

**Performance metric:** L/S decile portfolio on predicted returns.
Monthly Sharpe = mean_monthly / std_monthly * sqrt(12).

**Regime split:** Primary = NBER (nber column in monthly_regimes.csv).
Secondary robustness = VIX>25, credit>1.5%.

**Test period:** Post-2005 months only for primary results.
Full OOS (post-1987) for secondary results.

**Bootstrap:** 1,000 draws of test months (with replacement) for CI on
ablation ratio. This is month-level bootstrap, not stock-level, to
account for time-series dependence.

---

## Phase 4 Robustness Checks (deferred, not pre-registered)

The following were flagged but deferred to Phase 4:
- Month-level bootstrap for 2C patching effects 
- Gradient alignment check : cosine similarity between
  shift_resid and gradient of loss w.r.t. layer1 activations
- Layer 0 and 2 residual characterization
- On-manifold check: Mahalanobis distance of patched activations 
- Episode-specific recession mean patching

These are documented in notes/phase4_robustness_todo.md.

---

## HARK

Phase 3 results were not looked at before writing this memo.
The circuit definitions, hypotheses, and exact test specifications above
are final and will not be modified after Phase 3 scripts are run.

William Huang  4/1/26

---

