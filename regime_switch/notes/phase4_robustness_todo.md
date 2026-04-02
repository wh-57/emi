# Phase 4 Robustness Checks — Deferred from Phase 2/3
## EMI Project
## Created: 2026-04-01

These diagnostics were flagged during the Phase 2→3
transition but deferred to Phase 4 as they are robustness checks, not
prerequisites for the primary results.

---

## 1. Month-level bootstrap for 2C patching 
Current bootstrap resamples stocks within a fixed set of months.
Month-level bootstrap: resample months (with replacement), recompute
mean recession/expansion vectors per draw, recompute patching effects.
This accounts for time-series dependence in the patching estimates.
Expected outcome: similar to stock bootstrap given the results are
direction-specific (Exp3 >> Exp7 placebo), but necessary for reporting.
Script: add to 11_2c_activation_patching.py as an optional second pass.

## 2. Gradient alignment check 
Compute cosine similarity between shift_resid (the causal direction
from 2C) and the gradient of the model output w.r.t. layer1 activations,
averaged over many expansion-month stocks.
If shift_resid is aligned with the gradient, it confirms the residual
is a high-sensitivity direction and the functional decoupling is
mechanistically grounded.
Expected outcome: moderate alignment (the gradient direction should
partly coincide with shift_resid since both relate to output sensitivity).

## 3. Layer 0 and 2 residual characterization 
Run the same economic content projection (as in script 13) for:
- layer0: compute shift between recession and expansion mean activations,
  decompose into PC1 and residual, project onto input characteristics
- layer2: same for the 8-neuron final hidden layer
This would show whether the functional decoupling persists throughout
the network or is specific to layer1.
Expected outcome: layer2 may show convergence (PC1 and residual align
by the final layer) or persistence (dissociation holds throughout).

## 4. On-manifold check 
Compute Mahalanobis distance of patched activations from the observed
expansion activation distribution. If patched activations are far
off-manifold, the patching results could be dismissed as "moving to
regions the network never sees."
Implementation: fit a multivariate Gaussian to expansion activations,
compute Mahalanobis distance for original vs. patched activations,
report mean and 95th percentile.
Expected outcome: small Mahalanobis distance (the shift is modest
relative to the natural variance of the distribution).

## 5. Episode-specific recession mean patching 
Current 2C uses one global mean_rec vector. Alternative: compute
episode-specific recession centroids for each of the ~8 NBER recessions
and patch using each separately. Report whether the sign and magnitude
of Exp1/Exp3 are stable across recession episodes.
Expected outcome: stable sign (all positive), modest variation in
magnitude across episodes.

## 6. Seed stability of PR and dissociation 
Phase 2E (full seed stability analysis) will formally check whether
the Jaccard of SAE features is stable across seeds. Additionally, check:
- Is PR=1.40 stable across seeds? (refit PCA per seed)
- Does the angle between PC1 and shift_resid remain ~90° across seeds?
This is the seed stability version of the geometric dissociation result.

---

## Priority order for Phase 4
1. Month-level bootstrap (adds credibility to 2C results cited in paper)
2. On-manifold check (preempts most common reviewer attack)
3. Episode-specific patching (robustness of recession mean vector)
4. Gradient alignment (mechanistic grounding, interesting but not essential)
5. Layer 0/2 residual characterization (network-wide result if it holds)
6. Seed stability of dissociation (consolidates with 2E)
