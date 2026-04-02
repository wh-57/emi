"""
13_residual_characterization.py — Economic Content of Causal Residual Direction
=================================================================================
2C found that the output-sensitive causal direction is in PC2-PC16 (residual),
NOT in PC1. Exp3=+0.390 >> Exp2=+0.039 >> Exp7 placebo=+0.033.

This script identifies WHAT the residual direction encodes economically.

Method:
  1. Load shift_resid from 2c_shift_decomposition.csv (the causal direction)
  2. Load neuron-characteristic correlations from 2B
  3. Project shift_resid through neuron-char correlations to get
     characteristic-level loadings of the causal direction
  4. Compare: PC1 economic content vs residual economic content
  5. This gives us the economic label for the 'circuit' in 3A

Output: residual_economic_content.csv + summary print for pre-registration memo
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import h5py
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

BASE     = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
DATA_DIR = BASE / 'data'
ACTS_DIR = BASE / 'activations'

SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
             'prc', 'shrout', 'exchcd', 'shrcd']

# ── 1. Load shift decomposition from 2C ──────────────────────────────────────
print("Loading 2C shift decomposition...")
df_shift = pd.read_csv(DATA_DIR / '2c_shift_decomposition.csv')
shift_total  = df_shift['shift_total'].values    # (16,)
shift_pc1    = df_shift['shift_pc1'].values      # (16,)
shift_resid  = df_shift['shift_residual'].values  # (16,) — the causal direction
pc1_loading  = df_shift['pc1_loading'].values    # (16,)

print(f"  Shift total magnitude:   {np.linalg.norm(shift_total):.4f}")
print(f"  PC1 component magnitude: {np.linalg.norm(shift_pc1):.4f}")
print(f"  Residual magnitude:      {np.linalg.norm(shift_resid):.4f}")

# ── 2. Compute neuron-characteristic correlations (cross-sectional) ───────────
print("\nComputing neuron-characteristic correlations...")
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
char_cols = [c for c in panel.columns if c not in SKIP_COLS]

# Zero-variance filter
char_stds = panel[char_cols].std()
valid_chars = [c for c in char_cols if char_stds[c] > 1e-6]
print(f"  Valid characteristics: {len(valid_chars)}/{len(char_cols)}")

panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)

# Sample every 6th month for speed
h5_files = sorted(ACTS_DIR.glob('acts_*.h5'))
sample_files = [f for i, f in enumerate(h5_files) if i % 6 == 0]

neuron_char_corr = np.zeros((16, len(valid_chars)))  # (16 neurons, n_chars)
n_months_used = 0

for f in sample_files:
    yyyymm = int(f.stem.replace('acts_', ''))
    with h5py.File(f, 'r') as hf:
        if 'layer1' not in hf:
            continue
        acts    = hf['layer1'][:]       # (n_stocks, 16)
        permnos = hf['permno'][:]

    month_panel = panel[panel['yyyymm'] == yyyymm][['permno'] + valid_chars]
    df = pd.DataFrame({'permno': permnos.astype(int)}).merge(
        month_panel, on='permno', how='left')
    chars = df[valid_chars].values.astype(np.float32)

    for n in range(16):
        act_n = acts[:, n]
        for ci in range(len(valid_chars)):
            c = chars[:, ci]
            mask = ~np.isnan(c)
            if mask.sum() < 50:
                continue
            r = np.corrcoef(act_n[mask], c[mask])[0, 1]
            if not np.isnan(r):
                neuron_char_corr[n, ci] += r

    n_months_used += 1

neuron_char_corr /= (n_months_used + 1e-8)
print(f"  Used {n_months_used} months for correlations")


# ── 3. Project shift vectors onto characteristic space ───────────────────────
print("\nProjecting shift vectors onto characteristic space...")

# For each direction v in neuron space, the implied characteristic loading is:
# char_loading[c] = sum_n(v[n] * neuron_char_corr[n, c])
# This gives the characteristic-level signature of each direction

def char_projection(direction, neuron_char_corr):
    """Project a 16-dim neuron-space direction onto characteristic space."""
    return direction @ neuron_char_corr  # (n_chars,)

proj_pc1   = char_projection(pc1_loading,   neuron_char_corr)
proj_resid = char_projection(shift_resid,   neuron_char_corr)
proj_total = char_projection(shift_total,   neuron_char_corr)

# ── 4. Report ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("PC1 DIRECTION — Economic Content (representation geometry)")
print(f"{'='*65}")
top10_pc1 = np.argsort(np.abs(proj_pc1))[::-1][:10]
print(f"\n  Top-10 characteristics aligned with PC1:")
for i, ci in enumerate(top10_pc1):
    print(f"    #{i+1:2d}: {valid_chars[ci]:<22} {proj_pc1[ci]:+.4f}")

print(f"\n{'='*65}")
print("RESIDUAL DIRECTION — Economic Content (CAUSAL COMPUTATION direction)")
print(f"{'='*65}")
top10_resid = np.argsort(np.abs(proj_resid))[::-1][:10]
print(f"\n  Top-10 characteristics aligned with residual (causal) direction:")
for i, ci in enumerate(top10_resid):
    print(f"    #{i+1:2d}: {valid_chars[ci]:<22} {proj_resid[ci]:+.4f}")

print(f"\n{'='*65}")
print("COMPARISON: PC1 vs RESIDUAL top characteristics")
print(f"{'='*65}")
pc1_top5   = set(valid_chars[i] for i in top10_pc1[:5])
resid_top5 = set(valid_chars[i] for i in top10_resid[:5])
overlap    = pc1_top5 & resid_top5
print(f"\n  PC1 top-5:     {sorted(pc1_top5)}")
print(f"  Residual top-5:{sorted(resid_top5)}")
print(f"  Overlap:       {sorted(overlap)}")

if len(overlap) <= 1:
    print(f"\n  → LOW OVERLAP: PC1 and residual encode different economic content.")
    print(f"    Representation direction ≠ computation direction economically.")
else:
    print(f"\n  → HIGH OVERLAP: both directions encode similar economic content.")
    print(f"    Dissociation is geometric but not economically distinct.")

# ── 5. Economic label for the causal circuit ──────────────────────────────────
print(f"\n{'='*65}")
print("ECONOMIC LABEL FOR CAUSAL CIRCUIT")
print(f"{'='*65}")

# Classify top characteristics by economic family
VOLATILITY    = {'retvol','idiovol','maxret','stdcf','std_dolvol','std_turn'}
LIQUIDITY     = {'baspread','mvel1','dolvol','ill','zerotrade','turn'}
MOMENTUM      = {'mom12m','mom1m','mom6m','indmom','mom12moffseason'}
PROFITABILITY = {'roaq','gp','roe','roic','cfp','ep','sp','bm'}
DISTRESS      = {'age','nincr','chtx','pchcapx_ia','chinv','agr','lgr'}

top5_resid_chars = [valid_chars[i] for i in top10_resid[:5]]

families_hit = {}
for fam_name, fam_set in [('volatility', VOLATILITY), ('liquidity', LIQUIDITY),
                           ('momentum', MOMENTUM), ('profitability', PROFITABILITY),
                           ('distress', DISTRESS)]:
    hits = [c for c in top5_resid_chars if c in fam_set]
    if hits:
        families_hit[fam_name] = hits

print(f"\n  Top-5 residual chars: {top5_resid_chars}")
print(f"  Economic families hit:")
for fam, chars in families_hit.items():
    print(f"    {fam}: {chars}")

if families_hit:
    primary_family = max(families_hit, key=lambda k: len(families_hit[k]))
    print(f"\n  Primary economic label: '{primary_family.upper()} CIRCUIT'")
    print(f"  This is the name to use in the Phase 3A pre-registration memo.")
else:
    print(f"\n  No standard family match — label as 'UNIDENTIFIED CIRCUIT'")
    print(f"  Document and do not force an interpretation.")

# ── 6. Save ───────────────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    'characteristic':       valid_chars,
    'proj_pc1':             proj_pc1,
    'proj_residual_causal': proj_resid,
    'proj_total_shift':     proj_total,
})
df_out['abs_proj_resid'] = np.abs(proj_resid)
df_out = df_out.sort_values('abs_proj_resid', ascending=False)
df_out.to_csv(DATA_DIR / 'residual_economic_content.csv', index=False)
print(f"\n  Saved residual_economic_content.csv")

print(f"\n{'='*65}")
print("SUMMARY FOR PRE-REGISTRATION MEMO")
print(f"{'='*65}")
print(f"""
  Phase 2 causal finding:
    Exp1 (full patch):     +0.311  [+0.251, +0.363]
    Exp2 (PC1 only):       +0.039  [+0.021, +0.058]
    Exp3 (residual PC2+):  +0.390  [+0.316, +0.458]
    Exp7 (placebo ortho):  +0.033  (CI crosses zero)

  Dissociation confirmed:
    Representation geometry (PC1): explains 84.3% of activation variance
    Computation geometry (residual): drives 125% of causal output shift
    These are DIFFERENT directions in neuron space.

  Causal direction economic label:
    PC1 (representation): {[valid_chars[i] for i in top10_pc1[:3]]}
    Residual (causal):    {[valid_chars[i] for i in top10_resid[:3]]}

  Use these labels in 3A pre-registration memo.
  Predict: ablating the RESIDUAL direction hurts recession Sharpe more
  than expansion Sharpe (ratio > 2). Ablating PC1 should hurt expansion
  more than recession (it is the expansion representation direction).
""")
