"""
15_3b_ablation.py — Phase 3B: Ablation Tests
=============================================
Per Phase 3A pre-registration memo (signed 2026-04-01).
Specifications are LOCKED. Do not modify test design after running.

Three pre-registered ablations:

  H1 (PRIMARY): Ablate Circuit C (residual direction)
    h1_ablated = h1 - (h1 @ unit_C) * unit_C
    Prediction: ΔSharpe_NBER / ΔSharpe_expansion > 2

  H2 (SECONDARY/EXPLORATORY): Ablate Circuit R (PC1)
    h1_ablated = h1 - (h1 @ pc1) * pc1
    Prediction: smaller or more symmetric regime effect than H1

  H3 (PLACEBO): Ablate random direction orthogonal to PC1, same norm as unit_C
    Prediction: ablation ratio < 1.5, CI includes 1.0

H4 (structural break) and H5 (factor pricing) are in separate scripts.

Bootstrap: 1,000 draws of TEST MONTHS (month-level, not stock-level)
  to account for time-series dependence. Per memo specification.

Test period: post-2005 for primary results (post-1987 for secondary).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(2026)
np.random.seed(2026)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BASE      = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR  = BASE / 'activations'
DATA_DIR  = BASE / 'data'
MODEL_DIR = BASE / 'models'

SEED          = 42
N_BOOTSTRAP   = 1_000
PRIMARY_START = 200501   # post-2005 for primary results
FULL_START    = 198701   # full OOS for secondary

# ── GKX MLP ───────────────────────────────────────────────────────────────────
HIDDEN  = [32, 16, 8]
DROPOUT = 0.5

class GKXMLP(nn.Module):
    def __init__(self, n_chars=94):
        super().__init__()
        layers, prev = [], n_chars
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def forward_from_layer1(self, h1):
        x = h1
        for idx in range(7, len(self.net)):
            x = self.net[idx](x)
        return x.squeeze(-1)


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GKX model and panel...")
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
SKIP  = ['permno','yyyymm','ret_adj','vol','siccd','sic2',
         'prc','shrout','exchcd','shrcd']
char_cols = [c for c in panel.columns if c not in SKIP]
panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)

model = GKXMLP(n_chars=len(char_cols)).to(DEVICE)
ckpts = sorted(MODEL_DIR.glob(f'checkpoint_*_seed{SEED}.pt'))
if not ckpts:
    ckpts = sorted(MODEL_DIR.glob('checkpoint_*.pt'))
ckpt = ckpts[-1]
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print(f"  Loaded: {ckpt.name}")


# ── Load ablation directions (locked per memo) ────────────────────────────────
print("Loading ablation directions from 2C output...")
df_shift   = pd.read_csv(DATA_DIR / '2c_shift_decomposition.csv')
shift_resid = df_shift['shift_residual'].values   # (16,) Circuit C
pc1_loading = df_shift['pc1_loading'].values      # (16,) Circuit R

unit_C   = shift_resid / np.linalg.norm(shift_resid)
unit_R   = pc1_loading  / np.linalg.norm(pc1_loading)

# Verify they are orthogonal (as constructed)
cos = float(np.dot(unit_C, unit_R))
print(f"  Cosine(Circuit C, Circuit R) = {cos:.6f}  (should be ~0)")

# Placebo: random direction orthogonal to PC1, same norm as unit_C
rng      = np.random.default_rng(2026)
rand_dir = rng.standard_normal(16)
rand_dir = rand_dir - (rand_dir @ unit_R) * unit_R
rand_dir = rand_dir / np.linalg.norm(rand_dir)   # same norm as unit_C (=1)


# ── Load activations and regime labels ───────────────────────────────────────
print("Loading layer1 activations and regime labels...")
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

# Build month-level dataset: one row per month with activations + returns
print("Building month-level data...")
month_records = []

h5_files = sorted(ACTS_DIR.glob('acts_*.h5'))
for f in h5_files:
    yyyymm = int(f.stem.replace('acts_', ''))
    if yyyymm not in regime_map or pd.isna(regime_map[yyyymm]):
        continue
    nber = int(regime_map[yyyymm])

    with h5py.File(f, 'r') as hf:
        acts    = hf['layer1'][:].astype(np.float32)   # (n_stocks, 16)
        permnos = hf['permno'][:].astype(int)

    # Get returns for this month
    month_panel = panel[panel['yyyymm'] == yyyymm][['permno','ret_adj']].copy()
    df_m = pd.DataFrame({'permno': permnos}).merge(month_panel, on='permno', how='left')
    returns = df_m['ret_adj'].values.astype(np.float32)

    valid = ~np.isnan(returns)
    if valid.sum() < 50:
        continue

    month_records.append({
        'yyyymm':  yyyymm,
        'nber':    nber,
        'acts':    acts[valid],      # (n_valid, 16)
        'returns': returns[valid],   # (n_valid,)
        'permnos': permnos[valid],
    })

print(f"  Months loaded: {len(month_records)}")

# Normalization stats (computed over all months)
all_acts = np.vstack([r['acts'] for r in month_records])
acts_mu  = all_acts.mean(0, keepdims=True)
acts_sig = all_acts.std(0, keepdims=True) + 1e-8
del all_acts


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def predict_ablated(acts_raw, ablation_dir=None):
    """
    acts_raw: (n, 16) raw (un-normalized) layer1 activations
    ablation_dir: (16,) unit vector to ablate, or None for unablated
    Returns: (n,) predicted returns
    """
    acts_n = (acts_raw - acts_mu) / acts_sig   # normalize

    if ablation_dir is not None:
        # Remove component along ablation direction
        proj     = (acts_n @ ablation_dir)[:, None] * ablation_dir[None, :]
        acts_n   = acts_n - proj

    # De-normalize before feeding to model
    acts_orig = acts_n * acts_sig + acts_mu
    acts_t    = torch.tensor(acts_orig, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model.forward_from_layer1(acts_t).cpu().numpy()
    return preds


def monthly_sharpe(returns_by_month):
    """Annualized Sharpe from list of monthly L/S returns."""
    r = np.array(returns_by_month)
    if len(r) < 3 or r.std() < 1e-8:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(12))


def ls_return_for_month(acts_raw, returns, ablation_dir=None):
    """Compute L/S decile return for one month under given ablation."""
    preds = predict_ablated(acts_raw, ablation_dir)
    # Decile sort
    n     = len(preds)
    if n < 20:
        return np.nan
    decile_size = n // 10
    sorted_idx  = np.argsort(preds)
    bottom_idx  = sorted_idx[:decile_size]
    top_idx     = sorted_idx[-decile_size:]
    return float(returns[top_idx].mean() - returns[bottom_idx].mean())


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE BASELINE AND ABLATED SHARPES PER MONTH
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nComputing monthly L/S returns (baseline + 3 ablations)...")
print(f"Primary test period: {PRIMARY_START}+")

results_by_month = {key: {} for key in
                    ['baseline', 'ablate_C', 'ablate_R', 'ablate_placebo']}
ablation_dirs = {
    'baseline':       None,
    'ablate_C':       unit_C,
    'ablate_R':       unit_R,
    'ablate_placebo': rand_dir,
}

for i, rec in enumerate(month_records):
    if i % 50 == 0:
        print(f"  Month {i}/{len(month_records)} ({rec['yyyymm']})...")
    for key, adir in ablation_dirs.items():
        ls = ls_return_for_month(rec['acts'], rec['returns'], adir)
        results_by_month[key][rec['yyyymm']] = {
            'ls_ret': ls, 'nber': rec['nber']}


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE ABLATION RATIOS
# ══════════════════════════════════════════════════════════════════════════════

def ablation_stats(results_baseline, results_ablated, start_yyyymm,
                   label=''):
    """
    Compute ΔSharpe (baseline - ablated) for NBER and expansion months.
    Returns ablation_ratio = ΔSharpe_NBER / ΔSharpe_expansion
    """
    months = sorted([m for m in results_baseline if m >= start_yyyymm])

    nber_base, nber_abl = [], []
    exp_base,  exp_abl  = [], []

    for m in months:
        b = results_baseline[m]
        a = results_ablated[m]
        if np.isnan(b['ls_ret']) or np.isnan(a['ls_ret']):
            continue
        if b['nber'] == 1:
            nber_base.append(b['ls_ret'])
            nber_abl.append(a['ls_ret'])
        else:
            exp_base.append(b['ls_ret'])
            exp_abl.append(a['ls_ret'])

    sharpe_base_nber = monthly_sharpe(nber_base)
    sharpe_abl_nber  = monthly_sharpe(nber_abl)
    sharpe_base_exp  = monthly_sharpe(exp_base)
    sharpe_abl_exp   = monthly_sharpe(exp_abl)

    delta_nber = sharpe_base_nber - sharpe_abl_nber
    delta_exp  = sharpe_base_exp  - sharpe_abl_exp
    ratio      = delta_nber / (abs(delta_exp) + 1e-8) if abs(delta_exp) > 1e-4 else np.nan

    return {
        'label':             label,
        'start':             start_yyyymm,
        'n_nber_months':     len(nber_base),
        'n_exp_months':      len(exp_base),
        'sharpe_base_nber':  round(sharpe_base_nber, 3),
        'sharpe_abl_nber':   round(sharpe_abl_nber,  3),
        'sharpe_base_exp':   round(sharpe_base_exp,  3),
        'sharpe_abl_exp':    round(sharpe_abl_exp,   3),
        'delta_nber':        round(delta_nber, 3),
        'delta_exp':         round(delta_exp,  3),
        'ablation_ratio':    round(ratio, 3) if not np.isnan(ratio) else np.nan,
    }


# ── Month-level bootstrap for CI on ablation ratio ───────────────────────────
def bootstrap_ablation_ratio(results_baseline, results_ablated,
                              start_yyyymm, n_boot=N_BOOTSTRAP, seed=2026):
    rng    = np.random.default_rng(seed)
    months = sorted([m for m in results_baseline if m >= start_yyyymm])

    nber_months = [m for m in months
                   if results_baseline[m]['nber'] == 1
                   and not np.isnan(results_baseline[m]['ls_ret'])
                   and not np.isnan(results_ablated[m]['ls_ret'])]
    exp_months  = [m for m in months
                   if results_baseline[m]['nber'] == 0
                   and not np.isnan(results_baseline[m]['ls_ret'])
                   and not np.isnan(results_ablated[m]['ls_ret'])]

    if len(nber_months) < 3 or len(exp_months) < 3:
        return np.nan, np.nan, np.nan

    boot_ratios = []
    for _ in range(n_boot):
        nb = rng.choice(nber_months, len(nber_months), replace=True)
        eb = rng.choice(exp_months,  len(exp_months),  replace=True)

        nb_base = [results_baseline[m]['ls_ret'] for m in nb]
        nb_abl  = [results_ablated[m]['ls_ret']  for m in nb]
        eb_base = [results_baseline[m]['ls_ret'] for m in eb]
        eb_abl  = [results_ablated[m]['ls_ret']  for m in eb]

        d_nber = monthly_sharpe(nb_base) - monthly_sharpe(nb_abl)
        d_exp  = monthly_sharpe(eb_base) - monthly_sharpe(eb_abl)

        if abs(d_exp) > 1e-4:
            boot_ratios.append(d_nber / abs(d_exp))

    if len(boot_ratios) < 10:
        return np.nan, np.nan, np.nan

    return (float(np.mean(boot_ratios)),
            float(np.percentile(boot_ratios, 2.5)),
            float(np.percentile(boot_ratios, 97.5)))


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL ABLATIONS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("3B ABLATION RESULTS")
print(f"{'='*70}")

ablation_keys = ['ablate_C', 'ablate_R', 'ablate_placebo']
ablation_labels = {
    'ablate_C':       'H1: Circuit C (residual — CAUSAL)',
    'ablate_R':       'H2: Circuit R (PC1 — REPRESENTATION)',
    'ablate_placebo': 'H3: Placebo (random ortho to PC1)',
}

all_stats = []

for key in ablation_keys:
    label = ablation_labels[key]
    print(f"\n── {label} ──────────────────────────────────────")

    for start, period_name in [(PRIMARY_START, 'Primary (post-2005)'),
                                (FULL_START,    'Full OOS (post-1987)')]:
        stats = ablation_stats(results_by_month['baseline'],
                               results_by_month[key],
                               start, label=f"{key}_{period_name}")

        print(f"\n  Period: {period_name}")
        print(f"  NBER months: {stats['n_nber_months']}  "
              f"Expansion months: {stats['n_exp_months']}")
        print(f"  Sharpe baseline/ablated  NBER: "
              f"{stats['sharpe_base_nber']:.3f} → {stats['sharpe_abl_nber']:.3f}  "
              f"(Δ = {stats['delta_nber']:+.3f})")
        print(f"  Sharpe baseline/ablated  Exp:  "
              f"{stats['sharpe_base_exp']:.3f} → {stats['sharpe_abl_exp']:.3f}  "
              f"(Δ = {stats['delta_exp']:+.3f})")
        print(f"  Ablation ratio (ΔNBER / |ΔExp|): {stats['ablation_ratio']:.3f}")

        # Bootstrap CI (primary period only for speed)
        if start == PRIMARY_START:
            print(f"  Computing bootstrap CI ({N_BOOTSTRAP} month-level draws)...")
            mean_r, ci_lo, ci_hi = bootstrap_ablation_ratio(
                results_by_month['baseline'],
                results_by_month[key],
                start)
            print(f"  Bootstrap: mean={mean_r:.3f}  CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
            stats['boot_mean']  = round(mean_r, 3) if not np.isnan(mean_r) else np.nan
            stats['boot_ci_lo'] = round(ci_lo, 3)  if not np.isnan(ci_lo) else np.nan
            stats['boot_ci_hi'] = round(ci_hi, 3)  if not np.isnan(ci_hi) else np.nan

            # Pre-registered threshold check
            if key == 'ablate_C':
                ratio = stats['ablation_ratio']
                if ratio > 3:
                    print(f"  ✓ STRONG PASS: ratio={ratio:.3f} > 3")
                elif ratio > 2:
                    print(f"  ✓ PASS: ratio={ratio:.3f} > 2")
                elif ratio > 1.5:
                    print(f"  ~ BORDERLINE: ratio={ratio:.3f} (target > 2)")
                else:
                    print(f"  ✗ FAIL: ratio={ratio:.3f} < 2")
            elif key == 'ablate_placebo':
                ratio = stats['ablation_ratio']
                if ratio < 1.5:
                    print(f"  ✓ PLACEBO PASS: ratio={ratio:.3f} < 1.5")
                else:
                    print(f"  ✗ PLACEBO FAIL: ratio={ratio:.3f} > 1.5 — check")

        all_stats.append(stats)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SUMMARY — Primary period (post-2005), pre-registered hypotheses")
print(f"{'='*70}")
print(f"\n  {'Ablation':<35} {'ΔSharpe NBER':>14} {'ΔSharpe Exp':>13} {'Ratio':>8}")
print(f"  {'-'*72}")

primary_stats = [s for s in all_stats if 'Primary' in s['label']]
for s in primary_stats:
    key = s['label'].split('_Primary')[0]
    lbl = ablation_labels.get(key, key)[:34]
    print(f"  {lbl:<35} {s['delta_nber']:>+14.3f} {s['delta_exp']:>+13.3f} "
          f"{s['ablation_ratio']:>8.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PHASE 3 CHECKPOINT")
print(f"{'='*70}")

c_stats = next((s for s in primary_stats if 'ablate_C' in s['label']), None)
p_stats = next((s for s in primary_stats if 'ablate_placebo' in s['label']), None)

if c_stats:
    ratio_C = c_stats['ablation_ratio']
    print(f"\n  H1 (Circuit C ablation ratio): {ratio_C:.3f}")
    if ratio_C > 2:
        print(f"  ✓ H1 PASSES: ablation ratio {ratio_C:.3f} > 2")
        print(f"    Recession Sharpe drops {c_stats['delta_nber']:+.3f} when Circuit C ablated")
        print(f"    Expansion Sharpe drops {c_stats['delta_exp']:+.3f} when Circuit C ablated")
    else:
        print(f"  ✗ H1 FAILS: ablation ratio {ratio_C:.3f} < 2")

if p_stats:
    ratio_P = p_stats['ablation_ratio']
    print(f"\n  H3 (Placebo ratio): {ratio_P:.3f}")
    if ratio_P < 1.5:
        print(f"  ✓ H3 PASSES: placebo ratio {ratio_P:.3f} < 1.5")
    else:
        print(f"  ✗ H3 FAILS: placebo ratio {ratio_P:.3f} > 1.5")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(all_stats).to_csv(DATA_DIR / '3b_ablation_results.csv', index=False)
print(f"\n  Saved 3b_ablation_results.csv")
print(f"\nNext: 3C (structural break event study) and 3D (factor pricing)")
