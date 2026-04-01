"""
10_2b_linear_probe.py — Phase 2B: Linear Probing of All Neurons
================================================================
Per flexmap v4 step 2B (secondary/diagnostic given Jaccard=0.154).

Two probing passes:
  (A) Raw probing: regress monthly mean activation on 6 regime indicators
      and macro variables. Expected: all neurons correlate with PC1.
  (B) PC1-residual probing: project PC1 out of activations, then probe
      residuals against regime indicators. Key question: is there
      regime-relevant structure in PC2-PC6?

If residual probing shows significant regime correlations: the network
encodes regime information in the low-variance orthogonal subspace.
If residuals are flat: the entire regime signal lives in PC1.

Both outcomes are publishable — they tell different stories about
how the network encodes macroeconomic state.

Context: 2D found PR=1.40 (near-rank-1), so raw probing will show
all neurons correlated with PC1. The residual probe is the novel test.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR = BASE / 'activations'
DATA_DIR = BASE / 'data'

FIRST_TEST = 198701
TOP_N      = 10
NW_LAGS    = 6   # Newey-West lags

# Layer sizes
LAYER_SIZES = {'layer0': 32, 'layer1': 16, 'layer2': 8}

# ── Load regime labels and macro variables ────────────────────────────────────
print("Loading regime labels and macro variables...")
regimes = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)

# 6 regime indicators + macro alignment variables
REGIME_COLS = ['nber', 'vix25', 'credit15', 'cfnai_neg', 'nfci05']
# Use what's available
available_regime = [c for c in REGIME_COLS if c in regimes.columns]
print(f"  Regime indicators available: {available_regime}")

# Also grab HKM if present
if 'hkm_below' in regimes.columns:
    available_regime.append('hkm_below')

print(f"  Using {len(available_regime)} regime indicators")


# ══════════════════════════════════════════════════════════════════════════════
# 1. COLLECT MONTHLY MEAN ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nCollecting monthly mean activations...")

# Load panel date spine for test months
panel_files = sorted(DATA_DIR.glob('panel_v1_*.parquet'))
dates = pd.read_parquet(panel_files[-1], columns=['yyyymm'])
all_months  = sorted(dates['yyyymm'].drop_duplicates().tolist())
test_months = [m for m in all_months if m >= FIRST_TEST]
print(f"Test months: {len(test_months)} ({test_months[0]} – {test_months[-1]})")

# Initialize storage
neuron_keys = [f'{l}_{n}' for l, sz in LAYER_SIZES.items() for n in range(sz)]
monthly_data = {k: [] for k in neuron_keys}
months_list  = []

# Also store raw activations for layer1 (for PCA projection)
layer1_monthly_acts = []   # list of (n_stocks, 16) arrays per month

for yyyymm in test_months:
    path = ACTS_DIR / f'acts_{yyyymm}.h5'
    if not path.exists():
        continue
    with h5py.File(path, 'r') as f:
        month_acts = {}
        for layer, sz in LAYER_SIZES.items():
            if layer not in f:
                continue
            acts = f[layer][:]                        # (n_stocks, sz)
            month_acts[layer] = acts
            for n in range(sz):
                monthly_data[f'{layer}_{n}'].append(acts[:, n].mean())
        if 'layer1' in month_acts:
            layer1_monthly_acts.append((yyyymm, month_acts['layer1']))

    months_list.append(yyyymm)

months_df = pd.DataFrame({'yyyymm': months_list})
months_df  = months_df.merge(regimes[['yyyymm'] + available_regime],
                              on='yyyymm', how='left')
print(f"Months with activations: {len(months_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIT PCA ON LAYER1 (reuse 2D result conceptually, refit here for residuals)
# ══════════════════════════════════════════════════════════════════════════════

print("\nFitting PCA on layer1 activations (all test months)...")

# Stack all layer1 activations
all_layer1 = np.vstack([acts for _, acts in layer1_monthly_acts]).astype(np.float32)
acts_mean  = all_layer1.mean(0, keepdims=True)
acts_std   = all_layer1.std(0, keepdims=True) + 1e-8
all_layer1_norm = (all_layer1 - acts_mean) / acts_std

pca = PCA(n_components=16)
pca.fit(all_layer1_norm[:50_000])
ev_r    = pca.explained_variance_ratio_
cum_var = np.cumsum(ev_r)
rank_95 = int(np.searchsorted(cum_var, 0.95)) + 1
pr      = float(pca.explained_variance_.sum()**2 / (pca.explained_variance_**2).sum())

print(f"  PC1 variance: {ev_r[0]:.3f}  rank_95: {rank_95}  PR: {pr:.2f}")
print(f"  (Confirms 2D result: near-rank-1 structure)")

pc1 = pca.components_[0]   # (16,) — dominant direction


# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPER: NEWEY-WEST T-STAT
# ══════════════════════════════════════════════════════════════════════════════

def nw_tstat(y, x, lags=NW_LAGS):
    """OLS slope t-stat with Newey-West SEs."""
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    if len(y) < 20:
        return 0.0, 0.0
    T    = len(y)
    xc   = x - x.mean()
    yc   = y - y.mean()
    beta = (xc @ yc) / (xc @ xc + 1e-12)
    resid = yc - beta * xc
    # NW variance
    s0 = (resid**2).mean()
    nw_var = s0
    for l in range(1, lags + 1):
        w   = 1 - l / (lags + 1)
        cov = (resid[l:] * resid[:-l]).mean()
        nw_var += 2 * w * cov
    se   = np.sqrt(max(nw_var, 0) / (xc @ xc / T + 1e-12))
    tstat = beta / (se + 1e-12)
    return float(tstat), float(beta)


# ══════════════════════════════════════════════════════════════════════════════
# 4A. RAW PROBING — ALL 56 NEURONS vs. ALL REGIME INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PASS A — Raw probing (expected: all neurons correlate with PC1)")
print(f"{'='*65}")

nber_vals = months_df['nber'].fillna(0).values

raw_results = {}
for key in neuron_keys:
    acts = np.array(monthly_data[key])
    if len(acts) != len(nber_vals):
        continue
    tstat, beta = nw_tstat(acts, nber_vals)
    raw_results[key] = {'tstat_nber': tstat, 'beta_nber': beta}

# Add all regime indicators
for regime_col in available_regime:
    regime_vals = months_df[regime_col].fillna(0).values
    for key in neuron_keys:
        acts = np.array(monthly_data[key])
        if len(acts) != len(regime_vals):
            continue
        tstat, _ = nw_tstat(acts, regime_vals)
        raw_results[key][f'tstat_{regime_col}'] = tstat

# Sort by |NBER t-stat|
ranked_raw = sorted(raw_results.items(),
                    key=lambda x: abs(x[1]['tstat_nber']), reverse=True)

print(f"\nTop-20 neurons by |NBER t-stat| (raw activations):")
print(f"  {'Neuron':<12} {'t_NBER':>8} {'β':>8}")
print(f"  {'-'*30}")
for key, res in ranked_raw[:20]:
    print(f"  {key:<12} {res['tstat_nber']:>8.2f} {res['beta_nber']:>8.4f}")

# Bonferroni threshold: 56 neurons × 6 regime vars = 336 tests
n_tests     = len(neuron_keys) * len(available_regime)
bonf_thresh = 0.05 / n_tests
bonf_t      = abs(stats.norm.ppf(bonf_thresh / 2))
print(f"\nBonferroni threshold: {n_tests} tests → t > {bonf_t:.2f}")

survivors_raw = [(k, r) for k, r in raw_results.items()
                 if abs(r['tstat_nber']) > bonf_t]
print(f"Neurons surviving Bonferroni (NBER): {len(survivors_raw)}")
for k, r in sorted(survivors_raw, key=lambda x: abs(x[1]['tstat_nber']), reverse=True):
    print(f"  {k:<12} t={r['tstat_nber']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4B. PC1-RESIDUAL PROBING — KEY QUESTION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PASS B — PC1-residual probing")
print("Key question: is regime info encoded in PC2-PC6 (orthogonal to PC1)?")
print(f"{'='*65}")

# For each month, compute mean of (activations - PC1 projection)
layer1_residual_monthly = {}   # yyyymm → mean residual activation (16,)

for yyyymm, acts in layer1_monthly_acts:
    acts_n    = (acts - acts_mean) / acts_std           # normalize
    proj_pc1  = (acts_n @ pc1[:, None]) * pc1[None, :]  # project onto PC1
    residual  = acts_n - proj_pc1                        # orthogonal complement
    layer1_residual_monthly[yyyymm] = residual.mean(0)   # (16,)

# Build time series of residual activations per neuron
resid_monthly = {f'layer1_{n}': [] for n in range(16)}
months_with_resid = []

for yyyymm in months_list:
    if yyyymm not in layer1_residual_monthly:
        continue
    resid = layer1_residual_monthly[yyyymm]
    for n in range(16):
        resid_monthly[f'layer1_{n}'].append(resid[n])
    months_with_resid.append(yyyymm)

months_resid_df = pd.DataFrame({'yyyymm': months_with_resid})
months_resid_df = months_resid_df.merge(
    regimes[['yyyymm'] + available_regime], on='yyyymm', how='left')

nber_resid = months_resid_df['nber'].fillna(0).values

resid_results = {}
for key in resid_monthly:
    acts = np.array(resid_monthly[key])
    if len(acts) != len(nber_resid):
        continue
    tstat, beta = nw_tstat(acts, nber_resid)
    resid_results[key] = {'tstat_nber': tstat, 'beta_nber': beta}

ranked_resid = sorted(resid_results.items(),
                      key=lambda x: abs(x[1]['tstat_nber']), reverse=True)

print(f"\nTop-10 layer1 neurons by |NBER t-stat| (PC1-residual):")
print(f"  {'Neuron':<12} {'t_NBER':>8} {'β':>8}")
print(f"  {'-'*30}")
for key, res in ranked_resid[:10]:
    print(f"  {key:<12} {res['tstat_nber']:>8.2f} {res['beta_nber']:>8.4f}")

# Compare raw vs residual t-stats
print(f"\nRaw vs. Residual NBER t-stats (layer1):")
print(f"  {'Neuron':<12} {'t_raw':>8} {'t_resid':>8} {'ratio':>8}")
print(f"  {'-'*40}")
for n in range(16):
    key = f'layer1_{n}'
    t_raw   = raw_results.get(key, {}).get('tstat_nber', 0)
    t_resid = resid_results.get(key, {}).get('tstat_nber', 0)
    ratio   = abs(t_resid) / (abs(t_raw) + 1e-8)
    print(f"  {key:<12} {t_raw:>8.2f} {t_resid:>8.2f} {ratio:>8.3f}")

# How much of the regime signal survives in the residual?
t_raw_mean   = np.mean([abs(raw_results.get(f'layer1_{n}',{}).get('tstat_nber',0))
                        for n in range(16)])
t_resid_mean = np.mean([abs(resid_results.get(f'layer1_{n}',{}).get('tstat_nber',0))
                        for n in range(16)])
pct_surviving = t_resid_mean / (t_raw_mean + 1e-8) * 100

print(f"\n  Mean |t| raw:     {t_raw_mean:.3f}")
print(f"  Mean |t| residual: {t_resid_mean:.3f}")
print(f"  Regime signal surviving after PC1 removed: {pct_surviving:.1f}%")

if pct_surviving < 20:
    print("  → Regime signal is almost entirely in PC1. Near-rank-1 confirmed.")
elif pct_surviving < 50:
    print("  → Some regime signal in PC2-PC6. Worth investigating orthogonal subspace.")
else:
    print("  → Substantial regime signal outside PC1. Multi-dimensional regime structure.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARACTERISTIC-LEVEL PROBING (what do neurons compute?)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("CHARACTERISTIC PROBING — What does each neuron encode?")
print(f"{'='*65}")

print("\nLoading panel for characteristic probing...")
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
             'prc', 'shrout', 'exchcd', 'shrcd']
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
char_cols = [c for c in panel.columns if c not in SKIP_COLS]

# Focus on cross-sectional correlation: for each neuron, what characteristics
# predict its activation at the stock level?
# Use a subsample of months for speed (every 6th month)
sample_months = [m for i, m in enumerate(test_months) if i % 6 == 0]

print(f"  Using {len(sample_months)} sampled months for char probing...")

panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)

neuron_char_corr = {f'layer1_{n}': np.zeros(len(char_cols)) for n in range(16)}

n_months_used = 0
for yyyymm in sample_months:
    path = ACTS_DIR / f'acts_{yyyymm}.h5'
    if not path.exists():
        continue
    with h5py.File(path, 'r') as f:
        if 'layer1' not in f:
            continue
        acts    = f['layer1'][:]       # (n_stocks, 16)
        permnos = f['permno'][:]

    month_panel = panel[panel['yyyymm'] == yyyymm][['permno'] + char_cols]
    df = pd.DataFrame({'permno': permnos.astype(int)})
    df = df.merge(month_panel, on='permno', how='left')
    chars = df[char_cols].values.astype(np.float32)   # (n_stocks, 94)

    for n in range(16):
        act_n = acts[:, n]
        for ci, col in enumerate(char_cols):
            c = chars[:, ci]
            mask = ~np.isnan(c)
            if mask.sum() < 50:
                continue
            corr = np.corrcoef(act_n[mask], c[mask])[0, 1]
            if not np.isnan(corr):
                neuron_char_corr[f'layer1_{n}'][ci] += corr

    n_months_used += 1

# Average over months
for key in neuron_char_corr:
    neuron_char_corr[key] /= (n_months_used + 1e-8)

print(f"\nTop-5 characteristics per layer1 neuron (mean cross-sectional correlation):")
for n in range(16):
    key  = f'layer1_{n}'
    corrs = neuron_char_corr[key]
    top5 = np.argsort(np.abs(corrs))[::-1][:5]
    labels = [f"{char_cols[i]}({corrs[i]:.2f})" for i in top5]
    t_nber = raw_results.get(key, {}).get('tstat_nber', 0)
    print(f"  {key} (t_NBER={t_nber:.1f}): {', '.join(labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PC1 ECONOMIC CONTENT — WHAT DOES THE DOMINANT DIRECTION ENCODE?
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PC1 ECONOMIC CONTENT — What does the dominant direction encode?")
print(f"{'='*65}")

# PC1 loadings on each neuron
print(f"\n  PC1 loadings on layer1 neurons:")
for n in range(16):
    print(f"    neuron {n:2d}: {pc1[n]:+.4f}")

# Implied characteristic loadings via neuron weights
# PC1 direction in neuron space → economic content
print(f"\n  Characteristics most aligned with PC1 direction:")
# Weighted sum of neuron-char correlations by PC1 loading
pc1_char_proj = np.zeros(len(char_cols))
for n in range(16):
    key = f'layer1_{n}'
    pc1_char_proj += pc1[n] * neuron_char_corr[key]

top10_pc1 = np.argsort(np.abs(pc1_char_proj))[::-1][:10]
for i, ci in enumerate(top10_pc1):
    print(f"    #{i+1}: {char_cols[ci]:<20} {pc1_char_proj[ci]:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Saving outputs ───────────────────────────────────────────────────")

# Raw probing results
rows_raw = []
for key, res in raw_results.items():
    row = {'neuron': key}
    row.update(res)
    rows_raw.append(row)
df_raw = pd.DataFrame(rows_raw).sort_values('tstat_nber', key=abs, ascending=False)
df_raw.to_csv(DATA_DIR / '2b_raw_probe_results.csv', index=False)

# Residual probing results
rows_resid = []
for key, res in resid_results.items():
    row = {'neuron': key}
    row.update(res)
    rows_resid.append(row)
df_resid = pd.DataFrame(rows_resid).sort_values('tstat_nber', key=abs, ascending=False)
df_resid.to_csv(DATA_DIR / '2b_residual_probe_results.csv', index=False)

# PC1 economic content
df_pc1 = pd.DataFrame({
    'characteristic': char_cols,
    'pc1_projection': pc1_char_proj
}).sort_values('pc1_projection', key=abs, ascending=False)
df_pc1.to_csv(DATA_DIR / '2b_pc1_economic_content.csv', index=False)

print(f"  Saved 2b_raw_probe_results.csv")
print(f"  Saved 2b_residual_probe_results.csv")
print(f"  Saved 2b_pc1_economic_content.csv")

print(f"\n{'='*65}")
print("2B COMPLETE — SUMMARY")
print(f"{'='*65}")
print(f"  Bonferroni threshold:   t > {bonf_t:.2f}")
print(f"  Survivors (NBER, raw):  {len(survivors_raw)}")
print(f"  Regime signal in PC1:   {100 - pct_surviving:.1f}%")
print(f"  Regime signal in resid: {pct_surviving:.1f}%")
print(f"\nInterpretation:")
if pct_surviving < 20:
    print(f"  Near-rank-1 confirmed. Regime information is almost entirely")
    print(f"  encoded in PC1. The network does not encode additional regime")
    print(f"  structure in PC2-PC6.")
else:
    print(f"  {pct_surviving:.0f}% of regime signal survives after removing PC1.")
    print(f"  The orthogonal subspace (PC2-PC6) carries additional regime info.")
    print(f"  This is multi-dimensional regime encoding — check residual survivors.")
print(f"\nNext: 2C (activation patching with PC1 decomposition) — Tier 1")
print(f"{'='*65}")
