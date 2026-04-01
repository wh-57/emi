"""
12_layer_pca_diagnostic.py — PCA Diagnostic Across All Three Layers
====================================================================
Run BEFORE 2C patching. Required to scope claims correctly.

Computes participation ratio, effective rank, and economic content
for all three layers: layer0 (32 neurons), layer1 (16), layer2 (8).

Also computes residual predictive power diagnostic:
  - Project layer1 activations onto PC2-PC16 (orthogonal to PC1)
  - Regress normalized returns on residual projection
  - If residual has no predictive power → Exp 3 in 2C will be near zero

Scenarios:
  Bottleneck: layer0 high-rank, layer1 rank-1, layer2 expands
  → Layer 1 is an information bottleneck
  Global collapse: all layers low PR
  → Network is fundamentally 1D throughout
  Progressive compression: PR decreases layer-by-layer
  → Gradual distillation to single factor
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

BASE     = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR = BASE / 'activations'
DATA_DIR = BASE / 'data'

LAYER_SIZES = {'layer0': 32, 'layer1': 16, 'layer2': 8}
SKIP_COLS   = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
               'prc', 'shrout', 'exchcd', 'shrcd']

# ── Load regime labels ────────────────────────────────────────────────────────
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

# ── Load activations for all layers ──────────────────────────────────────────
print("Loading activations for all three layers...")
layer_data = {layer: [] for layer in LAYER_SIZES}
regime_list, yyyymm_list, permno_list = [], [], []

h5_files = sorted(ACTS_DIR.glob('acts_*.h5'))
for f in h5_files:
    yyyymm = int(f.stem.replace('acts_', ''))
    if yyyymm not in regime_map or pd.isna(regime_map[yyyymm]):
        continue
    with h5py.File(f, 'r') as hf:
        for layer in LAYER_SIZES:
            if layer in hf:
                layer_data[layer].append(hf[layer][:])
        regime_list.append(np.full(
            layer_data['layer1'][-1].shape[0],
            regime_map[yyyymm], dtype=np.float32))
        yyyymm_list.append(np.full(
            layer_data['layer1'][-1].shape[0],
            yyyymm, dtype=np.int32))
        if 'permno' in hf:
            permno_list.append(hf['permno'][:])

regime_np = np.concatenate(regime_list)
yyyymm_np = np.concatenate(yyyymm_list)
permno_np = np.concatenate(permno_list) if permno_list else None
exp_mask  = regime_np == 0

# Stack and normalize each layer
layer_norm = {}
for layer, data in layer_data.items():
    arr  = np.vstack(data).astype(np.float32)
    mu   = arr.mean(0, keepdims=True)
    sig  = arr.std(0, keepdims=True) + 1e-8
    layer_norm[layer] = (arr - mu) / sig
    print(f"  {layer}: {arr.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PCA ACROSS ALL LAYERS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PCA DIAGNOSTIC — ALL LAYERS")
print(f"{'='*65}")

layer_results = {}
for layer, sz in LAYER_SIZES.items():
    acts = layer_norm[layer]
    n_comp = min(sz, 16)  # cap for memory

    pca = PCA(n_components=n_comp)
    pca.fit(acts[exp_mask][:50_000])

    ev     = pca.explained_variance_
    ev_r   = pca.explained_variance_ratio_
    cum    = np.cumsum(ev_r)
    rank90 = int(np.searchsorted(cum, 0.90)) + 1
    rank95 = int(np.searchsorted(cum, 0.95)) + 1
    pr     = float(ev.sum()**2 / (ev**2).sum())

    layer_results[layer] = {
        'sz': sz, 'rank90': rank90, 'rank95': rank95,
        'pr': pr, 'pc1_var': ev_r[0], 'pca': pca, 'ev_r': ev_r, 'cum': cum
    }

    print(f"\n  {layer} ({sz} neurons):")
    print(f"    PC1 variance:    {ev_r[0]:.3f}")
    print(f"    Rank @ 90%:      {rank90}")
    print(f"    Rank @ 95%:      {rank95}")
    print(f"    Rank @ 99%:      {int(np.searchsorted(cum, 0.99)) + 1}")
    print(f"    Participation ratio: {pr:.2f}")

    # Print top-5 PCs
    for i in range(min(5, n_comp)):
        print(f"    PC{i+1}: {ev_r[i]:.3f}  (cum={cum[i]:.3f})")

# Summary table
print(f"\n  {'Layer':<10} {'Neurons':>8} {'PC1 var':>9} {'Rank95':>8} {'PR':>8}")
print(f"  {'-'*50}")
for layer in ['layer0', 'layer1', 'layer2']:
    r = layer_results[layer]
    print(f"  {layer:<10} {r['sz']:>8} {r['pc1_var']:>9.3f} "
          f"{r['rank95']:>8} {r['pr']:>8.2f}")

# Narrative interpretation
print(f"\n  Interpretation:")
pr0 = layer_results['layer0']['pr']
pr1 = layer_results['layer1']['pr']
pr2 = layer_results['layer2']['pr']

if pr1 < pr0 and pr1 < pr2:
    print(f"  BOTTLENECK: Layer1 is the compression point (PR: {pr0:.1f}→{pr1:.1f}→{pr2:.1f})")
    print(f"  The network compresses to near-rank-1 at layer1, then expands.")
    print(f"  2C patching at layer1 is patching the bottleneck — the key compression stage.")
elif pr0 < 3 and pr1 < 3 and pr2 < 3:
    print(f"  GLOBAL COLLAPSE: All layers are near-rank-1 (PRs: {pr0:.1f}, {pr1:.1f}, {pr2:.1f})")
    print(f"  The network is fundamentally 1D throughout.")
    print(f"  This is the strongest version of the compression finding.")
elif pr0 > pr1 > pr2:
    print(f"  PROGRESSIVE COMPRESSION: PR decreases {pr0:.1f}→{pr1:.1f}→{pr2:.1f}")
    print(f"  The network gradually distills to lower dimensionality.")
else:
    print(f"  COMPLEX STRUCTURE: PRs are {pr0:.1f}, {pr1:.1f}, {pr2:.1f}")
    print(f"  Layer1 compression may be a local feature — check claims carefully.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RESIDUAL PREDICTIVE POWER DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("RESIDUAL PREDICTIVE POWER DIAGNOSTIC")
print("Does PC2-PC16 of layer1 predict returns?")
print(f"If not: Exp 3 in 2C will be near zero by construction.")
print(f"{'='*65}")

# Load panel returns
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet',
                        columns=['permno', 'yyyymm', 'ret_adj'])
panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)
panel = panel.dropna(subset=['ret_adj'])

# Normalized return
panel['ret_norm'] = panel.groupby('yyyymm')['ret_adj'].transform(
    lambda x: x / (x.std() + 1e-8))

# PC1 of layer1
pca1  = layer_results['layer1']['pca']
pc1   = pca1.components_[0]   # (16,)
acts1 = layer_norm['layer1']

# PC1 projection and residual
proj_pc1 = (acts1 @ pc1[:, None]) * pc1[None, :]   # (N, 16)
resid    = acts1 - proj_pc1                          # (N, 16) residual

# Reduce residual to scalar via PC2
pc2      = pca1.components_[1]
resid_pc2 = resid @ pc2    # (N,) — projection onto PC2

# Build dataset: pc1_score, residual_score, return
if permno_np is not None:
    df_acts = pd.DataFrame({
        'permno':   permno_np.astype(int),
        'yyyymm':   yyyymm_np.astype(int),
        'pc1_score': acts1 @ pc1,
        'pc2_score': resid_pc2,
    })
    df_merged = df_acts.merge(panel[['permno', 'yyyymm', 'ret_norm']],
                               on=['permno', 'yyyymm'], how='inner')
    df_merged = df_merged.dropna(subset=['ret_norm'])

    # Monthly FM regression: PC1 vs PC2 slope t-stats
    monthly_slopes_pc1 = []
    monthly_slopes_pc2 = []
    months = sorted(df_merged['yyyymm'].unique())

    for yyyymm in months:
        grp = df_merged[df_merged['yyyymm'] == yyyymm]
        if len(grp) < 50:
            continue
        try:
            s1, *_ = stats.linregress(grp['pc1_score'], grp['ret_norm'])
            s2, *_ = stats.linregress(grp['pc2_score'], grp['ret_norm'])
            monthly_slopes_pc1.append(s1)
            monthly_slopes_pc2.append(s2)
        except Exception:
            pass

    T = len(monthly_slopes_pc1)
    if T > 0:
        pc1_mean = np.mean(monthly_slopes_pc1)
        pc1_t    = pc1_mean / (np.std(monthly_slopes_pc1) / np.sqrt(T) + 1e-8)
        pc2_mean = np.mean(monthly_slopes_pc2)
        pc2_t    = pc2_mean / (np.std(monthly_slopes_pc2) / np.sqrt(T) + 1e-8)

        print(f"\n  Fama-MacBeth slopes (T={T} months):")
        print(f"  {'Predictor':<20} {'Mean slope':>12} {'FM t-stat':>12}")
        print(f"  {'-'*46}")
        print(f"  {'PC1 score':<20} {pc1_mean:>12.4f} {pc1_t:>12.2f}")
        print(f"  {'PC2 score (resid)':<20} {pc2_mean:>12.4f} {pc2_t:>12.2f}")

        if abs(pc1_t) > 2:
            print(f"\n  ✓ PC1 is a significant return predictor (t={pc1_t:.2f})")
        if abs(pc2_t) > 2:
            print(f"  ✓ PC2 residual is ALSO a significant predictor (t={pc2_t:.2f})")
            print(f"    → Exp 3 in 2C may show non-zero effect. Worth watching.")
        else:
            print(f"  PC2 residual is NOT a significant predictor (t={pc2_t:.2f})")
            print(f"  → Exp 3 in 2C likely near zero. This is informative, not a failure.")
else:
    print("  Skipping: permno alignment not available.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

rows = []
for layer in ['layer0', 'layer1', 'layer2']:
    r = layer_results[layer]
    rows.append({
        'layer': layer,
        'n_neurons': r['sz'],
        'pc1_var':  round(r['pc1_var'], 4),
        'rank_90':  r['rank90'],
        'rank_95':  r['rank95'],
        'participation_ratio': round(r['pr'], 2),
    })
df_summary = pd.DataFrame(rows)
df_summary.to_csv(DATA_DIR / 'layer_pca_summary.csv', index=False)
print(f"\n  Saved layer_pca_summary.csv")
print(f"\n{df_summary.to_string(index=False)}")

print(f"\n{'='*65}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*65}")
print(f"\nNext: review narrative interpretation above, then run 2C.")
print(f"The layer PCA scenario determines how broadly to scope 2C claims.")
