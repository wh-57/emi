"""
09_2d_sae_real.py — Phase 2D: SAE on Real Layer-1 Activations
==============================================================
Per flexmap v2 step 2D (SAE-first path, confirmed by 1E Jaccard=0.154):
  - Load layer1 (16-neuron) activations from HDF5 files
  - Train SAE (16→128→16, 8x overcomplete) on expansion months only
  - Tune lambda for 5-10 active features per input
  - Label each SAE feature by top-5 characteristics
  - Apply to recession months: count dormant-in-expansion, active-in-recession
  - Stability: retrain 5x, compute cross-run cosine similarity
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(2026)
np.random.seed(2026)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR  = BASE / 'activations'
DATA_DIR  = BASE / 'data'
OUT_DIR   = BASE / 'data'

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_KEY     = 'layer1'       # 16-neuron layer
N_IN         = 16
N_FEAT       = 128            # 8x overcomplete
SKIP_COLS    = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
                'prc', 'shrout', 'exchcd', 'shrcd']
N_STABILITY  = 5              # SAE reruns for stability check
LAM_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ACTIVATIONS + REGIME LABELS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading regime labels...")
regimes = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

print("Loading layer1 activations from HDF5 files...")
acts_list    = []
regime_list  = []
yyyymm_list  = []
permno_list  = []

h5_files = sorted(ACTS_DIR.glob('acts_*.h5'))
print(f"Found {len(h5_files)} HDF5 files")

for f in h5_files:
    yyyymm = int(f.stem.replace('acts_', ''))
    if yyyymm not in regime_map:
        continue
    nber = regime_map[yyyymm]
    if pd.isna(nber):
        continue

    with h5py.File(f, 'r') as hf:
        acts    = hf[HDF5_KEY][:]           # (n_stocks, 16)
        permnos = hf['permno'][:]           # (n_stocks,)

    acts_list.append(acts)
    regime_list.append(np.full(len(acts), nber, dtype=np.float32))
    yyyymm_list.append(np.full(len(acts), yyyymm, dtype=np.int32))
    permno_list.append(permnos)

acts_np    = np.vstack(acts_list).astype(np.float32)    # (N_total, 16)
regime_np  = np.concatenate(regime_list)
yyyymm_np  = np.concatenate(yyyymm_list)
permno_np  = np.concatenate(permno_list)

n_rec = int((regime_np == 1).sum())
n_exp = int((regime_np == 0).sum())
print(f"Total obs: {len(acts_np):,}  |  Expansion: {n_exp:,}  |  Recession: {n_rec:,}")

# ── Normalize ─────────────────────────────────────────────────────────────────
acts_mean = acts_np.mean(axis=0, keepdims=True)
acts_std  = acts_np.std(axis=0, keepdims=True) + 1e-8
acts_norm = (acts_np - acts_mean) / acts_std
print(f"Activations normalized. Mean abs: {np.abs(acts_norm).mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SAE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SAE(nn.Module):
    def __init__(self, n_in=N_IN, n_feat=N_FEAT):
        super().__init__()
        self.enc = nn.Linear(n_in, n_feat)
        self.dec = nn.Linear(n_feat, n_in)
        nn.init.orthogonal_(self.enc.weight)
        nn.init.orthogonal_(self.dec.weight)

    def forward(self, x):
        z = torch.relu(self.enc(x))
        return self.dec(z), z


def train_sae(X_train, lam, epochs=50, seed=2026):
    torch.manual_seed(seed)
    sae = SAE().to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    X_t = torch.tensor(X_train, dtype=torch.float32)

    for epoch in range(epochs):
        idx = torch.randperm(len(X_t))
        for start in range(0, len(X_t), 256):
            xb = X_t[idx[start:start+256]].to(DEVICE)
            opt.zero_grad()
            recon, z = sae(xb)
            (nn.MSELoss()(recon, xb) + lam * z.abs().mean()).backward()
            opt.step()

    sae.eval()
    return sae


def get_encoded(sae, X):
    sae.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_t), 2048):
            _, z = sae(X_t[start:start+2048].to(DEVICE))
            chunks.append(z.cpu().numpy())
    return np.vstack(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LAMBDA TUNING
# ══════════════════════════════════════════════════════════════════════════════

exp_mask  = regime_np == 0
X_exp     = acts_norm[exp_mask]

print(f"\n── Lambda tuning (target: 5–10 active features per input) ──────────")
best_lam  = None
best_diff = np.inf

for lam in LAM_GRID:
    # Quick tune: 20 epochs on 20k subsample
    n_sub  = min(20_000, len(X_exp))
    idx    = np.random.choice(len(X_exp), n_sub, replace=False)
    sae_t  = train_sae(X_exp[idx], lam=lam, epochs=20, seed=2026)
    z      = get_encoded(sae_t, X_exp[idx])
    n_dead = (z.mean(0) < 1e-6).sum()
    mean_active = (z > 1e-6).sum(axis=1).mean()
    print(f"  lam={lam:.3f}  |  mean active: {mean_active:.1f}/128  |  dead: {n_dead}/128")
    diff = abs(mean_active - 7.5)   # target midpoint = 7.5
    if diff < best_diff:
        best_diff = diff
        best_lam  = lam

print(f"\n  → Selected lambda: {best_lam}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FULL SAE TRAINING ON EXPANSION MONTHS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Training full SAE (lam={best_lam}, 50 epochs) ────────────────────")
sae_main = train_sae(X_exp, lam=best_lam, epochs=50, seed=2026)
z_exp    = get_encoded(sae_main, X_exp)

n_dead_final = (z_exp.mean(0) < 1e-6).sum()
mean_active_final = (z_exp > 1e-6).sum(axis=1).mean()
print(f"  Dead features: {n_dead_final}/128")
print(f"  Mean active per input: {mean_active_final:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. APPLY TO ALL MONTHS + RECESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Applying SAE to all months ───────────────────────────────────────")
z_all = get_encoded(sae_main, acts_norm)   # (N_total, 128)

rec_mask = regime_np == 1
z_rec    = z_all[rec_mask]
z_exp_all = z_all[exp_mask]

# Per-feature mean activation in each regime
mean_exp = z_exp_all.mean(axis=0)    # (128,)
mean_rec = z_rec.mean(axis=0)        # (128,)

# Dormant-in-expansion, active-in-recession
DORMANT_THRESH = 0.05   # < 5th percentile of expansion mean = "dormant"
ACTIVE_THRESH  = 0.10   # > 10th percentile of recession mean = "active"
exp_threshold  = np.percentile(mean_exp, 5)
rec_threshold  = np.percentile(mean_rec, 10)

dormant_in_exp  = mean_exp < exp_threshold
active_in_rec   = mean_rec > rec_threshold
crisis_features = dormant_in_exp & active_in_rec

print(f"  Features dormant in expansion: {dormant_in_exp.sum()}")
print(f"  Features active in recession:  {active_in_rec.sum()}")
print(f"  Crisis features (both):        {crisis_features.sum()}")

# Regime ratio per feature
regime_ratio = mean_rec / (mean_exp + 1e-8)
top_crisis_idx = np.argsort(regime_ratio)[::-1][:10]
print(f"\n  Top-10 features by recession/expansion ratio:")
for i, fi in enumerate(top_crisis_idx):
    print(f"    #{i+1}: feature {fi:3d}  ratio={regime_ratio[fi]:.2f}x  "
          f"exp={mean_exp[fi]:.4f}  rec={mean_rec[fi]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHARACTERISTIC LABELING
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Loading panel for characteristic labeling ────────────────────────")
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
char_cols = [c for c in panel.columns if c not in SKIP_COLS]
print(f"  {len(char_cols)} characteristics")

# Align panel rows with activations via permno + yyyymm
panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)

df_idx = pd.DataFrame({'permno': permno_np.astype(int),
                       'yyyymm': yyyymm_np.astype(int)})
panel_aligned = df_idx.merge(panel[['permno', 'yyyymm'] + char_cols],
                              on=['permno', 'yyyymm'], how='left')
chars_np = panel_aligned[char_cols].values.astype(np.float32)
print(f"  Aligned panel shape: {chars_np.shape}")
print(f"  Missing rate: {np.isnan(chars_np).mean():.1%}")

# For each SAE feature: correlation with each characteristic
print("\n  Computing feature–characteristic correlations (top 5 per feature)...")
feature_labels = {}
for fi in range(N_FEAT):
    feat_acts = z_all[:, fi]
    if feat_acts.std() < 1e-8:
        feature_labels[fi] = ['dead']
        continue
    corrs = []
    for ci, col in enumerate(char_cols):
        char_vals = chars_np[:, ci]
        mask = ~np.isnan(char_vals)
        if mask.sum() < 1000:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(feat_acts[mask], char_vals[mask])[0, 1])
    top5_idx = np.argsort(np.abs(corrs))[::-1][:5]
    feature_labels[fi] = [f"{char_cols[i]}({corrs[i]:.2f})" for i in top5_idx]

# Print labels for top crisis features
print("\n  Labels for top-10 crisis features:")
for fi in top_crisis_idx:
    print(f"    Feature {fi:3d} (ratio={regime_ratio[fi]:.2f}x): "
          f"{', '.join(feature_labels[fi])}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MONOSEMANTICITY SCORES
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Monosemanticity scores for crisis features ───────────────────────")
mono_scores = {}
for fi in top_crisis_idx:
    feat_acts = z_all[:, fi]
    if feat_acts.std() < 1e-8:
        mono_scores[fi] = 0.0
        continue
    corrs = []
    for ci in range(len(char_cols)):
        char_vals = chars_np[:, ci]
        mask = ~np.isnan(char_vals)
        if mask.sum() < 1000:
            corrs.append(0.0)
        else:
            corrs.append(abs(np.corrcoef(feat_acts[mask], char_vals[mask])[0, 1]))
    corrs = np.array(corrs)
    mono_scores[fi] = corrs.max() / (corrs.sum() + 1e-8)
    print(f"  Feature {fi:3d}: mono={mono_scores[fi]:.3f}  "
          f"top={feature_labels[fi][0]}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. STABILITY: RETRAIN 5x, COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── SAE stability ({N_STABILITY} runs) ───────────────────────────────────────")
enc_weights = []
for run in range(N_STABILITY):
    sae_r = train_sae(X_exp, lam=best_lam, epochs=50, seed=run * 100)
    enc_weights.append(sae_r.enc.weight.data.cpu().numpy())  # (128, 16)
    print(f"  Run {run+1}/{N_STABILITY} done")

# Cross-run cosine similarity: for each pair of runs, match features greedily
def mean_matched_cosine(W1, W2):
    """For each feature in W1, find best match in W2 by cosine similarity."""
    sim = 1 - cdist(W1, W2, metric='cosine')   # (128, 128)
    return sim.max(axis=1).mean()

cosine_scores = []
for i in range(N_STABILITY):
    for j in range(i+1, N_STABILITY):
        score = mean_matched_cosine(enc_weights[i], enc_weights[j])
        cosine_scores.append(score)

mean_cosine = np.mean(cosine_scores)
print(f"\n  Mean matched cosine similarity across {len(cosine_scores)} pairs: "
      f"{mean_cosine:.3f}")
print(f"  {'✓ STABLE' if mean_cosine > 0.7 else '✗ UNSTABLE'} "
      f"(threshold: > 0.70)")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Saving outputs ───────────────────────────────────────────────────")

# Feature summary table
rows = []
for fi in range(N_FEAT):
    rows.append({
        'feature':       fi,
        'mean_exp':      round(float(mean_exp[fi]), 5),
        'mean_rec':      round(float(mean_rec[fi]), 5),
        'regime_ratio':  round(float(regime_ratio[fi]), 3),
        'dormant_exp':   bool(dormant_in_exp[fi]),
        'active_rec':    bool(active_in_rec[fi]),
        'crisis_feature': bool(crisis_features[fi]),
        'mono_score':    round(mono_scores.get(fi, np.nan), 3),
        'top_char':      feature_labels[fi][0] if feature_labels[fi] else '',
        'label_top5':    ' | '.join(feature_labels[fi]),
    })

df_features = pd.DataFrame(rows)
df_features.to_csv(OUT_DIR / '2d_sae_features.csv', index=False)
print(f"  Saved 2d_sae_features.csv ({len(df_features)} features)")

# Summary stats
summary = {
    'best_lambda':       best_lam,
    'mean_active_features': round(float(mean_active_final), 2),
    'dead_features':     int(n_dead_final),
    'crisis_features':   int(crisis_features.sum()),
    'mean_cosine_stability': round(float(mean_cosine), 3),
    'stable':            bool(mean_cosine > 0.7),
}
pd.DataFrame([summary]).to_csv(OUT_DIR / '2d_sae_summary.csv', index=False)
print(f"  Saved 2d_sae_summary.csv")

print(f"\n{'='*65}")
print("2D COMPLETE")
print(f"{'='*65}")
print(f"  Lambda:              {best_lam}")
print(f"  Mean active features: {mean_active_final:.1f}/128")
print(f"  Crisis features:     {crisis_features.sum()}")
print(f"  Cosine stability:    {mean_cosine:.3f} "
      f"({'✓' if mean_cosine > 0.7 else '✗'})")
print(f"\nNext: proceed to 2B (linear probing, secondary) or 2E (full seed stability).")
print(f"{'='*65}")