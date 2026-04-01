"""
09_2d_sae_real.py — Phase 2D: SAE on Real Layer-1 Activations (v3 FINAL)
=========================================================================
Per flexmap v3 step 2D (SAE-first path, confirmed by 1E Jaccard=0.154).

Changes from v2 (finalized after 2-round multi-LLM debate 2026-03-31):
  - Participation ratio added to PCA diagnostic
  - 3-way adaptive dictionary size: 32 / 64 / 128 based on rank_95
  - Activation scale verification after normalization
  - Subperiod PCA stability check (non-stationarity diagnostic)
  - Framing: SAE is a tool; activation patching is Tier 1

Unchanged from v2 (confirmed correct by all LLMs):
  - L1 SAE, expansion-only training
  - Orthogonal initialization
  - 300k/40ep lambda tuning subsample
  - Lambda grid [0.3, 0.6, 1.0, 1.5, 2.0]
  - Two-step alive-feature crisis detection
  - Zero-variance characteristic filter
  - Vectorized float64 correlation
  - Filtered cosine stability (alive features only)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(2026)
np.random.seed(2026)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR = BASE / 'activations'
DATA_DIR = BASE / 'data'

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_KEY     = 'layer1'
N_IN         = 16
SKIP_COLS    = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
                'prc', 'shrout', 'exchcd', 'shrcd']
N_STABILITY  = 5
LAM_GRID     = [0.3, 0.6, 1.0, 1.5, 2.0]
LAM_TUNE_N   = 300_000
LAM_TUNE_EP  = 40
FULL_EP      = 50
ALIVE_THRESH = 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ACTIVATIONS + REGIME LABELS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading regime labels...")
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

print("Loading layer1 activations from HDF5 files...")
acts_list, regime_list, yyyymm_list, permno_list = [], [], [], []

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
        acts    = hf[HDF5_KEY][:]
        permnos = hf['permno'][:]
    acts_list.append(acts)
    regime_list.append(np.full(len(acts), nber,   dtype=np.float32))
    yyyymm_list.append(np.full(len(acts), yyyymm, dtype=np.int32))
    permno_list.append(permnos)

acts_np   = np.vstack(acts_list).astype(np.float32)
regime_np = np.concatenate(regime_list)
yyyymm_np = np.concatenate(yyyymm_list)
permno_np = np.concatenate(permno_list)

n_rec = int((regime_np == 1).sum())
n_exp = int((regime_np == 0).sum())
print(f"Total: {len(acts_np):,}  |  Expansion: {n_exp:,}  |  Recession: {n_rec:,}")

# ── Normalize ─────────────────────────────────────────────────────────────────
acts_mean = acts_np.mean(axis=0, keepdims=True)
acts_std  = acts_np.std(axis=0, keepdims=True) + 1e-8
acts_norm = (acts_np - acts_mean) / acts_std

# ── Activation scale verification (post-normalization) ────────────────────────
print(f"\n── Activation scale verification ────────────────────────────────────")
frac_near_zero = (np.abs(acts_norm) < 1e-6).mean()
print(f"  Range:           [{acts_norm.min():.3e}, {acts_norm.max():.3e}]")
print(f"  Std:             {acts_norm.std():.4f}")
print(f"  Mean abs:        {np.abs(acts_norm).mean():.4f}")
print(f"  Fraction |x|<1e-6: {frac_near_zero:.2%}")
if frac_near_zero > 0.90:
    print("  WARNING: >90% of activations near zero after normalization.")
    print("  This indicates scale collapse in layer1 — check GKX training.")
else:
    print("  ✓ Activation scale looks healthy.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA RANK DIAGNOSTIC (with participation ratio + subperiod stability)
# ══════════════════════════════════════════════════════════════════════════════

exp_mask = regime_np == 0
X_exp    = acts_norm[exp_mask]
exp_yyyymm = yyyymm_np[exp_mask]

print(f"\n── PCA rank diagnostic ──────────────────────────────────────────────")
pca = PCA(n_components=N_IN)
pca.fit(X_exp[:50_000])

ev      = pca.explained_variance_
ev_r    = pca.explained_variance_ratio_
cum_var = np.cumsum(ev_r)

rank_90 = int(np.searchsorted(cum_var, 0.90)) + 1
rank_95 = int(np.searchsorted(cum_var, 0.95)) + 1
rank_99 = int(np.searchsorted(cum_var, 0.99)) + 1
part_ratio = float(ev.sum()**2 / (ev**2).sum())

print(f"  {'PC':>4}  {'Var':>7}  {'Cumul':>7}")
for i in range(N_IN):
    marker = " ←" if i+1 in [rank_90, rank_95, rank_99] else ""
    print(f"  {i+1:>4}  {ev_r[i]:>7.3f}  {cum_var[i]:>7.3f}{marker}")

print(f"\n  Effective rank @ 90%: {rank_90}")
print(f"  Effective rank @ 95%: {rank_95}")
print(f"  Effective rank @ 99%: {rank_99}")
print(f"  Participation ratio:  {part_ratio:.2f}  "
      f"(PR≈1 = rank-1; PR≈16 = full rank)")

# ── Adaptive dictionary size (3-way) ─────────────────────────────────────────
if rank_95 <= 6:
    N_FEAT = 32
    dict_label = "low rank"
elif rank_95 <= 10:
    N_FEAT = 64
    dict_label = "moderate rank"
else:
    N_FEAT = 128
    dict_label = "high rank"
print(f"\n  → {dict_label} (rank_95={rank_95}) → dictionary size: {N_FEAT}")

# ── Subperiod PCA stability (non-stationarity check) ─────────────────────────
print(f"\n── Subperiod PCA stability (non-stationarity diagnostic) ────────────")
subperiods = [
    ('1987–1999', 198701, 199912),
    ('2000–2009', 200001, 200912),
    ('2010–2023', 201001, 202312),
]
sub_ranks = {}
for label, lo, hi in subperiods:
    mask = (exp_yyyymm >= lo) & (exp_yyyymm <= hi)
    if mask.sum() < 5000:
        print(f"  {label}: too few obs ({mask.sum()}), skip")
        continue
    pca_sub = PCA(n_components=N_IN)
    pca_sub.fit(X_exp[mask][:30_000])
    cv_sub  = np.cumsum(pca_sub.explained_variance_ratio_)
    r95_sub = int(np.searchsorted(cv_sub, 0.95)) + 1
    pr_sub  = float(pca_sub.explained_variance_.sum()**2 /
                    (pca_sub.explained_variance_**2).sum())
    sub_ranks[label] = r95_sub
    print(f"  {label}: rank_95={r95_sub}  PR={pr_sub:.2f}  "
          f"(n={mask.sum():,})")

rank_spread = max(sub_ranks.values()) - min(sub_ranks.values()) if sub_ranks else 0
if rank_spread <= 2:
    print(f"  ✓ Rank stable across subperiods (spread={rank_spread})")
else:
    print(f"  ⚠ Rank varies across subperiods (spread={rank_spread}) "
          f"— basis may be time-varying. Document as diagnostic finding.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SAE(nn.Module):
    def __init__(self, n_in=N_IN, n_feat=None):
        super().__init__()
        if n_feat is None:
            n_feat = N_FEAT
        self.n_feat = n_feat
        self.enc = nn.Linear(n_in, n_feat)
        self.dec = nn.Linear(n_feat, n_in)
        nn.init.orthogonal_(self.enc.weight)
        nn.init.orthogonal_(self.dec.weight)

    def forward(self, x):
        z = torch.relu(self.enc(x))
        return self.dec(z), z


def train_sae(X_train, lam, epochs=FULL_EP, seed=2026, n_feat=None):
    torch.manual_seed(seed)
    sae = SAE(n_feat=n_feat or N_FEAT).to(DEVICE)
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
    X_t    = torch.tensor(X, dtype=torch.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_t), 2048):
            _, z = sae(X_t[start:start+2048].to(DEVICE))
            chunks.append(z.cpu().numpy())
    return np.vstack(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LAMBDA TUNING (300k subsample, 40 epochs)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Lambda tuning ({LAM_TUNE_N//1000}k subsample, {LAM_TUNE_EP} epochs, "
      f"dict={N_FEAT}) ───────────────")
print(f"   Target: 5–15 active features, dead < 30%")

n_sub   = min(LAM_TUNE_N, len(X_exp))
sub_idx = np.random.choice(len(X_exp), n_sub, replace=False)
X_tune  = X_exp[sub_idx]

best_lam   = None
best_score = np.inf

for lam in LAM_GRID:
    sae_t       = train_sae(X_tune, lam=lam, epochs=LAM_TUNE_EP,
                            seed=2026, n_feat=N_FEAT)
    z           = get_encoded(sae_t, X_tune)
    n_dead      = (z.mean(0) < 1e-6).sum()
    mean_active = (z > 1e-6).sum(axis=1).mean()
    dead_pct    = n_dead / N_FEAT * 100
    score       = abs(mean_active - 7.5)
    flag        = " ← candidate" if mean_active <= 15 and dead_pct < 30 else ""
    print(f"  lam={lam:.2f}  active={mean_active:5.1f}/{N_FEAT}  "
          f"dead={n_dead}/{N_FEAT} ({dead_pct:.0f}%){flag}")
    if score < best_score and dead_pct < 30:
        best_score = score
        best_lam   = lam

if best_lam is None:
    best_lam = LAM_GRID[0]
    print(f"  Warning: no lambda met criteria. Defaulting to {best_lam}. "
          f"Consider extending grid downward to [0.05, 0.1, 0.2].")

print(f"\n  → Selected lambda: {best_lam}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FULL SAE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Full SAE training (lam={best_lam}, {FULL_EP} epochs, "
      f"N={len(X_exp):,}) ──────")
sae_main = train_sae(X_exp, lam=best_lam, epochs=FULL_EP, seed=2026)
z_exp    = get_encoded(sae_main, X_exp)

n_dead_final     = (z_exp.mean(0) < 1e-6).sum()
mean_active_final = (z_exp > 1e-6).sum(axis=1).mean()
dead_pct_final   = n_dead_final / N_FEAT * 100

print(f"  Dead features:         {n_dead_final}/{N_FEAT} ({dead_pct_final:.0f}%)")
print(f"  Mean active per input: {mean_active_final:.1f}")

if dead_pct_final > 50:
    print(f"  WARNING: >50% dead after full training. "
          f"Try reducing lambda one step to {LAM_GRID[max(0,LAM_GRID.index(best_lam)-1)]}.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. APPLY TO ALL MONTHS + TWO-STEP CRISIS DETECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Applying SAE to all months ───────────────────────────────────────")
z_all = get_encoded(sae_main, acts_norm)

rec_mask  = regime_np == 1
z_rec     = z_all[rec_mask]
z_exp_all = z_all[exp_mask]

mean_exp = z_exp_all.mean(axis=0)
mean_rec = z_rec.mean(axis=0)

# Step 1: alive filter
alive_mask = (mean_exp > ALIVE_THRESH) | (mean_rec > ALIVE_THRESH)
n_alive    = alive_mask.sum()
print(f"  Alive features (active in ≥1 regime): {n_alive}/{N_FEAT}")

# Step 2: within-alive crisis filter
if n_alive >= 10:
    exp_threshold = np.percentile(mean_exp[alive_mask], 25)
    rec_threshold = np.percentile(mean_rec[alive_mask], 75)
else:
    exp_threshold, rec_threshold = 0.01, 0.05
    print(f"  Warning: only {n_alive} alive features — using absolute thresholds")

dormant_in_exp  = (mean_exp < exp_threshold) & alive_mask
active_in_rec   = (mean_rec > rec_threshold) & alive_mask
crisis_features = dormant_in_exp & active_in_rec

print(f"  Exp Q1 threshold: {exp_threshold:.5f}")
print(f"  Rec Q3 threshold: {rec_threshold:.5f}")
print(f"  Dormant in expansion: {dormant_in_exp.sum()}")
print(f"  Active in recession:  {active_in_rec.sum()}")
print(f"  Crisis features:      {crisis_features.sum()}")

regime_ratio = np.zeros(N_FEAT)
regime_ratio[alive_mask] = (mean_rec[alive_mask] /
                             (mean_exp[alive_mask] + 1e-8))

top_crisis_idx = np.argsort(regime_ratio * alive_mask)[::-1][:10]
print(f"\n  Top-10 alive features by recession/expansion ratio:")
for i, fi in enumerate(top_crisis_idx):
    if not alive_mask[fi]:
        continue
    print(f"    #{i+1}: feature {fi:3d}  ratio={regime_ratio[fi]:.2f}x  "
          f"exp={mean_exp[fi]:.5f}  rec={mean_rec[fi]:.5f}  "
          f"crisis={'✓' if crisis_features[fi] else '—'}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHARACTERISTIC LABELING (vectorized, float64, zero-var filtered)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Loading panel for characteristic labeling ────────────────────────")
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
char_cols = [c for c in panel.columns if c not in SKIP_COLS]

panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)
df_idx = pd.DataFrame({'permno': permno_np.astype(int),
                       'yyyymm': yyyymm_np.astype(int)})
panel_aligned = df_idx.merge(panel[['permno', 'yyyymm'] + char_cols],
                              on=['permno', 'yyyymm'], how='left')
chars_np = panel_aligned[char_cols].values.astype(np.float32)

# Zero-variance filter
char_stds       = np.nanstd(chars_np, axis=0)
valid_mask      = char_stds > 1e-6
valid_char_cols = [char_cols[i] for i in range(len(char_cols)) if valid_mask[i]]
chars_valid     = chars_np[:, valid_mask]
dropped         = [char_cols[i] for i in range(len(char_cols)) if not valid_mask[i]]
print(f"  Dropped zero-variance: {dropped}")
print(f"  Valid characteristics: {len(valid_char_cols)}/{len(char_cols)}")

# Vectorized correlation (float64)
print("\n  Computing correlations (vectorized, float64)...")
N   = len(z_all)
z_c = (z_all - z_all.mean(0)).astype(np.float64)
ch_c = np.where(np.isnan(chars_valid.astype(np.float64)),
                0.0,
                chars_valid.astype(np.float64) -
                np.nanmean(chars_valid.astype(np.float64), axis=0))

z_std  = z_c.std(0)  + 1e-8
ch_std = ch_c.std(0) + 1e-8
corr_matrix = (z_c.T @ ch_c) / (N * z_std[:, None] * ch_std[None, :])
print(f"  Correlation matrix: {corr_matrix.shape}")

feature_labels = {}
for fi in range(N_FEAT):
    if not alive_mask[fi]:
        feature_labels[fi] = ['dead']
        continue
    row  = corr_matrix[fi]
    top5 = np.argsort(np.abs(row))[::-1][:5]
    feature_labels[fi] = [f"{valid_char_cols[i]}({row[i]:.2f})" for i in top5]

print("\n  Labels for top-10 crisis/alive features:")
for fi in top_crisis_idx:
    if not alive_mask[fi]:
        continue
    print(f"    Feature {fi:3d} (ratio={regime_ratio[fi]:.2f}x): "
          f"{', '.join(feature_labels[fi])}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MONOSEMANTICITY
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Monosemanticity scores (alive features) ──────────────────────────")
mono_scores = {}
for fi in top_crisis_idx:
    if not alive_mask[fi]:
        mono_scores[fi] = 0.0
        continue
    row  = np.abs(corr_matrix[fi])
    mono_scores[fi] = float(row.max() / (row.sum() + 1e-8))
    print(f"  Feature {fi:3d}: mono={mono_scores[fi]:.3f}  "
          f"top={feature_labels[fi][0]}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. STABILITY: 5 RERUNS, FILTERED COSINE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── SAE stability ({N_STABILITY} runs, filtered cosine) ──────────────────────")
enc_weights  = []
alive_counts = []

for run in range(N_STABILITY):
    sae_r   = train_sae(X_exp, lam=best_lam, epochs=FULL_EP,
                        seed=run * 100, n_feat=N_FEAT)
    z_r     = get_encoded(sae_r, X_exp[:50_000])
    alive_r = z_r.mean(0) > ALIVE_THRESH
    enc_weights.append(sae_r.enc.weight.data.cpu().numpy())
    alive_counts.append(alive_r.sum())
    print(f"  Run {run+1}/{N_STABILITY}  alive={alive_r.sum()}/{N_FEAT}")


def filtered_cosine(W1, W2, alive1, alive2):
    both = alive1 & alive2
    if both.sum() < 2:
        return 0.0
    sim = 1 - cdist(W1[both], W2[both], metric='cosine')
    return sim.max(axis=1).mean()


cosine_scores = []
for i in range(N_STABILITY):
    for j in range(i+1, N_STABILITY):
        a_i = enc_weights[i].mean(1) > 0
        a_j = enc_weights[j].mean(1) > 0
        cosine_scores.append(filtered_cosine(enc_weights[i], enc_weights[j],
                                             a_i, a_j))

mean_cosine = float(np.mean(cosine_scores)) if cosine_scores else 0.0
print(f"\n  Filtered cosine (alive features only): {mean_cosine:.3f}")
print(f"  {'✓ STABLE' if mean_cosine > 0.7 else '✗ UNSTABLE'} (threshold: > 0.70)")
print(f"  Mean alive across runs: {np.mean(alive_counts):.1f}/{N_FEAT}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Saving outputs ───────────────────────────────────────────────────")

rows = []
for fi in range(N_FEAT):
    rows.append({
        'feature':         fi,
        'alive':           bool(alive_mask[fi]),
        'mean_exp':        round(float(mean_exp[fi]), 6),
        'mean_rec':        round(float(mean_rec[fi]), 6),
        'regime_ratio':    round(float(regime_ratio[fi]), 3),
        'dormant_exp':     bool(dormant_in_exp[fi]),
        'active_rec':      bool(active_in_rec[fi]),
        'crisis_feature':  bool(crisis_features[fi]),
        'mono_score':      round(mono_scores.get(fi, np.nan), 3),
        'top_char':        feature_labels[fi][0] if feature_labels[fi] else '',
        'label_top5':      ' | '.join(feature_labels[fi]),
    })

df_features = pd.DataFrame(rows)
df_features.to_csv(DATA_DIR / '2d_sae_features.csv', index=False)

summary = {
    'best_lambda':             best_lam,
    'n_feat':                  N_FEAT,
    'effective_rank_95':       rank_95,
    'participation_ratio':     round(part_ratio, 2),
    'rank_spread_subperiods':  rank_spread,
    'mean_active_features':    round(float(mean_active_final), 2),
    'dead_features':           int(n_dead_final),
    'alive_features':          int(n_alive),
    'crisis_features':         int(crisis_features.sum()),
    'filtered_cosine':         round(mean_cosine, 3),
    'stable':                  bool(mean_cosine > 0.7),
}
pd.DataFrame([summary]).to_csv(DATA_DIR / '2d_sae_summary.csv', index=False)
print(f"  Saved 2d_sae_features.csv and 2d_sae_summary.csv")

print(f"\n{'='*65}")
print("2D COMPLETE — DIAGNOSTIC SUMMARY")
print(f"{'='*65}")
print(f"  Effective rank (95%):   {rank_95}  (PR={part_ratio:.2f})")
print(f"  Rank subperiod spread:  {rank_spread}  "
      f"({'stable' if rank_spread<=2 else 'TIME-VARYING — document'})")
print(f"  Dictionary size:        {N_FEAT}")
print(f"  Lambda:                 {best_lam}")
print(f"  Dead features:          {n_dead_final}/{N_FEAT} ({dead_pct_final:.0f}%)")
print(f"  Alive features:         {n_alive}/{N_FEAT}")
print(f"  Crisis features:        {crisis_features.sum()}")
print(f"  Filtered cosine:        {mean_cosine:.3f} "
      f"({'✓' if mean_cosine > 0.7 else '✗'})")
print(f"\n{'='*65}")
print("INTERPRETATION GUIDE")
print(f"{'='*65}")
print(f"  SAE is a tool, not the contribution.")
print(f"  Tier 1 (must succeed): Activation patching → Phase 2C")
print(f"  Tier 2 (strong support): SAE crisis features → here")
print(f"  Tier 3 (nice-to-have): perfect sparsity")
print(f"\n  Case A — rank 8–12, active 5–12: proceed to crisis detection")
print(f"  Case B — rank 4–6, active ~5–6:  low-dim compression result")
print(f"  Case C — rank 12–16, active >15: distributed repr, adjust threshold")
print(f"\nNext: 2B (linear probing, secondary) → 2C (patching on SAE features)")
print(f"{'='*65}")