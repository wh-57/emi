"""
11_2c_activation_patching.py — Phase 2C: Activation Patching
=============================================================
Per flexmap v4 step 2C (Tier 1 — primary causal result).

Three-experiment design (new in v4, motivated by PR=1.40 finding):

  Exp 1 — Full patch (baseline):
    Replace expansion layer-1 activations with mean recession vector.
    patch_effect = mean|out_patched - out_exp| / mean|out_rec - out_exp|
    Target: CI lower bound > 0.25

  Exp 2 — PC1-only patch:
    Replace only the PC1 component of expansion activations with
    the PC1 component of mean recession vector.
    Keep PC2-PC16 from expansion.
    Tests: how much of the patching effect lives in the dominant direction?

  Exp 3 — Residual patch (PC2-PC6):
    Replace only the orthogonal complement (PC2-PC16).
    Keep PC1 from expansion.
    Tests: is there additional regime-specific structure outside PC1?

  Exp 4 — Reverse patch:
    Replace recession activations with mean expansion vector.
    Tests asymmetry — regime-specific computation should be asymmetric.

  Exp 5 — SAE feature patching:
    Patch specific SAE features (top recession-elevated: 6, 25, 23)
    by replacing their encoded values with recession means.
    Tests whether specific SAE features mediate the regime effect.

Bootstrap 95% CI across 1,000 randomly sampled stocks for each experiment.

Context: 2D found PR=1.40 (near-rank-1). 2B found 0 NW-significant regime
neurons. The patching test is more powerful — it measures output shifts
directly rather than correlating monthly means with a noisy binary indicator.
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR  = BASE / 'activations'
DATA_DIR  = BASE / 'data'
MODEL_DIR = BASE / 'models'

N_BOOTSTRAP  = 1_000
N_SAMPLE     = 500    # stocks per experiment draw
SEED         = 42

# Top SAE features from 2D (recession-elevated)
SAE_CRISIS_FEATURES = [6, 25, 23]   # ratios: 1.98x, 1.62x, 1.47x

# ── Rebuild GKX MLP (needed for forward pass) ─────────────────────────────────
HIDDEN  = [32, 16, 8]
DROPOUT = 0.5

class GKXMLP(nn.Module):
    def __init__(self, n_chars=94, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self._acts = {}
        layers, prev = [], n_chars
        for i, h in enumerate(hidden):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        # Register hooks to capture layer1 activations
        # Layer1 is at index 8 (after layer0: lin+bn+relu+drop=4, then lin+bn+relu+drop=8)
        self.net[8].register_forward_hook(self._hook('layer1_relu'))

    def _hook(self, name):
        def fn(module, inp, out):
            self._acts[name] = out
        return fn

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def forward_from_layer1(self, layer1_acts):
        """Forward pass starting from layer1 activations (skip layer0)."""
        # layer1 activations → dropout → layer2 linear → bn → relu → drop → out
        x = layer1_acts
        # indices 11-15: dropout, linear(16→8), bn, relu, dropout; 16: linear(8→1)
        for idx in range(11, len(self.net)):
            x = self.net[idx](x)
        return x.squeeze(-1)


# ── Load model checkpoint (latest) ───────────────────────────────────────────
print("Loading GKX model...")
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
             'prc', 'shrout', 'exchcd', 'shrcd']
panel_tmp = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet', columns=['yyyymm'])
char_cols  = None  # will be set after full panel load

# Load full panel to get char_cols
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
char_cols = [c for c in panel.columns if c not in SKIP_COLS]
n_chars   = len(char_cols)
print(f"  Characteristics: {n_chars}")

model = GKXMLP(n_chars=n_chars).to(DEVICE)
# Load latest checkpoint
ckpt_files = sorted(MODEL_DIR.glob(f'checkpoint_*_seed{SEED}.pt'))
if not ckpt_files:
    ckpt_files = sorted(MODEL_DIR.glob('checkpoint_*.pt'))
ckpt = ckpt_files[-1]
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print(f"  Loaded checkpoint: {ckpt.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ACTIVATIONS AND COMPUTE REGIME STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading layer1 activations and regime labels...")
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

acts_list, regime_list, yyyymm_list, permno_list = [], [], [], []
h5_files = sorted(ACTS_DIR.glob('acts_*.h5'))

for f in h5_files:
    yyyymm = int(f.stem.replace('acts_', ''))
    if yyyymm not in regime_map or pd.isna(regime_map[yyyymm]):
        continue
    with h5py.File(f, 'r') as hf:
        acts    = hf['layer1'][:]
        permnos = hf['permno'][:]
    acts_list.append(acts)
    regime_list.append(np.full(len(acts), regime_map[yyyymm], dtype=np.float32))
    yyyymm_list.append(np.full(len(acts), yyyymm, dtype=np.int32))
    permno_list.append(permnos)

acts_np   = np.vstack(acts_list).astype(np.float32)
regime_np = np.concatenate(regime_list)
yyyymm_np = np.concatenate(yyyymm_list)
permno_np = np.concatenate(permno_list)

# Normalize (same as 2D)
acts_mean = acts_np.mean(0, keepdims=True)
acts_std  = acts_np.std(0,  keepdims=True) + 1e-8
acts_norm = (acts_np - acts_mean) / acts_std

exp_mask = regime_np == 0
rec_mask = regime_np == 1
n_exp    = exp_mask.sum()
n_rec    = rec_mask.sum()
print(f"  Expansion: {n_exp:,}  Recession: {n_rec:,}")

# Mean recession and expansion activation vectors
mean_rec_vec = acts_norm[rec_mask].mean(0)   # (16,)
mean_exp_vec = acts_norm[exp_mask].mean(0)   # (16,)
print(f"  Mean rec activation: {mean_rec_vec.mean():.4f}")
print(f"  Mean exp activation: {mean_exp_vec.mean():.4f}")

# ── Fit PCA for decomposition ─────────────────────────────────────────────────
print("\nFitting PCA for PC1 decomposition...")
pca    = PCA(n_components=16)
pca.fit(acts_norm[exp_mask][:50_000])
pc1    = pca.components_[0]                  # (16,) dominant direction
print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.3f}")

# Decompose mean recession shift
shift        = mean_rec_vec - mean_exp_vec   # (16,)
shift_pc1    = (shift @ pc1) * pc1           # projection onto PC1
shift_resid  = shift - shift_pc1            # orthogonal complement

print(f"  Total shift magnitude:    {np.linalg.norm(shift):.4f}")
print(f"  PC1 component magnitude:  {np.linalg.norm(shift_pc1):.4f}  "
      f"({np.linalg.norm(shift_pc1)/np.linalg.norm(shift)*100:.1f}% of total)")
print(f"  Residual magnitude:       {np.linalg.norm(shift_resid):.4f}  "
      f"({np.linalg.norm(shift_resid)/np.linalg.norm(shift)*100:.1f}% of total)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD SAE FOR FEATURE PATCHING (Exp 5)
# ══════════════════════════════════════════════════════════════════════════════

class SAE(nn.Module):
    def __init__(self, n_in=16, n_feat=32):
        super().__init__()
        self.enc = nn.Linear(n_in, n_feat)
        self.dec = nn.Linear(n_feat, n_in)
        nn.init.orthogonal_(self.enc.weight)
        nn.init.orthogonal_(self.dec.weight)

    def forward(self, x):
        z = torch.relu(self.enc(x))
        return self.dec(z), z


def train_sae_quick(X_train, lam=0.6, epochs=50, seed=2026):
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


print("\nRetraining SAE for feature patching (lam=0.6, 50ep)...")
X_exp_norm = acts_norm[exp_mask]
sae = train_sae_quick(X_exp_norm, lam=0.6, epochs=50)

# Compute mean SAE encodings per regime
with torch.no_grad():
    _, z_exp_sae = sae(torch.tensor(X_exp_norm[:100_000],
                                     dtype=torch.float32).to(DEVICE))
    _, z_rec_sae = sae(torch.tensor(acts_norm[rec_mask][:50_000],
                                     dtype=torch.float32).to(DEVICE))
mean_z_exp = z_exp_sae.cpu().numpy().mean(0)   # (32,)
mean_z_rec = z_rec_sae.cpu().numpy().mean(0)   # (32,)
print(f"  SAE ready. Crisis feature recession means: "
      f"{[round(float(mean_z_rec[f]),4) for f in SAE_CRISIS_FEATURES]}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_predictions_from_chars(permnos, yyyymms):
    """Get model predictions from raw characteristics for given stocks."""
    panel_sub = pd.DataFrame({'permno': permnos.astype(int),
                               'yyyymm': yyyymms.astype(int)})
    panel_sub = panel_sub.merge(panel[['permno','yyyymm'] + char_cols],
                                 on=['permno','yyyymm'], how='left')
    X = panel_sub[char_cols].fillna(0).values.astype(np.float32)
    X_t = torch.tensor(X).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds


def patch_layer1_and_predict(acts_original, patch_delta):
    """
    Apply additive patch to layer1 activations and get output.
    acts_original: (n, 16) normalized activations
    patch_delta:   (16,) vector to add
    """
    acts_patched = acts_original + patch_delta[None, :]
    # De-normalize back to original scale for model forward pass
    acts_denorm  = acts_patched * acts_std + acts_mean
    acts_t       = torch.tensor(acts_denorm, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model.forward_from_layer1(acts_t).cpu().numpy()
    return preds


def patch_sae_features_and_predict(acts_original, feature_indices):
    """
    Patch specific SAE features: replace their encoded values with
    recession mean, decode back to activation space, then predict.
    """
    acts_t = torch.tensor(acts_original, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, z = sae(acts_t)
        z_patched = z.clone()
        for fi in feature_indices:
            z_patched[:, fi] = float(mean_z_rec[fi])
        acts_reconstructed = sae.dec(z_patched)
    acts_denorm = acts_reconstructed.cpu().numpy() * acts_std + acts_mean
    acts_t2     = torch.tensor(acts_denorm, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model.forward_from_layer1(acts_t2).cpu().numpy()
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# 4. BOOTSTRAP PATCH EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Running patching experiments ({N_BOOTSTRAP} bootstrap draws) ──────────")

# Sample expansion stocks for patching
exp_idx  = np.where(exp_mask)[0]
rec_idx  = np.where(rec_mask)[0]

# Pre-get recession predictions (baseline)
rec_sample = np.random.choice(rec_idx, min(N_SAMPLE, len(rec_idx)), replace=False)
pred_rec_baseline = patch_layer1_and_predict(
    acts_norm[rec_sample], np.zeros(16))

results = {f'exp{i}': [] for i in range(1, 8)}

for b in range(N_BOOTSTRAP):
    if b % 200 == 0:
        print(f"  Bootstrap {b}/{N_BOOTSTRAP}...")

    # Sample expansion stocks
    sample_idx = np.random.choice(exp_idx, N_SAMPLE, replace=True)
    acts_orig  = acts_norm[sample_idx]          # (N_SAMPLE, 16)

    # Baseline expansion predictions
    pred_exp = patch_layer1_and_predict(acts_orig, np.zeros(16))

    # Reference recession prediction (for normalization)
    pred_rec = pred_rec_baseline
    gap      = np.abs(pred_rec.mean() - pred_exp.mean())

    if gap < 1e-8:
        continue

    # Exp 1: Full patch — replace with mean recession vector
    pred_full = patch_layer1_and_predict(acts_orig, shift)
    e1 = np.abs(pred_full.mean() - pred_exp.mean()) / gap
    results['exp1'].append(e1)

    # Exp 2: PC1-only patch
    pred_pc1 = patch_layer1_and_predict(acts_orig, shift_pc1)
    e2 = np.abs(pred_pc1.mean() - pred_exp.mean()) / gap
    results['exp2'].append(e2)

    # Exp 3: Residual patch (PC2-PC16)
    pred_resid = patch_layer1_and_predict(acts_orig, shift_resid)
    e3 = np.abs(pred_resid.mean() - pred_exp.mean()) / gap
    results['exp3'].append(e3)

    # Exp 4: Reverse patch — replace recession with expansion mean
    rec_b = np.random.choice(rec_idx, N_SAMPLE, replace=True)
    pred_rec_b   = patch_layer1_and_predict(acts_norm[rec_b], np.zeros(16))
    pred_rev     = patch_layer1_and_predict(acts_norm[rec_b], -shift)
    gap_rev      = np.abs(pred_rec_b.mean() - pred_exp.mean())
    e4 = (np.abs(pred_rev.mean() - pred_rec_b.mean()) / gap_rev
          if gap_rev > 1e-8 else 0.0)
    results['exp4'].append(e4)

    # Exp 5: SAE feature patching (features 6, 25, 23)
    pred_sae = patch_sae_features_and_predict(acts_orig, SAE_CRISIS_FEATURES)
    e5 = np.abs(pred_sae.mean() - pred_exp.mean()) / gap
    results['exp5'].append(e5)

    # Exp 6: Placebo — random direction, same norm as full shift
    # If PC1 patch dominates but this does not → result is not just "big perturbation"
    np.random.seed(b)
    rand_dir  = np.random.randn(16)
    rand_dir  = rand_dir / np.linalg.norm(rand_dir) * np.linalg.norm(shift)
    pred_rand = patch_layer1_and_predict(acts_orig, rand_dir)
    e6 = np.abs(pred_rand.mean() - pred_exp.mean()) / gap
    results['exp6'].append(e6)

    # Exp 7: Placebo — random direction orthogonal to PC1, same norm as shift_resid
    # If residual patch does nothing but this also does nothing → residual is just noise
    rand_orth = np.random.randn(16)
    rand_orth = rand_orth - (rand_orth @ pc1) * pc1   # project out PC1
    norm_orth = np.linalg.norm(rand_orth)
    if norm_orth > 1e-8:
        rand_orth = rand_orth / norm_orth * np.linalg.norm(shift_resid)
        pred_orth = patch_layer1_and_predict(acts_orig, rand_orth)
        e7 = np.abs(pred_orth.mean() - pred_exp.mean()) / gap
    else:
        e7 = 0.0
    results['exp7'].append(e7)


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("2C ACTIVATION PATCHING — RESULTS")
print(f"{'='*65}")

print(f"\n  Mean recession-expansion activation shift:")
print(f"    Total:        {np.linalg.norm(shift):.4f}")
print(f"    PC1 share:    {np.linalg.norm(shift_pc1)/np.linalg.norm(shift)*100:.1f}%")
print(f"    Residual:     {np.linalg.norm(shift_resid)/np.linalg.norm(shift)*100:.1f}%")

print(f"\n  {'Experiment':<35} {'Mean':>7} {'CI_lo':>7} {'CI_hi':>7} {'Pass':>6}")
print(f"  {'-'*62}")

exp_labels = [
    ('Exp 1 — Full patch (baseline)',         'exp1', 0.25),
    ('Exp 2 — PC1-only patch',                'exp2', None),
    ('Exp 3 — Residual patch (PC2+)',         'exp3', None),
    ('Exp 4 — Reverse patch',                 'exp4', None),
    ('Exp 5 — SAE features 6,25,23',          'exp5', None),
    ('Exp 6 — Placebo: random, same norm',    'exp6', None),
    ('Exp 7 — Placebo: rand ortho to PC1',    'exp7', None),
]

output_rows = []
for label, key, threshold in exp_labels:
    vals = np.array(results[key])
    if len(vals) == 0:
        continue
    mean_e = float(vals.mean())
    ci_lo  = float(np.percentile(vals, 2.5))
    ci_hi  = float(np.percentile(vals, 97.5))
    pass_  = '✓' if (threshold and ci_lo > threshold) else \
             ('✓' if (threshold and mean_e > threshold) else
              ('—' if not threshold else '✗'))
    print(f"  {label:<35} {mean_e:>7.3f} {ci_lo:>7.3f} {ci_hi:>7.3f} {pass_:>6}")
    output_rows.append({
        'experiment': label,
        'mean':  round(mean_e, 4),
        'ci_lo': round(ci_lo,  4),
        'ci_hi': round(ci_hi,  4),
        'threshold': threshold,
        'pass': pass_,
    })

# PC1 share of patching effect
if results['exp1'] and results['exp2']:
    pc1_share = np.array(results['exp2']).mean() / (np.array(results['exp1']).mean() + 1e-8)
    print(f"\n  PC1 share of full patch effect: {pc1_share*100:.1f}%")
    print(f"  Residual share:                 {(1-pc1_share)*100:.1f}%")

    if pc1_share > 0.80:
        print(f"\n  → Regime shift is almost entirely in PC1 (near-rank-1 confirmed).")
        print(f"    The dominant direction drives the causal output change.")
    elif pc1_share > 0.50:
        print(f"\n  → Majority of regime shift in PC1, some in residual.")
        print(f"    Multi-dimensional but PC1-dominated structure.")
    else:
        print(f"\n  → Regime shift is distributed across PC1 and higher PCs.")
        print(f"    More complex than near-rank-1 structure suggests.")

# Placebo check
if results['exp2'] and results['exp6']:
    pc1_mean   = np.array(results['exp2']).mean()
    rand_mean  = np.array(results['exp6']).mean()
    orth_mean  = np.array(results['exp7']).mean() if results['exp7'] else 0
    print(f"\n  Placebo check (rules out 'any large perturbation moves output'):")
    print(f"    PC1 patch effect:           {pc1_mean:.3f}")
    print(f"    Random direction (same norm): {rand_mean:.3f}")
    print(f"    Random orth to PC1:           {orth_mean:.3f}")
    if pc1_mean > 2 * rand_mean:
        print(f"  ✓ PC1 patch >> random direction. Direction matters, not just magnitude.")
    else:
        print(f"  ~ PC1 patch ≈ random direction. Effect may be size-driven, not PC1-specific.")
if results['exp1'] and results['exp4']:
    forward_mean = np.array(results['exp1']).mean()
    reverse_mean = np.array(results['exp4']).mean()
    asymmetry    = forward_mean / (reverse_mean + 1e-8)
    print(f"\n  Patching asymmetry (forward/reverse): {asymmetry:.2f}x")
    if asymmetry > 1.2:
        print(f"  → Asymmetric: expansion→recession patch > recession→expansion.")
        print(f"    Consistent with regime-specific computation.")
    else:
        print(f"  → Approximately symmetric: same computation in both regimes.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PHASE 2 CHECKPOINT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PHASE 2 CHECKPOINT")
print(f"{'='*65}")

exp1_vals  = np.array(results['exp1'])
exp1_mean  = float(exp1_vals.mean()) if len(exp1_vals) > 0 else 0
exp1_ci_lo = float(np.percentile(exp1_vals, 2.5)) if len(exp1_vals) > 0 else 0

print(f"\n  Flexmap Phase 2 GO criterion:")
print(f"  patch_effect (Exp 1) ≥ 0.25 → mean={exp1_mean:.3f}, CI_lo={exp1_ci_lo:.3f}")

if exp1_ci_lo > 0.25:
    print(f"  ✓ PASS — CI lower bound {exp1_ci_lo:.3f} > 0.25")
    print(f"    Causal evidence: patching expansion with recession activations")
    print(f"    shifts output by {exp1_mean*100:.1f}% of the recession-expansion gap.")
elif exp1_mean > 0.25:
    print(f"  ~ BORDERLINE — mean {exp1_mean:.3f} > 0.25 but CI_lo={exp1_ci_lo:.3f} < 0.25")
    print(f"    Weak causal evidence. Check robustness.")
else:
    print(f"  ✗ FAIL — mean {exp1_mean:.3f} < 0.25")
    print(f"    Patching effect too small. Check model forward pass implementation.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Saving outputs ───────────────────────────────────────────────────")
df_results = pd.DataFrame(output_rows)
df_results.to_csv(DATA_DIR / '2c_patching_results.csv', index=False)

# Save shift decomposition
df_shift = pd.DataFrame({
    'neuron':          list(range(16)),
    'shift_total':     shift,
    'shift_pc1':       shift_pc1,
    'shift_residual':  shift_resid,
    'pc1_loading':     pca.components_[0],
})
df_shift.to_csv(DATA_DIR / '2c_shift_decomposition.csv', index=False)
print(f"  Saved 2c_patching_results.csv")
print(f"  Saved 2c_shift_decomposition.csv")

print(f"\n{'='*65}")
print("2C COMPLETE")
print(f"{'='*65}")
print(f"\nNext: 2E (full seed stability) → Phase 2 checkpoint decision")
print(f"{'='*65}")