"""
11_2c_activation_patching.py — Phase 2C: Activation Patching (v2)
==================================================================
Per flexmap v5 step 2C (Tier 1 — primary causal result).

v2 fixes from first run:
  - Denominator changed to std(predictions) — stable, stock-agnostic
  - Exp 4 (reverse patch) removed — stock composition confound
  - forward_from_layer1 starts at index 7, not 11
  - Placebo experiments retained and correctly interpreted

Six experiments:
  Exp 1 — Full patch:       shift full expansion activations to recession mean
  Exp 2 — PC1-only:         shift only the PC1 component
  Exp 3 — Residual:         shift only the PC2-PC16 component
  Exp 4 — Ablation:         zero out expansion activations (null machine)
  Exp 5 — SAE features:     patch SAE features 6, 25, 23
  Exp 6 — Placebo full:     random direction, same norm as full shift
  Exp 7 — Placebo residual: random direction orthogonal to PC1, same norm as residual

Effect measure: mean(output_patched - output_original) / std(output_original)
  Numerator:   mean shift in predicted return across N_SAMPLE stocks
  Denominator: std of unpatched predictions on same stocks
  Interpretation: effect in units of prediction standard deviations
  Target: |effect| > 0.10 for Exp 1 (economically meaningful shift)

Bootstrap 95% CI across 1,000 draws of N_SAMPLE expansion stocks.
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

N_BOOTSTRAP      = 1_000
N_SAMPLE         = 500
SEED             = 42
SAE_CRISIS_FEATS = [6, 25, 23]

# ── GKX MLP ───────────────────────────────────────────────────────────────────
HIDDEN  = [32, 16, 8]
DROPOUT = 0.5

class GKXMLP(nn.Module):
    def __init__(self, n_chars=94, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers, prev = [], n_chars
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def forward_from_layer1(self, h1):
        """Resume forward pass from post-ReLU layer1 activations.
        Layer indices: 0-3=layer0, 4-7=layer1, 8-11=layer2, 12=output linear
        Post-layer1-ReLU = index 6. Resume at index 7 (dropout after layer1 ReLU).
        """
        x = h1
        for idx in range(7, len(self.net)):
            x = self.net[idx](x)
        return x.squeeze(-1)


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GKX model...")
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
SKIP  = ['permno','yyyymm','ret_adj','vol','siccd','sic2','prc','shrout','exchcd','shrcd']
char_cols = [c for c in panel.columns if c not in SKIP]
print(f"  {len(char_cols)} characteristics")

model = GKXMLP(n_chars=len(char_cols)).to(DEVICE)
ckpts = sorted(MODEL_DIR.glob(f'checkpoint_*_seed{SEED}.pt'))
if not ckpts:
    ckpts = sorted(MODEL_DIR.glob('checkpoint_*.pt'))
ckpt = ckpts[-1]
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print(f"  Loaded: {ckpt.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading layer1 activations...")
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

acts_list, regime_list, permno_list = [], [], []
for f in sorted(ACTS_DIR.glob('acts_*.h5')):
    yyyymm = int(f.stem.replace('acts_', ''))
    if yyyymm not in regime_map or pd.isna(regime_map[yyyymm]):
        continue
    with h5py.File(f, 'r') as hf:
        acts_list.append(hf['layer1'][:])
        regime_list.append(np.full(hf['layer1'].shape[0],
                                   regime_map[yyyymm], dtype=np.float32))
        permno_list.append(hf['permno'][:])

acts_np   = np.vstack(acts_list).astype(np.float32)
regime_np = np.concatenate(regime_list)
permno_np = np.concatenate(permno_list)

acts_mu  = acts_np.mean(0, keepdims=True)
acts_sig = acts_np.std(0, keepdims=True) + 1e-8
acts_norm = (acts_np - acts_mu) / acts_sig

exp_mask = regime_np == 0
rec_mask = regime_np == 1
print(f"  Expansion: {exp_mask.sum():,}  Recession: {rec_mask.sum():,}")

# Mean shift vector
mean_rec = acts_norm[rec_mask].mean(0)
mean_exp = acts_norm[exp_mask].mean(0)
shift    = mean_rec - mean_exp


# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

print("\nFitting PCA...")
pca = PCA(n_components=16)
pca.fit(acts_norm[exp_mask][:50_000])
pc1 = pca.components_[0]

shift_pc1   = (shift @ pc1) * pc1
shift_resid = shift - shift_pc1
frac_pc1    = np.linalg.norm(shift_pc1) / np.linalg.norm(shift)

print(f"  PC1 variance:  {pca.explained_variance_ratio_[0]:.3f}")
print(f"  Shift total:   {np.linalg.norm(shift):.4f}")
print(f"  Shift in PC1:  {np.linalg.norm(shift_pc1):.4f}  ({frac_pc1*100:.1f}%)")
print(f"  Shift residual:{np.linalg.norm(shift_resid):.4f}  ({(1-frac_pc1)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAE FOR EXP 5
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

print("\nRetraining SAE (lam=0.6, 50ep)...")
X_exp = acts_norm[exp_mask]
sae   = SAE().to(DEVICE)
opt   = torch.optim.Adam(sae.parameters(), lr=1e-3)
X_t   = torch.tensor(X_exp, dtype=torch.float32)
torch.manual_seed(2026)
for epoch in range(50):
    idx = torch.randperm(len(X_t))
    for s in range(0, len(X_t), 256):
        xb = X_t[idx[s:s+256]].to(DEVICE)
        opt.zero_grad()
        r, z = sae(xb)
        (nn.MSELoss()(r, xb) + 0.6 * z.abs().mean()).backward()
        opt.step()
sae.eval()

with torch.no_grad():
    _, z_rec = sae(torch.tensor(acts_norm[rec_mask][:50_000],
                                dtype=torch.float32).to(DEVICE))
mean_z_rec = z_rec.cpu().numpy().mean(0)
print(f"  SAE ready.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PATCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def predict_from_layer1(h1_norm):
    """h1_norm: (n,16) normalized → de-normalize → forward from layer1."""
    h1_orig = h1_norm * acts_sig + acts_mu
    h1_t    = torch.tensor(h1_orig, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return model.forward_from_layer1(h1_t).cpu().numpy()


def patch_effect(h1_norm, delta):
    """Compute mean(pred_patched - pred_orig) / std(pred_orig)."""
    pred_orig   = predict_from_layer1(h1_norm)
    pred_patched = predict_from_layer1(h1_norm + delta[None, :])
    std_orig     = pred_orig.std() + 1e-8
    return float((pred_patched - pred_orig).mean() / std_orig), pred_orig


def patch_effect_sae(h1_norm, feat_indices):
    """Patch SAE features to recession mean, decode back to activation space."""
    h1_t = torch.tensor(h1_norm, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, z = sae(h1_t)
        z_p  = z.clone()
        for fi in feat_indices:
            z_p[:, fi] = float(mean_z_rec[fi])
        h1_recon = sae.dec(z_p).cpu().numpy()
    # h1_recon is in normalized space
    pred_orig    = predict_from_layer1(h1_norm)
    pred_patched = predict_from_layer1(h1_recon)
    std_orig     = pred_orig.std() + 1e-8
    return float((pred_patched - pred_orig).mean() / std_orig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Bootstrap ({N_BOOTSTRAP} draws, N={N_SAMPLE}/draw) ──────────────────")
exp_idx = np.where(exp_mask)[0]
results = {f'exp{i}': [] for i in range(1, 8)}

for b in range(N_BOOTSTRAP):
    if b % 200 == 0:
        print(f"  {b}/{N_BOOTSTRAP}...")

    sample = np.random.choice(exp_idx, N_SAMPLE, replace=True)
    h1     = acts_norm[sample]

    # Exp 1: full shift
    e1, _ = patch_effect(h1, shift)
    results['exp1'].append(e1)

    # Exp 2: PC1-only
    e2, _ = patch_effect(h1, shift_pc1)
    results['exp2'].append(e2)

    # Exp 3: residual (PC2-PC16)
    e3, _ = patch_effect(h1, shift_resid)
    results['exp3'].append(e3)

    # Exp 4: ablation — zero out activations (null machine baseline)
    pred_orig = predict_from_layer1(h1)
    pred_zero = predict_from_layer1(np.zeros_like(h1))
    std_orig  = pred_orig.std() + 1e-8
    results['exp4'].append(float((pred_zero - pred_orig).mean() / std_orig))

    # Exp 5: SAE feature patching
    e5 = patch_effect_sae(h1, SAE_CRISIS_FEATS)
    results['exp5'].append(e5)

    # Exp 6: placebo — random direction, same norm as full shift
    rng = np.random.default_rng(b)
    rd  = rng.standard_normal(16)
    rd  = rd / np.linalg.norm(rd) * np.linalg.norm(shift)
    e6, _ = patch_effect(h1, rd)
    results['exp6'].append(e6)

    # Exp 7: placebo — random direction orthogonal to PC1, same norm as residual
    ro  = rng.standard_normal(16)
    ro  = ro - (ro @ pc1) * pc1
    n_ro = np.linalg.norm(ro)
    if n_ro > 1e-8:
        ro  = ro / n_ro * np.linalg.norm(shift_resid)
        e7, _ = patch_effect(h1, ro)
    else:
        e7 = 0.0
    results['exp7'].append(e7)


# ══════════════════════════════════════════════════════════════════════════════
# 6. REPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("2C ACTIVATION PATCHING — RESULTS")
print(f"{'='*70}")
print(f"\nEffect measure: mean(pred_patched - pred_orig) / std(pred_orig)")
print(f"Interpretation: shift in predicted return, in units of prediction std")
print(f"Target |effect| > 0.10 for Exp 1 (economically meaningful)\n")

labels = [
    ('Exp 1: Full shift (baseline)',               'exp1'),
    ('Exp 2: PC1-only shift',                      'exp2'),
    ('Exp 3: Residual shift (PC2-PC16)',            'exp3'),
    ('Exp 4: Ablation (zero activations)',          'exp4'),
    ('Exp 5: SAE features 6/25/23',                'exp5'),
    ('Exp 6: Placebo — random, same norm',         'exp6'),
    ('Exp 7: Placebo — rand ortho PC1',            'exp7'),
]

rows = []
print(f"  {'Experiment':<42} {'Mean':>7} {'CI_lo':>7} {'CI_hi':>7}")
print(f"  {'-'*66}")
for lbl, key in labels:
    v    = np.array(results[key])
    mean = float(v.mean())
    lo   = float(np.percentile(v, 2.5))
    hi   = float(np.percentile(v, 97.5))
    print(f"  {lbl:<42} {mean:>+7.3f} {lo:>+7.3f} {hi:>+7.3f}")
    rows.append({'experiment': lbl, 'mean': round(mean,4),
                 'ci_lo': round(lo,4), 'ci_hi': round(hi,4)})

# ── Key diagnostics ───────────────────────────────────────────────────────────
e1 = np.array(results['exp1']).mean()
e2 = np.array(results['exp2']).mean()
e3 = np.array(results['exp3']).mean()
e6 = np.array(results['exp6']).mean()
e7 = np.array(results['exp7']).mean()

print(f"\n── Key diagnostics ──────────────────────────────────────────────────")

# 1. Does patching move output at all?
print(f"\n  1. Does patching matter? (Exp 1 vs Exp 4 ablation)")
print(f"     Full shift effect:  {e1:+.3f}")
print(f"     Ablation effect:    {np.array(results['exp4']).mean():+.3f}")

# 2. Where does the effect live?
if abs(e1) > 1e-6:
    print(f"\n  2. Where does the effect live?")
    print(f"     PC1 share of full effect:      {e2/e1*100:.1f}%")
    print(f"     Residual share:                {e3/e1*100:.1f}%")
    if abs(e3) > abs(e2):
        print(f"     → OUTPUT-SENSITIVE DIRECTIONS ARE IN THE RESIDUAL (PC2+)")
        print(f"       The dominant variance direction (PC1) is not the causal pathway.")
        print(f"       The network routes computation through low-variance directions.")
    else:
        print(f"     → PC1 is the dominant causal direction.")

# 3. Placebo check
print(f"\n  3. Is the effect direction-specific or magnitude-driven?")
print(f"     Exp 3 (recession residual):      {e3:+.3f}")
print(f"     Exp 7 (random, same norm):       {e7:+.3f}")
if abs(e3) > 2 * abs(e7):
    print(f"     ✓ Residual patch >> random ortho. Direction is specific.")
elif abs(e3) < 1.5 * abs(e7):
    print(f"     ~ Residual ≈ random. Residual effect may be magnitude-driven.")

# 4. Sign consistency
e1_lo = float(np.percentile(results['exp1'], 2.5))
e1_hi = float(np.percentile(results['exp1'], 97.5))
print(f"\n  4. Sign and significance (Exp 1):")
print(f"     Effect: {e1:+.3f}  CI: [{e1_lo:+.3f}, {e1_hi:+.3f}]")
if e1_lo * e1_hi > 0:
    print(f"     ✓ CI does not cross zero — consistent direction.")
else:
    print(f"     ~ CI crosses zero — noisy estimate.")

print(f"\n{'='*70}")
print("PHASE 2 CHECKPOINT")
print(f"{'='*70}")
print(f"\n  Target: |Exp 1 effect| > 0.10 (1/10th of prediction std)")
if abs(e1) > 0.10:
    print(f"  ✓ PASS — effect = {e1:+.3f}")
    print(f"    Causal evidence: patching layer1 activations to recession mean")
    print(f"    shifts predictions by {abs(e1):.2f} standard deviations.")
else:
    print(f"  ✗ FAIL — effect = {e1:+.3f} < 0.10")
    print(f"    Patching has minimal causal effect on outputs.")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(rows).to_csv(DATA_DIR / '2c_patching_results.csv', index=False)
pd.DataFrame({
    'neuron':         list(range(16)),
    'shift_total':    shift,
    'shift_pc1':      shift_pc1,
    'shift_residual': shift_resid,
    'pc1_loading':    pc1,
}).to_csv(DATA_DIR / '2c_shift_decomposition.csv', index=False)
print(f"\n  Saved 2c_patching_results.csv and 2c_shift_decomposition.csv")
print(f"\nNext: interpret results, then Phase 3 if effect > 0.10.")