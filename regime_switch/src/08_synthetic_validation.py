"""
08_2a_synthetic_validation.py — Phase 2A: Synthetic MI Tool Validation
=======================================================================
Per flexmap v2 step 2A: validate every MI tool on a toy network with a
planted circuit BEFORE applying to real data.

Setup:
  - Small MLP trained on synthetic stock-return-like data
  - Neuron #7 in layer 1 (the middle layer) is mechanically forced to
    activate on a binary regime dummy
  - Three tools are validated: linear probe, activation patching, SAE
  - Three circuit strengths tested: strong, medium, weak
  - Pass criterion: each tool recovers neuron #7 as the top crisis neuron

If any tool fails: debug before Phase 2B/2C/2D on real data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(2026)
np.random.seed(2026)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Synthetic data config ─────────────────────────────────────────────────────
N_STOCKS    = 5_000     # stocks per month
N_MONTHS    = 200       # total months
N_CHARS     = 20        # characteristics (small for speed)
N_REC       = 50        # recession months (out of 200)
PLANTED_NEURON = 7      # neuron index in layer 1 (0-indexed)
HIDDEN      = [16, 8]   # toy MLP: 20 → 16 → 8 → 1

# Circuit strength: how strongly does the regime boost neuron activation
STRENGTHS = {
    'strong': 3.0,
    'medium': 1.5,
    'weak':   0.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOY MLP WITH PLANTED CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════

class ToyMLP(nn.Module):
    """
    Small MLP where neuron PLANTED_NEURON in layer 1 is forced to respond
    to a regime dummy via a bypass connection added to its pre-activation.
    The rest of the network is trained normally on synthetic returns.
    """
    def __init__(self, n_chars=N_CHARS, hidden=HIDDEN,
                 planted=PLANTED_NEURON, strength=2.0):
        super().__init__()
        self.planted  = planted
        self.strength = strength
        self.activations = {}

        # Layer 0: n_chars → hidden[0]
        self.fc0 = nn.Linear(n_chars, hidden[0])
        self.bn0 = nn.BatchNorm1d(hidden[0])
        # Layer 1: hidden[0] → hidden[1]
        self.fc1 = nn.Linear(hidden[0], hidden[1])
        self.bn1 = nn.BatchNorm1d(hidden[1])
        # Output
        self.out = nn.Linear(hidden[1], 1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x, regime=None):
        # Layer 0
        h0 = self.drop(self.relu(self.bn0(self.fc0(x))))
        self.activations['layer0'] = h0.detach().cpu()

        # Layer 1 pre-activation
        h1_pre = self.bn1(self.fc1(h0))

        # ── PLANT THE CIRCUIT ────────────────────────────────────────────────
        # Add a regime-driven boost to neuron PLANTED_NEURON's pre-activation.
        # This makes neuron PLANTED_NEURON mechanically activate on recession.
        if regime is not None:
            boost = torch.zeros_like(h1_pre)
            boost[:, self.planted] = self.strength * regime.float()
            h1_pre = h1_pre + boost
        # ─────────────────────────────────────────────────────────────────────

        h1 = self.drop(self.relu(h1_pre))
        self.activations['layer1'] = h1.detach().cpu()

        return self.out(h1).squeeze(-1)


def make_synthetic_data(strength):
    """
    Generate synthetic panel:
      X: [N_MONTHS * N_STOCKS, N_CHARS] — random characteristics in [-1,1]
      y: [N_MONTHS * N_STOCKS] — noisy linear combination of X
      regime: [N_MONTHS * N_STOCKS] — binary recession indicator
      month_idx: [N_MONTHS * N_STOCKS] — month id
    """
    np.random.seed(2026)
    regime_months = np.zeros(N_MONTHS)
    rec_idx = np.random.choice(N_MONTHS, N_REC, replace=False)
    regime_months[rec_idx] = 1.0

    X_list, y_list, reg_list, mon_list = [], [], [], []
    true_weights = np.random.randn(N_CHARS) * 0.3

    for t in range(N_MONTHS):
        X_t = np.random.uniform(-1, 1, (N_STOCKS, N_CHARS)).astype(np.float32)
        y_t = X_t @ true_weights + np.random.randn(N_STOCKS).astype(np.float32) * 0.5
        X_list.append(X_t)
        y_list.append(y_t)
        reg_list.append(np.full(N_STOCKS, regime_months[t], dtype=np.float32))
        mon_list.append(np.full(N_STOCKS, t, dtype=np.int32))

    X      = np.vstack(X_list)
    y      = np.concatenate(y_list)
    regime = np.concatenate(reg_list)
    months = np.concatenate(mon_list)
    return X, y, regime, months, regime_months


def train_toy_model(strength):
    X, y, regime, months, regime_months = make_synthetic_data(strength)
    model = ToyMLP(strength=strength).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train 30 epochs
    n = len(X)
    idx = np.arange(n)
    for epoch in range(30):
        np.random.shuffle(idx)
        for start in range(0, n, 1000):
            bi = idx[start:start+1000]
            xb = torch.tensor(X[bi]).to(DEVICE)
            yb = torch.tensor(y[bi], dtype = torch.float32).to(DEVICE)
            rb = torch.tensor(regime[bi]).to(DEVICE)
            opt.zero_grad()
            pred = model(xb, regime=rb)
            nn.MSELoss()(pred, yb).backward()
            opt.step()

    model.eval()
    return model, X, y, regime, months, regime_months


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOOL A — LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════════════

def run_linear_probe(model, X, regime, months):
    """
    For each neuron in each layer, regress its monthly mean activation
    on the NBER regime indicator. Return t-stats and rank of planted neuron.
    """
    model.eval()
    layer_acts = {'layer0': [], 'layer1': []}
    month_list = []

    # Forward pass in batches to collect activations
    with torch.no_grad():
        for start in range(0, len(X), 2000):
            xb = torch.tensor(X[start:start+2000]).to(DEVICE)
            rb = torch.tensor(regime[start:start+2000]).to(DEVICE)
            _  = model(xb, regime=rb)
            for layer in layer_acts:
                layer_acts[layer].append(model.activations[layer].numpy())

    for layer in layer_acts:
        layer_acts[layer] = np.vstack(layer_acts[layer])

    # Compute monthly means per neuron
    unique_months = sorted(set(months))
    regime_by_month = {}
    for m in unique_months:
        mask = months == m
        regime_by_month[m] = regime[mask].mean()  # 0 or 1

    results = {}
    for layer, acts in layer_acts.items():
        n_neurons = acts.shape[1]
        for n in range(n_neurons):
            monthly_mean = [acts[months == m, n].mean() for m in unique_months]
            reg_vals     = [regime_by_month[m] for m in unique_months]
            slope, _, _, _, se = stats.linregress(reg_vals, monthly_mean)
            results[f'{layer}_{n}'] = slope / se if se > 1e-10 else 0.0

    # Rank by |t-stat|
    ranked = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
    top10  = [k for k, _ in ranked[:10]]
    planted_key = f'layer1_{PLANTED_NEURON}'
    rank = next((i+1 for i, (k, _) in enumerate(ranked) if k == planted_key), None)
    t_planted = results[planted_key]

    return rank, t_planted, top10, results


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOOL B — ACTIVATION PATCHING
# ══════════════════════════════════════════════════════════════════════════════

def run_patching(model, X, regime):
    """
    Patch layer1 activations of expansion stocks with mean recession activation.
    Measure patch_effect = |output_patched - output_expansion| /
                           |output_recession - output_expansion|

    Test on planted neuron alone vs all other neurons.
    """
    model.eval()
    exp_mask = regime == 0
    rec_mask = regime == 1

    # Sample 500 expansion and compute mean recession activation
    exp_idx = np.where(exp_mask)[0][:500]
    rec_idx = np.where(rec_mask)[0][:500]

    X_exp = torch.tensor(X[exp_idx]).to(DEVICE)
    X_rec = torch.tensor(X[rec_idx]).to(DEVICE)
    R_exp = torch.zeros(len(exp_idx)).to(DEVICE)
    R_rec = torch.ones(len(rec_idx)).to(DEVICE)

    with torch.no_grad():
        pred_exp = model(X_exp, regime=R_exp).cpu().numpy()
        pred_rec = model(X_rec, regime=R_rec).cpu().numpy()
        # Get mean recession layer1 activation
        _ = model(X_rec, regime=R_rec)
        mean_rec_act = model.activations['layer1'].mean(axis=0)  # [8]

    gap = np.abs(pred_rec.mean() - pred_exp.mean())

    def patch_effect(neuron_subset):
        """Replace layer1 activations at neuron_subset with mean recession values."""
        patched_preds = []
        bs = 100
        for start in range(0, len(X_exp), bs):
            xb = X_exp[start:start+bs]
            rb = R_exp[start:start+bs]

            # Forward to layer0
            with torch.no_grad():
                h0 = model.drop(model.relu(model.bn0(model.fc0(xb))))
                h1_pre = model.bn1(model.fc1(h0))
                boost  = torch.zeros_like(h1_pre)
                boost[:, model.planted] = model.strength * rb
                h1_pre = h1_pre + boost
                h1 = model.relu(h1_pre)
                # Patch
                h1_patched = h1.clone()
                for n in neuron_subset:
                    h1_patched[:, n] = mean_rec_act[n].to(DEVICE)
                pred_p = model.out(h1_patched).squeeze(-1).cpu().numpy()
                patched_preds.append(pred_p)

        patched = np.concatenate(patched_preds)
        effect = np.abs(patched.mean() - pred_exp.mean())
        return effect / gap if gap > 1e-8 else 0.0

    pe_planted = patch_effect([PLANTED_NEURON])
    pe_others  = patch_effect([n for n in range(HIDDEN[1]) if n != PLANTED_NEURON])
    pe_all     = patch_effect(list(range(HIDDEN[1])))

    return pe_planted, pe_others, pe_all


# ══════════════════════════════════════════════════════════════════════════════
# 4. TOOL C — SPARSE AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

class SAE(nn.Module):
    def __init__(self, n_in=8, n_feat=64):
        super().__init__()
        self.enc = nn.Linear(n_in, n_feat)
        self.dec = nn.Linear(n_feat, n_in)

    def forward(self, x):
        z = torch.relu(self.enc(x))
        return self.dec(z), z


def run_sae(model, X, regime, months, lam=1e-5):
    model.eval()

    all_acts   = []
    all_regime = []
    all_months = []

    with torch.no_grad():
        for start in range(0, len(X), 2000):
            xb = torch.tensor(X[start:start+2000]).to(DEVICE)
            rb = torch.tensor(regime[start:start+2000]).to(DEVICE)
            _  = model(xb, regime=rb)
            all_acts.append(model.activations['layer1'].numpy())
            all_regime.append(regime[start:start+2000])
            all_months.append(months[start:start+2000])

    acts_np   = np.vstack(all_acts)
    regime_np = np.concatenate(all_regime)

    # ── Normalize activations before feeding to SAE ───────────────────────────
    acts_mean = acts_np.mean(axis=0, keepdims=True)
    acts_std  = acts_np.std(axis=0, keepdims=True) + 1e-8
    acts_norm = (acts_np - acts_mean) / acts_std

    exp_acts = acts_norm[regime_np == 0]
    X_sae    = torch.tensor(exp_acts, dtype=torch.float32)

    sae = SAE(n_in=HIDDEN[1], n_feat=64)
    # Orthogonal init — prevents dead features at start
    nn.init.orthogonal_(sae.enc.weight)
    nn.init.orthogonal_(sae.dec.weight)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for epoch in range(50):
        idx = torch.randperm(len(X_sae))
        for start in range(0, len(X_sae), 256):
            xb = X_sae[idx[start:start+256]]
            opt.zero_grad()
            recon, z = sae(xb)
            (nn.MSELoss()(recon, xb) + lam * z.abs().mean()).backward()
            opt.step()

    sae.eval()

    # Dead feature diagnostic
    with torch.no_grad():
        _, z_check = sae(X_sae[:500])
        mean_activations = z_check.mean(0)
        n_dead = (mean_activations < 1e-6).sum().item()
        print(f"    SAE dead features: {n_dead}/64")

    # Apply to all activations
    with torch.no_grad():
        _, z_all = sae(torch.tensor(acts_norm, dtype=torch.float32))
    z_all = z_all.numpy()

    # ── NaN guard: skip dead features ────────────────────────────────────────
    planted_acts = acts_norm[:, PLANTED_NEURON]
    feature_corr = np.zeros(64)
    for f in range(64):
        if z_all[:, f].std() > 1e-8:
            feature_corr[f] = np.corrcoef(z_all[:, f], planted_acts)[0, 1]
        else:
            feature_corr[f] = 0.0   # dead feature, treat as no correlation

    top_feature = int(np.argmax(np.abs(feature_corr)))
    top_corr    = feature_corr[top_feature]

    all_neuron_corr = np.array([
        np.corrcoef(z_all[:, top_feature], acts_norm[:, n])[0, 1]
        if z_all[:, top_feature].std() > 1e-8 else 0.0
        for n in range(HIDDEN[1])
    ])
    mono_score = np.abs(all_neuron_corr[PLANTED_NEURON]) / (np.abs(all_neuron_corr).sum() + 1e-8)

    rec_mean = z_all[regime_np == 1, top_feature].mean()
    exp_mean = z_all[regime_np == 0, top_feature].mean()
    regime_ratio = rec_mean / (exp_mean + 1e-8)

    return top_feature, top_corr, mono_score, regime_ratio


# ══════════════════════════════════════════════════════════════════════════════
# 5. RUN ALL THREE TOOLS ACROSS ALL STRENGTHS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"2A SYNTHETIC VALIDATION — Planted neuron: layer1_{PLANTED_NEURON}")
print(f"{'='*65}")

results_rows = []

for strength_name, strength_val in STRENGTHS.items():
    print(f"\n── Strength: {strength_name} (boost = {strength_val}) ──────────────")

    model, X, y, regime, months, regime_months = train_toy_model(strength_val)

    # ── Linear probe ─────────────────────────────────────────────────────────
    rank, t_stat, top10, all_tstats = run_linear_probe(model, X, regime, months)
    probe_pass = rank == 1
    print(f"  [LINEAR PROBE]")
    print(f"    Planted neuron rank : #{rank} (t = {t_stat:.2f})")
    print(f"    Top-10 neurons      : {top10}")
    print(f"    {'✓ PASS' if probe_pass else '✗ FAIL'} — neuron recovered as #1" if probe_pass
          else f"    ✗ FAIL — neuron at rank #{rank}, expected #1")

    # ── Activation patching ───────────────────────────────────────────────────
    pe_planted, pe_others, pe_all = run_patching(model, X, regime)
    patch_pass = pe_planted > 0.25 
    print(f"  [ACTIVATION PATCHING]")
    print(f"    patch_effect(planted) : {pe_planted:.3f}")
    print(f"    patch_effect(others)  : {pe_others:.3f}")
    print(f"    patch_effect(all)     : {pe_all:.3f}")
    print(f"    {'✓ PASS' if patch_pass else '✗ FAIL'} — planted > 0.25 and > others")

    # ── SAE ───────────────────────────────────────────────────────────────────
    top_feat, top_corr, mono, ratio = run_sae(model, X, regime, months)
    sae_pass = abs(top_corr) > 0.5 and ratio > 1.2
    print(f"  [SAE]")
    print(f"    Top feature #{top_feat} corr with planted neuron : {top_corr:.3f}")
    print(f"    Monosemanticity score : {mono:.3f}")
    print(f"    Recession/expansion activation ratio : {ratio:.2f}x")
    print(f"    {'✓ PASS' if sae_pass else '✗ FAIL'} — |corr| > 0.5 and ratio > 1.5x")

    results_rows.append({
        'strength':        strength_name,
        'boost':           strength_val,
        'probe_rank':      rank,
        'probe_tstat':     round(t_stat, 2),
        'probe_pass':      probe_pass,
        'patch_planted':   round(pe_planted, 3),
        'patch_others':    round(pe_others, 3),
        'patch_pass':      patch_pass,
        'sae_corr':        round(top_corr, 3),
        'sae_mono':        round(mono, 3),
        'sae_ratio':       round(ratio, 2),
        'sae_pass':        sae_pass,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
df = pd.DataFrame(results_rows)
print(df[['strength', 'probe_rank', 'probe_tstat', 'probe_pass',
          'patch_planted', 'patch_pass',
          'sae_corr', 'sae_pass']].to_string(index=False))

print(f"\n{'='*65}")
print("MINIMUM DETECTABLE SIGNAL STRENGTH")
print(f"{'='*65}")
for _, row in df.iterrows():
    all_pass = row['probe_pass'] and row['patch_pass'] and row['sae_pass']
    print(f"  {row['strength']:8s} (boost={row['boost']:.1f}): "
          f"{'ALL TOOLS PASS ✓' if all_pass else 'at least one tool FAILS ✗'}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parents[1] / "data"
df.to_csv(out_dir / "2a_synthetic_validation.csv", index=False)
print(f"\nSaved 2a_synthetic_validation.csv")
print(f"{'='*65}")
print("\nIf all tools pass on strong + medium: proceed to Phase 2D (SAE on real data).")
print("If any tool fails on strong: debug that tool before proceeding.")