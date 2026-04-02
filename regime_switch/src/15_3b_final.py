"""
15_3b_final.py — Phase 3B: Ablation Tests — Path A (FIXED FINAL MODEL)
=======================================================================
Per Phase 3A pre-registration memo (signed 2026-04-01).

PATH A: MI analysis of the fully trained 2023 GKX model.
  This is the standard MI approach — study a fixed trained model.
  All activations are computed fresh using checkpoint_202301_seed42.pt,
  not loaded from HDF5 files (which used contemporaneous checkpoints).
  Encoder and decoder are both from the 2023 model → fully consistent.

  Scientific question: "What does the fully trained GKX model compute,
  and is Circuit C causally necessary for its recession performance?"

  NOTE: The OOS Sharpe (1.609) comes from Phase 1's rolling procedure.
  This MI analysis is separate — it studies the architecture of the
  final trained model, not the real-time trading performance.
  No look-ahead concern: we are deliberately asking how the final model
  is structured, not whether a trader could have used it in 1990.

Compare with 15_3b_rolling.py (Path B):
  Path B uses contemporaneous checkpoints per month — answers whether
  the PC1 counterproductivity holds across all training vintages.
  Path B H2p (DiD=-0.953) is the cross-vintage generalization.
  Path A H1 is the specific MI claim about the 2023 model's structure.

Three pre-registered ablations (H1, H2, H3) + H2p permutation test.
Bootstrap: 1,000 month-level draws.
Leave-one-recession-out and 20 placebos included.
Primary statistic: DiD = ΔSharpe_NBER − ΔSharpe_Exp.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(2026)
np.random.seed(2026)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BASE      = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
DATA_DIR  = BASE / 'data'
MODEL_DIR = BASE / 'models'

SEED          = 42
N_BOOTSTRAP   = 1_000
N_PLACEBOS    = 20
PRIMARY_START = 200501
FULL_START    = 198701
BATCH_SIZE    = 10_000

HIDDEN  = [32, 16, 8]
DROPOUT = 0.5

# ── GKX MLP with layer-1 activation capture ───────────────────────────────────
class GKXMLP(nn.Module):
    def __init__(self, n_chars=94):
        super().__init__()
        self._layer1_acts = None
        layers, prev = [], n_chars
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        # Hook to capture layer1 post-ReLU activations (index 6)
        self.net[6].register_forward_hook(self._capture_layer1)

    def _capture_layer1(self, module, inp, out):
        self._layer1_acts = out.detach()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def forward_from_layer1(self, h1):
        """Resume forward pass from post-ReLU layer1 activations."""
        x = h1
        for idx in range(7, len(self.net)):
            x = self.net[idx](x)
        return x.squeeze(-1)

    def get_layer1_and_predict(self, x):
        """Full forward pass: returns (predictions, layer1_activations)."""
        preds = self.forward(x)
        return preds, self._layer1_acts.clone()


# ── Load 2023 model ───────────────────────────────────────────────────────────
print("Loading final 2023 GKX model...")
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet')
SKIP  = ['permno','yyyymm','ret_adj','vol','siccd','sic2','prc','shrout','exchcd','shrcd']
char_cols = [c for c in panel.columns if c not in SKIP]
panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno']  = panel['permno'].astype(int)

model = GKXMLP(n_chars=len(char_cols)).to(DEVICE)
ckpt  = MODEL_DIR / f'checkpoint_202301_seed{SEED}.pt'
if not ckpt.exists():
    ckpts = sorted(MODEL_DIR.glob(f'checkpoint_*_seed{SEED}.pt'))
    ckpt  = ckpts[-1]
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print(f"  Loaded: {ckpt.name}  (Path A uses this model for ALL months)")

# Verify BN eval mode
bn_eval = all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d))
print(f"  All BN layers in eval mode: {bn_eval}")


# ── Load ablation directions ───────────────────────────────────────────────────
df_shift    = pd.read_csv(DATA_DIR / '2c_shift_decomposition.csv')
shift_resid = df_shift['shift_residual'].values
pc1_loading = df_shift['pc1_loading'].values
unit_C = shift_resid / np.linalg.norm(shift_resid)
unit_R = pc1_loading  / np.linalg.norm(pc1_loading)
print(f"  Cosine(C, R) = {np.dot(unit_C, unit_R):.6f}  (should be ~0)")

# 20 placebo directions orthogonal to PC1
rng_p = np.random.default_rng(2026)
placebo_dirs = []
for _ in range(N_PLACEBOS):
    rd = rng_p.standard_normal(16)
    rd = rd - (rd @ unit_R) * unit_R
    rd = rd / np.linalg.norm(rd)
    placebo_dirs.append(rd)


# ── Load regime labels ────────────────────────────────────────────────────────
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

all_months = sorted(panel['yyyymm'].unique())
test_months = [m for m in all_months if m >= FULL_START]


# ── FRESH ACTIVATION COMPUTATION via 2023 model ───────────────────────────────
print(f"\nComputing fresh layer-1 activations via 2023 model...")
print(f"  (Not using stored HDF5 — encoder and decoder both from 2023 model)")

# Normalization stats will be computed from fresh activations
# First pass: collect all fresh activations for normalization
all_fresh_acts  = []
month_char_data = {}  # cache for efficiency

print(f"  Pass 1: collect activations for normalization stats...")
for t_idx, yyyymm in enumerate(test_months):
    mdata = panel[panel['yyyymm'] == yyyymm]
    if len(mdata) < 50:
        continue
    X = mdata[char_cols].values.astype(np.float32)
    permnos = mdata['permno'].values
    returns = mdata['ret_adj'].values.astype(np.float32)
    month_char_data[yyyymm] = (X, permnos, returns)

    # Collect layer-1 acts in small batches for memory
    acts_parts = []
    for s in range(0, len(X), BATCH_SIZE):
        xb = torch.tensor(X[s:s+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            _, acts_b = model.get_layer1_and_predict(xb)
        acts_parts.append(acts_b.cpu().numpy())
        del xb
    month_acts = np.vstack(acts_parts).astype(np.float32)

    # Only keep for normalization (discard individual months, too much memory)
    if t_idx % 20 == 0:  # sample every 20th month for norm stats
        all_fresh_acts.append(month_acts)

    if t_idx % 50 == 0:
        print(f"    {t_idx}/{len(test_months)} months processed...")

all_fresh_np = np.vstack(all_fresh_acts)
ACTS_MU  = all_fresh_np.mean(0, keepdims=True)
ACTS_SIG = all_fresh_np.std(0, keepdims=True) + 1e-8
del all_fresh_acts, all_fresh_np
print(f"  Normalization stats computed from sampled months.")


# ── PREDICTION FUNCTIONS ──────────────────────────────────────────────────────
def get_fresh_layer1(X_raw):
    """Run 2023 model forward pass, return normalized layer-1 activations."""
    acts_parts = []
    for s in range(0, len(X_raw), BATCH_SIZE):
        xb = torch.tensor(X_raw[s:s+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            _, acts_b = model.get_layer1_and_predict(xb)
        acts_parts.append(acts_b.cpu().numpy())
        del xb
    acts = np.vstack(acts_parts).astype(np.float32)
    return (acts - ACTS_MU) / ACTS_SIG   # return normalized


def predict_from_layer1_norm(h1_norm, ablation_dir=None):
    """Forward from normalized layer-1 activations with optional ablation."""
    acts_n = h1_norm.copy()
    if ablation_dir is not None:
        proj   = (acts_n @ ablation_dir)[:, None] * ablation_dir[None, :]
        acts_n = acts_n - proj
    h1_orig = acts_n * ACTS_SIG + ACTS_MU
    h1_t    = torch.tensor(h1_orig, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model.forward_from_layer1(h1_t).cpu().numpy()
    return preds


def predict_permute(h1_norm, ablation_dir, rng_seed=0):
    """H2p: shuffle ablation_dir scores within month."""
    rng     = np.random.default_rng(rng_seed)
    acts_n  = h1_norm.copy()
    scores  = acts_n @ ablation_dir
    shuffled = rng.permutation(scores)
    acts_n  = (acts_n
               - scores[:, None]   * ablation_dir[None, :]
               + shuffled[:, None] * ablation_dir[None, :])
    h1_orig = acts_n * ACTS_SIG + ACTS_MU
    h1_t    = torch.tensor(h1_orig, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model.forward_from_layer1(h1_t).cpu().numpy()
    return preds


def monthly_sharpe(returns_list):
    r = np.array(returns_list)
    if len(r) < 3 or r.std() < 1e-8:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(12))


def ls_return_from_preds(preds, returns):
    n = len(preds)
    if n < 20:
        return np.nan
    d   = n // 10
    idx = np.argsort(preds)
    return float(returns[idx[-d:]].mean() - returns[idx[:d]].mean())


# ── COMPUTE MONTHLY L/S RETURNS ───────────────────────────────────────────────
print(f"\nPass 2: Computing monthly L/S returns under each ablation...")
print(f"  All using 2023 model for both encoding and decoding.")

ablation_keys = (['baseline', 'ablate_C', 'ablate_R', 'ablate_R_permute'] +
                 [f'placebo_{i}' for i in range(N_PLACEBOS)])
results_by_month = {k: {} for k in ablation_keys}

for t_idx, yyyymm in enumerate(test_months):
    if yyyymm not in regime_map or pd.isna(regime_map[yyyymm]):
        continue
    nber = int(regime_map[yyyymm])

    if yyyymm not in month_char_data:
        continue
    X_raw, permnos, returns = month_char_data[yyyymm]
    valid = ~np.isnan(returns)
    if valid.sum() < 50:
        continue
    X_raw   = X_raw[valid]
    returns = returns[valid]

    if t_idx % 50 == 0:
        print(f"  Month {t_idx}/{len(test_months)} ({yyyymm})...")

    # Get fresh layer-1 activations from 2023 model
    h1_norm = get_fresh_layer1(X_raw)

    # Baseline
    preds_base = predict_from_layer1_norm(h1_norm)
    results_by_month['baseline'][yyyymm] = {
        'ls_ret': ls_return_from_preds(preds_base, returns), 'nber': nber}

    # Circuit C ablation
    results_by_month['ablate_C'][yyyymm] = {
        'ls_ret': ls_return_from_preds(
            predict_from_layer1_norm(h1_norm, unit_C), returns),
        'nber': nber}

    # Circuit R ablation
    results_by_month['ablate_R'][yyyymm] = {
        'ls_ret': ls_return_from_preds(
            predict_from_layer1_norm(h1_norm, unit_R), returns),
        'nber': nber}

    # Circuit R permutation (H2p)
    results_by_month['ablate_R_permute'][yyyymm] = {
        'ls_ret': ls_return_from_preds(
            predict_permute(h1_norm, unit_R, rng_seed=t_idx), returns),
        'nber': nber}

    # 20 placebos
    for pi, pdir in enumerate(placebo_dirs):
        results_by_month[f'placebo_{pi}'][yyyymm] = {
            'ls_ret': ls_return_from_preds(
                predict_from_layer1_norm(h1_norm, pdir), returns),
            'nber': nber}

print(f"  All months processed.")


# ── STATS FUNCTIONS ───────────────────────────────────────────────────────────
def compute_stats(res_base, res_abl, start, label=''):
    months = sorted([m for m in res_base if m >= start])
    nb, na, eb, ea = [], [], [], []
    for m in months:
        b = res_base[m]; a = res_abl[m]
        if np.isnan(b['ls_ret']) or np.isnan(a['ls_ret']):
            continue
        if b['nber'] == 1:
            nb.append(b['ls_ret']); na.append(a['ls_ret'])
        else:
            eb.append(b['ls_ret']); ea.append(a['ls_ret'])
    sbn = monthly_sharpe(nb); san = monthly_sharpe(na)
    sbe = monthly_sharpe(eb); sae = monthly_sharpe(ea)
    dn = sbn - san; de = sbe - sae
    did = dn - de
    ratio = dn / (abs(de) + 1e-8) if abs(de) > 1e-4 else np.nan
    return {'label': label, 'start': start,
            'n_nber': len(nb), 'n_exp': len(eb),
            'sharpe_base_nber': round(sbn,3), 'sharpe_abl_nber': round(san,3),
            'sharpe_base_exp':  round(sbe,3), 'sharpe_abl_exp':  round(sae,3),
            'delta_nber': round(dn,3), 'delta_exp': round(de,3),
            'did': round(did,3),
            'ratio': round(ratio,3) if not np.isnan(ratio) else np.nan}


def bootstrap_did(res_base, res_abl, start, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(2026)
    months = sorted([m for m in res_base if m >= start])
    nm = [m for m in months if res_base[m]['nber']==1
          and not np.isnan(res_base[m]['ls_ret'])
          and not np.isnan(res_abl[m]['ls_ret'])]
    em = [m for m in months if res_base[m]['nber']==0
          and not np.isnan(res_base[m]['ls_ret'])
          and not np.isnan(res_abl[m]['ls_ret'])]
    if len(nm) < 3 or len(em) < 3:
        return None
    bd, bed, bdid, br = [], [], [], []
    for _ in range(n_boot):
        nb = rng.choice(nm, len(nm), replace=True)
        eb = rng.choice(em, len(em), replace=True)
        dn = (monthly_sharpe([res_base[m]['ls_ret'] for m in nb]) -
              monthly_sharpe([res_abl[m]['ls_ret']  for m in nb]))
        de = (monthly_sharpe([res_base[m]['ls_ret'] for m in eb]) -
              monthly_sharpe([res_abl[m]['ls_ret']  for m in eb]))
        bd.append(dn); bed.append(de); bdid.append(dn-de)
        if abs(de) > 1e-4:
            br.append(dn/abs(de))
    def ci(arr):
        a = np.array(arr)
        return float(a.mean()), float(np.median(a)), \
               float(np.percentile(a,2.5)), float(np.percentile(a,97.5))
    return {'dn': ci(bd), 'de': ci(bed), 'did': ci(bdid),
            'ratio': ci(br) if len(br) > 10 else None}


# ── MAIN RESULTS ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("3B FINAL — PATH A: MI ANALYSIS OF 2023 GKX MODEL")
print(f"{'='*70}")

all_stats = []
main_ablations = [
    ('H1: Circuit C (residual — CAUSAL)',   'ablate_C'),
    ('H2: Circuit R (PC1 — zero ablation)', 'ablate_R'),
    ('H2p: Circuit R (permutation test)',   'ablate_R_permute'),
]

for label, key in main_ablations:
    print(f"\n── {label} ──────────────────────────────────────")
    for start, pname in [(PRIMARY_START,'Primary (post-2005)'),
                          (FULL_START,  'Full OOS (post-1987)')]:
        s = compute_stats(results_by_month['baseline'],
                          results_by_month[key], start,
                          label=f"{key}_{pname}")
        print(f"\n  Period: {pname}  (NBER={s['n_nber']}, Exp={s['n_exp']})")
        print(f"  Sharpe NBER: {s['sharpe_base_nber']:.3f} → "
              f"{s['sharpe_abl_nber']:.3f}  Δ={s['delta_nber']:+.3f}")
        print(f"  Sharpe Exp:  {s['sharpe_base_exp']:.3f} → "
              f"{s['sharpe_abl_exp']:.3f}  Δ={s['delta_exp']:+.3f}")
        print(f"  DiD: {s['did']:+.3f}  Ratio: {s['ratio']}")

        if start == PRIMARY_START:
            boot = bootstrap_did(results_by_month['baseline'],
                                 results_by_month[key], start)
            if boot:
                dn = boot['dn']; de = boot['de']; dd = boot['did']
                print(f"  Bootstrap ΔSharpe_NBER: "
                      f"mean={dn[0]:+.3f} median={dn[1]:+.3f} "
                      f"CI=[{dn[2]:+.3f},{dn[3]:+.3f}]")
                print(f"  Bootstrap ΔSharpe_Exp:  "
                      f"mean={de[0]:+.3f} median={de[1]:+.3f} "
                      f"CI=[{de[2]:+.3f},{de[3]:+.3f}]")
                print(f"  Bootstrap DiD:          "
                      f"mean={dd[0]:+.3f} median={dd[1]:+.3f} "
                      f"CI=[{dd[2]:+.3f},{dd[3]:+.3f}]")
                s['boot_did_lo']     = round(dd[2],3)
                s['boot_did_hi']     = round(dd[3],3)
                s['boot_did_median'] = round(dd[1],3)
                if boot['ratio']:
                    rr = boot['ratio']
                    print(f"  Bootstrap ratio: median={rr[1]:.2f} "
                          f"CI=[{rr[2]:.2f},{rr[3]:.1f}]")
                if key == 'ablate_C':
                    did_lo = s.get('boot_did_lo', np.nan)
                    if not np.isnan(did_lo) and did_lo > 0:
                        print(f"  ✓ H1 STRONG PASS: DiD CI_lo={did_lo:+.3f} > 0")
                    elif s['did'] > 0:
                        print(f"  ~ H1 PASS: DiD positive, wide CI")
                    else:
                        print(f"  ✗ H1 FAIL: DiD <= 0")
        all_stats.append(s)


# ── LEAVE-ONE-RECESSION-OUT ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LEAVE-ONE-RECESSION-OUT (Path A, primary period)")
print(f"{'='*70}")

nber_months_primary = sorted([m for m, v in results_by_month['baseline'].items()
                               if v['nber']==1 and m >= PRIMARY_START])
episodes = []
if nber_months_primary:
    ep_s, ep_prev = nber_months_primary[0], nber_months_primary[0]
    for m in nber_months_primary[1:]:
        gap = ((m//100)-(ep_prev//100))*12 + (m%100)-(ep_prev%100)
        if gap <= 2:
            ep_prev = m
        else:
            episodes.append((ep_s, ep_prev))
            ep_s = ep_prev = m
    episodes.append((ep_s, ep_prev))

print(f"\n  Recession episodes: {episodes}")
s_full = compute_stats(results_by_month['baseline'],
                       results_by_month['ablate_C'], PRIMARY_START)
print(f"\n  {'Excluded':<22} {'N_NBER':>7} {'DiD':>8} {'Verdict'}")
print(f"  {'-'*50}")
print(f"  {'None (full sample)':<22} {s_full['n_nber']:>7} "
      f"{s_full['did']:>+8.3f}  baseline")
for ep_s, ep_e in episodes:
    rb = {m:v for m,v in results_by_month['baseline'].items()
          if not (ep_s<=m<=ep_e)}
    ra = {m:v for m,v in results_by_month['ablate_C'].items()
          if not (ep_s<=m<=ep_e)}
    sd = compute_stats(rb, ra, PRIMARY_START)
    verd = ("✓ survives" if sd['did'] > 0.5 else
            ("~ weakens" if sd['did'] > 0 else "✗ reverses"))
    print(f"  {ep_s}-{ep_e:<12}      {sd['n_nber']:>7} "
          f"{sd['did']:>+8.3f}  {verd}")


# ── MULTIPLE PLACEBOS ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"MULTIPLE PLACEBOS ({N_PLACEBOS} directions, Path A)")
print(f"{'='*70}")

s_circ = compute_stats(results_by_month['baseline'],
                       results_by_month['ablate_C'], PRIMARY_START)
placebo_dids = []
for pi in range(N_PLACEBOS):
    s_p = compute_stats(results_by_month['baseline'],
                        results_by_month[f'placebo_{pi}'], PRIMARY_START)
    placebo_dids.append(s_p['did'])
placebo_dids = np.array(placebo_dids)
c_did = s_circ['did']
pct_rank = (placebo_dids < c_did).mean() * 100

print(f"\n  Circuit C DiD: {c_did:+.3f}")
print(f"  Placebo distribution: mean={placebo_dids.mean():+.3f} "
      f"std={placebo_dids.std():.3f} "
      f"p95={np.percentile(placebo_dids,95):+.3f}")
print(f"  Circuit C at {pct_rank:.0f}th percentile of placebo DiDs")

if pct_rank >= 95:
    print(f"  ✓ STRONG: Circuit C is an extreme outlier in the placebo distribution.")
elif pct_rank >= 80:
    print(f"  ~ MODERATE: Circuit C above most placebos.")
else:
    print(f"  ✗ WEAK: Circuit C not distinguishable from random directions.")


# ── SUMMARY TABLES ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY — PATH A (Fixed 2023 model, both periods)")
print(f"{'='*70}")
for period, pname in [(PRIMARY_START,'Primary (post-2005)'),
                       (FULL_START,  'Full OOS (post-1987)')]:
    print(f"\n  {pname}:")
    print(f"  {'Ablation':<42} {'ΔSharpe NBER':>14} {'ΔSharpe Exp':>13} {'DiD':>8}")
    print(f"  {'-'*80}")
    for _, key in main_ablations:
        s = next((x for x in all_stats if key in x['label']
                  and ('Primary' in x['label'] if period==PRIMARY_START
                       else 'Full' in x['label'])), None)
        if s:
            lbl = {
                'ablate_C': 'H1: Circuit C (residual — CAUSAL)',
                'ablate_R': 'H2: Circuit R (zero ablation)',
                'ablate_R_permute': 'H2p: Circuit R (permutation)',
            }.get(key, key)[:41]
            print(f"  {lbl:<42} {s['delta_nber']:>+14.3f} "
                  f"{s['delta_exp']:>+13.3f} {s['did']:>+8.3f}")


# ── PHASE 3 CHECKPOINT ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PHASE 3 CHECKPOINT — PATH A")
print(f"{'='*70}")
c = next((s for s in all_stats
          if 'ablate_C' in s['label'] and 'Primary' in s['label']), None)
if c:
    did_lo = c.get('boot_did_lo', np.nan)
    print(f"\n  H1 Circuit C (Path A — 2023 model MI analysis):")
    print(f"    DiD primary:   {c['did']:+.3f}  CI_lo: {did_lo:+.3f}")
    print(f"    Full OOS DiD:  "
          f"{next(s['did'] for s in all_stats if 'ablate_C' in s['label'] and 'Full' in s['label']):.3f}")
    if not np.isnan(did_lo) and did_lo > 0:
        print(f"  ✓ H1 PASSES: causally necessary in 2023 model's recession computation.")
    elif c['did'] > 0:
        print(f"  ~ H1 DIRECTIONAL PASS: positive DiD, CI wide (22 NBER months).")
    else:
        print(f"  ✗ H1 FAIL: no causal necessity even in fixed 2023 model.")

print(f"\n  Multiple placebo check: Circuit C at {pct_rank:.0f}th pct of 20 placebos.")

r2p = next((s for s in all_stats
            if 'ablate_R_permute' in s['label'] and 'Primary' in s['label']), None)
if r2p:
    print(f"\n  H2p (PC1 permutation — Path A):")
    print(f"    ΔSharpe_NBER: {r2p['delta_nber']:+.3f}  DiD: {r2p['did']:+.3f}")
    print(f"    Note: compare to Path B H2p DiD=-0.953 (rolling checkpoints)")
    print(f"    If both negative: PC1 counterproductivity holds in 2023 model")
    print(f"    AND generalizes across all training vintages.")

print(f"\n  Path A vs Path B summary:")
print(f"    Path A (this script): MI of 2023 model — what does it compute?")
print(f"    Path B (15_3b_rolling): Does PC1 counterproductivity generalize")
print(f"      across all 37 training vintages? (H2p DiD=-0.953, confirmed)")


# ── SAVE ──────────────────────────────────────────────────────────────────────
pd.DataFrame(all_stats).to_csv(DATA_DIR / '3b_ablation_results_final.csv', index=False)
pd.DataFrame({'placebo_did': placebo_dids}).to_csv(
    DATA_DIR / '3b_placebo_distribution_final.csv', index=False)
print(f"\n  Saved 3b_ablation_results_final.csv")
print(f"  Saved 3b_placebo_distribution_final.csv")
print(f"\nNext steps after reviewing Path A results:")
print(f"  1. If H1 holds: update pre-reg memo with Path A/B distinction")
print(f"  2. Write 3C (structural break — lead-lag ONLY, no contemporaneous)")
print(f"  3. Write 3D (factor pricing + linear characteristic benchmark)")
print(f"  4. Update flexmap to v6")
