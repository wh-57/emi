"""
14_pre_memo_diagnostics.py — Pre-Memo Diagnostics
===================================================
Run BEFORE writing the Phase 3A pre-registration memo.

Two diagnostics flagged as essential:

(A) Angle between PC1 and shift_resid
    Quantifies how geometrically different the representation direction
    (PC1) and computation direction (shift_resid) actually are.
    If angle ~ 90°: clean orthogonal dissociation.
    If angle < 30°: directions are nearly collinear, dissociation is weak.
    Also: cosine similarity between shift_resid and each PC1..PC6.

(B) Regime-conditional FM regression
    Run Fama-MacBeth separately for expansion vs. recession months.
    Predictors: PC1 projection score and residual projection score.
    Key question: does the residual direction predict returns in
    recessions but not expansions? If yes: functional decoupling is
    economically meaningful and the ablation prediction is sharp.

Results go into 2d_findings.md update and pre-registration memo.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

BASE     = Path(r'C:\Users\willi\Desktop\emi\regime_switch')
ACTS_DIR = BASE / 'activations'
DATA_DIR = BASE / 'data'

SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'vol', 'siccd', 'sic2',
             'prc', 'shrout', 'exchcd', 'shrcd']

# ── Load shift decomposition and PCA ─────────────────────────────────────────
print("Loading shift decomposition and PCA components...")
df_shift    = pd.read_csv(DATA_DIR / '2c_shift_decomposition.csv')
shift_resid = df_shift['shift_residual'].values   # (16,) causal direction
shift_total = df_shift['shift_total'].values
pc1_loading = df_shift['pc1_loading'].values       # (16,) PC1 direction

# Load activations to refit full PCA (need PC2-PC6 too)
print("Loading layer1 activations...")
regimes    = pd.read_csv(DATA_DIR / 'monthly_regimes.csv')
regimes['yyyymm'] = regimes['yyyymm'].astype(int)
regime_map = dict(zip(regimes['yyyymm'], regimes['nber']))

acts_list, regime_list, yyyymm_list, permno_list = [], [], [], []
for f in sorted(ACTS_DIR.glob('acts_*.h5')):
    yyyymm = int(f.stem.replace('acts_', ''))
    if yyyymm not in regime_map or pd.isna(regime_map[yyyymm]):
        continue
    with h5py.File(f, 'r') as hf:
        acts = hf['layer1'][:]
        acts_list.append(acts)
        regime_list.append(np.full(len(acts), regime_map[yyyymm], dtype=np.float32))
        yyyymm_list.append(np.full(len(acts), yyyymm, dtype=np.int32))
        permno_list.append(hf['permno'][:])

acts_np   = np.vstack(acts_list).astype(np.float32)
regime_np = np.concatenate(regime_list)
yyyymm_np = np.concatenate(yyyymm_list)
permno_np = np.concatenate(permno_list)

acts_mu   = acts_np.mean(0, keepdims=True)
acts_sig  = acts_np.std(0, keepdims=True) + 1e-8
acts_norm = (acts_np - acts_mu) / acts_sig

exp_mask = regime_np == 0

# Refit PCA on expansion months
pca = PCA(n_components=16)
pca.fit(acts_norm[exp_mask][:50_000])
pc_directions = pca.components_   # (16, 16) — all principal directions


# ══════════════════════════════════════════════════════════════════════════════
# A. ANGLE BETWEEN PC1 AND SHIFT_RESID
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("DIAGNOSTIC A — Geometric angle between PC1 and causal residual")
print(f"{'='*65}")

# Normalize both directions
unit_pc1   = pc1_loading / np.linalg.norm(pc1_loading)
unit_resid = shift_resid / (np.linalg.norm(shift_resid) + 1e-12)

cos_angle   = float(np.dot(unit_pc1, unit_resid))
angle_deg   = float(np.degrees(np.arccos(np.clip(abs(cos_angle), 0, 1))))
angle_signed = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

print(f"\n  Cosine similarity (PC1, shift_resid):  {cos_angle:.4f}")
print(f"  Angle (unsigned):                      {angle_deg:.1f}°")
print(f"  Angle (signed):                        {angle_signed:.1f}°")

if angle_deg > 75:
    print(f"  → STRONG GEOMETRIC DISSOCIATION: directions are nearly orthogonal.")
    print(f"    The same economic inputs are encoded in two very different directions.")
elif angle_deg > 45:
    print(f"  → MODERATE GEOMETRIC DISSOCIATION: significant angular separation.")
elif angle_deg > 20:
    print(f"  → WEAK DISSOCIATION: directions are somewhat collinear.")
else:
    print(f"  → MINIMAL DISSOCIATION: directions nearly collinear. Revisit claim.")

# Decompose shift_resid onto each PC
print(f"\n  Decomposition of shift_resid onto PCs:")
print(f"  {'PC':>4}  {'Projection':>12}  {'Frac of norm':>14}  {'Var explained':>15}")
print(f"  {'-'*50}")
resid_norm_sq = np.dot(shift_resid, shift_resid)
for i in range(8):
    proj = float(np.dot(shift_resid, pc_directions[i]))
    frac = proj**2 / (resid_norm_sq + 1e-12)
    print(f"  {i+1:>4}  {proj:>12.4f}  {frac:>13.1%}  "
          f"{pca.explained_variance_ratio_[i]:>14.3f}")

# By construction, PC1 projection of shift_resid should be ~0
print(f"\n  Note: shift_resid was constructed by removing PC1 from shift_total.")
print(f"  PC1 projection of shift_resid = "
      f"{float(np.dot(shift_resid, pc_directions[0])):.6f} (should be ~0)")


# ══════════════════════════════════════════════════════════════════════════════
# B. REGIME-CONDITIONAL FM REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("DIAGNOSTIC B — Regime-conditional return predictability")
print("Question: does residual direction predict returns in recessions?")
print(f"{'='*65}")

# Compute PC1 and residual projection scores per stock-month
pc1_scores   = acts_norm @ pc_directions[0]   # (N,) — projection onto PC1
resid_scores = acts_norm @ unit_resid          # (N,) — projection onto causal residual

# Load panel returns
panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260330.parquet',
                        columns=['permno', 'yyyymm', 'ret_adj'])
panel['yyyymm'] = panel['yyyymm'].astype(int)
panel['permno'] = panel['permno'].astype(int)
panel = panel.dropna(subset=['ret_adj'])

# Normalized return
panel['ret_norm'] = panel.groupby('yyyymm')['ret_adj'].transform(
    lambda x: x / (x.std() + 1e-8))

# Build stock-month dataframe with scores
df = pd.DataFrame({
    'permno':       permno_np.astype(int),
    'yyyymm':       yyyymm_np.astype(int),
    'nber':         regime_np.astype(int),
    'pc1_score':    pc1_scores,
    'resid_score':  resid_scores,
})
df = df.merge(panel[['permno', 'yyyymm', 'ret_norm']], on=['permno', 'yyyymm'], how='inner')
df = df.dropna(subset=['ret_norm'])

print(f"\n  Matched observations: {len(df):,}")
print(f"  Expansion months:     {(df['nber']==0)['yyyymm'].nunique() if False else df[df['nber']==0]['yyyymm'].nunique()}")
print(f"  Recession months:     {df[df['nber']==1]['yyyymm'].nunique()}")

def fm_tstat(df_sub, predictor):
    """Fama-MacBeth t-stat for predictor → ret_norm."""
    slopes = []
    for yyyymm, grp in df_sub.groupby('yyyymm'):
        if len(grp) < 50:
            continue
        x = grp[predictor].values
        y = grp['ret_norm'].values
        if x.std() < 1e-8:
            continue
        s, *_ = stats.linregress(x, y)
        slopes.append(s)
    T = len(slopes)
    if T < 5:
        return np.nan, np.nan, T
    mean_s = np.mean(slopes)
    t      = mean_s / (np.std(slopes) / np.sqrt(T) + 1e-12)
    return mean_s, t, T

# Split by regime
df_exp = df[df['nber'] == 0]
df_rec = df[df['nber'] == 1]

print(f"\n  Fama-MacBeth results:")
print(f"  {'Predictor':<18} {'Regime':<12} {'Mean slope':>12} {'FM t-stat':>12} {'T months':>10}")
print(f"  {'-'*66}")

results = {}
for pred in ['pc1_score', 'resid_score']:
    for label, df_sub in [('Expansion', df_exp), ('Recession', df_rec), ('Pooled', df)]:
        mean_s, t, T = fm_tstat(df_sub, pred)
        print(f"  {pred:<18} {label:<12} {mean_s:>12.4f} {t:>12.2f} {T:>10}")
        results[f'{pred}_{label}'] = {'slope': mean_s, 't': t, 'T': T}

# Key comparison
print(f"\n  Key comparison — residual score:")
t_exp = results['resid_score_Expansion']['t']
t_rec = results['resid_score_Recession']['t']
print(f"    FM t in expansion: {t_exp:.2f}")
print(f"    FM t in recession: {t_rec:.2f}")

if abs(t_rec) > 2 and abs(t_exp) < 1:
    print(f"  ✓ STRONG FUNCTIONAL DECOUPLING: residual predicts returns in")
    print(f"    recessions (t={t_rec:.2f}) but not expansions (t={t_exp:.2f}).")
    print(f"    This supports regime-specific use of the causal direction.")
elif abs(t_rec) > abs(t_exp) * 1.5:
    print(f"  ~ MODERATE DECOUPLING: residual has stronger predictive power")
    print(f"    in recessions than expansions.")
elif abs(t_rec) < 1 and abs(t_exp) < 1:
    print(f"  ~ NEITHER PREDICTS LINEARLY: consistent with nonlinear computation")
    print(f"    through later layers. Ablation test remains the primary evidence.")
else:
    print(f"  ~ MIXED: symmetric or expansion-dominated. Review before memo.")

print(f"\n  Key comparison — PC1 score:")
t_exp_pc1 = results['pc1_score_Expansion']['t']
t_rec_pc1 = results['pc1_score_Recession']['t']
print(f"    FM t in expansion: {t_exp_pc1:.2f}")
print(f"    FM t in recession: {t_rec_pc1:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# C. MAGNITUDE RATIO CHECK (Gemini suggestion)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("DIAGNOSTIC C — Magnitude ratios: PC1 vs residual characteristic loadings")
print(f"{'='*65}")

df_econ = pd.read_csv(DATA_DIR / 'residual_economic_content.csv')
df_econ['abs_proj_pc1']   = np.abs(df_econ['proj_pc1'])
df_econ['abs_proj_resid'] = np.abs(df_econ['proj_residual_causal'])
df_econ['ratio_resid_pc1'] = (df_econ['abs_proj_resid'] /
                               (df_econ['abs_proj_pc1'] + 1e-8))

print(f"\n  Top-10 characteristics: PC1 loading vs residual loading")
print(f"  {'Char':<22} {'|PC1|':>8} {'|Resid|':>8} {'Resid/PC1':>10}")
print(f"  {'-'*52}")

top_chars = df_econ.nlargest(10, 'abs_proj_pc1')
for _, row in top_chars.iterrows():
    print(f"  {row['characteristic']:<22} "
          f"{row['abs_proj_pc1']:>8.4f} "
          f"{row['abs_proj_resid']:>8.4f} "
          f"{row['ratio_resid_pc1']:>10.2f}x")

# Is there a characteristic where residual >> PC1?
high_resid_chars = df_econ[df_econ['ratio_resid_pc1'] > 1.5].nlargest(5, 'abs_proj_resid')
if len(high_resid_chars) > 0:
    print(f"\n  Characteristics where residual loading > 1.5x PC1 loading:")
    for _, row in high_resid_chars.iterrows():
        print(f"    {row['characteristic']:<22} ratio={row['ratio_resid_pc1']:.2f}x")
else:
    print(f"\n  No characteristic has residual loading > 1.5x PC1 loading.")
    print(f"  Both directions load proportionally on the same characteristics.")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS FOR MEMO
# ══════════════════════════════════════════════════════════════════════════════

summary = {
    'angle_pc1_resid_deg':        round(angle_deg, 1),
    'cosine_pc1_resid':           round(cos_angle, 4),
    'resid_fm_t_expansion':       round(t_exp, 2),
    'resid_fm_t_recession':       round(t_rec, 2),
    'pc1_fm_t_expansion':         round(t_exp_pc1, 2),
    'pc1_fm_t_recession':         round(t_rec_pc1, 2),
}
pd.DataFrame([summary]).to_csv(DATA_DIR / 'pre_memo_diagnostics.csv', index=False)
print(f"\n  Saved pre_memo_diagnostics.csv")

print(f"\n{'='*65}")
print("SUMMARY FOR PRE-REGISTRATION MEMO")
print(f"{'='*65}")
print(f"""
  Geometric dissociation:
    Angle between PC1 and causal residual: {angle_deg:.1f}°
    Cosine similarity:                     {cos_angle:.4f}
    {'→ Directions are geometrically distinct' if angle_deg > 45 else '→ Directions are similar — document carefully'}

  Regime-conditional predictability:
    Residual FM t (expansion): {t_exp:.2f}
    Residual FM t (recession): {t_rec:.2f}
    PC1 FM t (expansion):      {t_exp_pc1:.2f}
    PC1 FM t (recession):      {t_rec_pc1:.2f}

  Economic overlap (from script 13):
    Both PC1 and residual top-5: baspread, idiovol, retvol, std_dolvol
    Interpretation: functional decoupling within same economic family

  Memo language:
    Circuit R (representation): PC1 direction — baseline volatility/illiq ranking
    Circuit C (computation):    Residual direction — regime-sensitive output driver
    These are geometrically {('distinct ({:.0f}°)'.format(angle_deg))} but economically similar.

  Primary ablation prediction:
    Ablating Circuit C (residual) degrades recession Sharpe more than expansion.
    Ablating Circuit R (PC1) has smaller or symmetric regime effect.
    Placebo (random ortho direction) shows no regime-differential effect.
""")
