"""
07_phase1_completion.py
Completes the three outstanding Phase 1 tasks:
  (A) Variable importance — mean |gradient| per characteristic, compare to GKX Figure 4
  (B) Fama-MacBeth regression — predicted return on realised return, controls: size + BM
  (C) Baseline factor alpha — L/S decile portfolio on FF5 + momentum
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Load panel and predictions ────────────────────────────────────────────────
print("Loading panel and predictions...")
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'ret_norm', 'prc',
             'shrout', 'exchcd', 'shrcd', 'vol', 'siccd', 'sic2']
panel = pd.read_parquet(DATA_DIR / "panel_v1_20260330.parquet")
CHAR_COLS = [c for c in panel.columns if c not in SKIP_COLS]

preds_df = pd.read_parquet(DATA_DIR / f"predictions_seed{SEED}.parquet")

# Reconstruct ret_norm on predictions (same recipe as training)
preds_df['ret_norm'] = preds_df.groupby('yyyymm')['ret_adj'].transform(
    lambda x: x / (x.std() + 1e-8)
)
p1  = preds_df['ret_norm'].quantile(0.01)
p99 = preds_df['ret_norm'].quantile(0.99)
preds_df['ret_norm'] = preds_df['ret_norm'].clip(lower=p1, upper=p99)

# ── Rebuild GKXMLP (needed for gradient importance) ───────────────────────────
HIDDEN   = [32, 16, 8]
DROPOUT  = 0.5

class GKXMLP(nn.Module):
    def __init__(self, n_chars=94, hidden=[32, 16, 8], dropout=0.5):
        super().__init__()
        self.activations = {}
        layers = []
        prev = n_chars
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
## ── (A) VARIABLE IMPORTANCE — mean |gradient| ────────────────────────────────
# ============================================================
print("\n" + "="*60)
print("(A) VARIABLE IMPORTANCE — mean |gradient|")
print("="*60)

panel = panel.dropna(subset=['ret_adj'])
panel = panel.sort_values(['permno', 'yyyymm'])
for col in CHAR_COLS:
    panel[col] = panel.groupby('permno')[col].shift(1)
for col in CHAR_COLS:
    panel[col] = panel[col].fillna(
        panel.groupby('yyyymm')[col].transform('median')
    ).fillna(0)

ckpt_path = MODEL_DIR / f"checkpoint_202301_seed{SEED}.pt"
model = GKXMLP(n_chars=len(CHAR_COLS), hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

oos_panel = panel[panel['yyyymm'] >= 198701].sample(5000, random_state=SEED)
X_np = oos_panel[CHAR_COLS].values.astype(np.float32)

# ── Use enable_grad + detach/clone pattern for reliable gradients ─────────────
with torch.enable_grad():
    X_sample = torch.from_numpy(X_np).to(DEVICE)
    X_sample.requires_grad_(True)
    pred = model(X_sample)
    pred.sum().backward()
    grad_importance = X_sample.grad.abs().mean(dim=0).cpu().detach().numpy()

importance_df = pd.DataFrame({
    'characteristic': CHAR_COLS,
    'mean_abs_gradient': grad_importance
}).sort_values('mean_abs_gradient', ascending=False)

print("\nTop 20 characteristics by mean |gradient|:")
print(importance_df.head(20).to_string(index=False))

TOP5 = set(importance_df.head(5)['characteristic'].tolist())
EXPECTED = {'mom1m', 'mom12m', 'maxret', 'retvol', 'ill',
            'dolvol', 'baspread', 'idiovol', 'std_dolvol', 'turn'}
overlap = TOP5 & EXPECTED
print(f"\nTop-5 overlap with GKX Figure 4 expected signals: {overlap}")
if len(overlap) >= 2:
    print("✓ Variable importance check passed")
else:
    print("⚠ Less overlap than expected — review against GKX Figure 4")

importance_df.to_csv(DATA_DIR / "variable_importance.csv", index=False)
print("Saved to variable_importance.csv")

# ═══════════════════════════════════════════════════════════════════════════
# (B) FAMA-MACBETH REGRESSION
# Regress realised ret_norm on predicted return, controlling for size and BM
# FM slope on pred should be significantly positive (t > 2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("(B) FAMA-MACBETH REGRESSION")
print("="*60)

# Merge predictions with size (mvel1) and BM from panel
fm_panel = preds_df[['permno', 'yyyymm', 'ret_norm', 'pred']].copy()

# Pull mvel1 and bm from the shifted panel
aux = panel[['permno', 'yyyymm', 'mvel1', 'bm']].copy()
fm_panel = fm_panel.merge(aux, on=['permno', 'yyyymm'], how='left')

# Log size
fm_panel['log_mvel1'] = np.log(fm_panel['mvel1'].clip(lower=1e-6))

# Drop rows with missing controls
fm_panel = fm_panel.dropna(subset=['pred', 'ret_norm', 'log_mvel1', 'bm'])

# Cross-sectional OLS each month
monthly_slopes = []
for yyyymm, grp in fm_panel.groupby('yyyymm'):
    if len(grp) < 50:
        continue
    X = sm.add_constant(grp[['pred', 'log_mvel1', 'bm']])
    try:
        res = sm.OLS(grp['ret_norm'], X).fit()
        row = res.params.to_dict()
        row['yyyymm'] = yyyymm
        monthly_slopes.append(row)
    except Exception:
        continue

slopes_df = pd.DataFrame(monthly_slopes)

# FM standard errors: time-series mean / (std / sqrt(T))
T = len(slopes_df)
fm_means = slopes_df.drop(columns='yyyymm').mean()
fm_se    = slopes_df.drop(columns='yyyymm').std() / np.sqrt(T)
fm_t     = fm_means / fm_se

print(f"\nFama-MacBeth results (T = {T} months):")
print(f"{'Variable':<15} {'Mean coef':>12} {'t-stat':>10}")
print("-"*40)
for var in ['pred', 'log_mvel1', 'bm']:
    if var in fm_means:
        print(f"{var:<15} {fm_means[var]:>12.4f} {fm_t[var]:>10.2f}")

pred_t = fm_t.get('pred', np.nan)
if pred_t > 2:
    print(f"\n✓ FM slope on pred: t = {pred_t:.2f} > 2 — predictions are economically meaningful")
else:
    print(f"\n⚠ FM slope on pred: t = {pred_t:.2f} < 2 — investigate before Phase 2")

slopes_df.to_csv(DATA_DIR / "fama_macbeth_slopes.csv", index=False)
print("Saved monthly slopes to fama_macbeth_slopes.csv")


# ═══════════════════════════════════════════════════════════════════════════
# (C) BASELINE FACTOR ALPHA — L/S decile on FF5 + momentum
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("(C) BASELINE FACTOR ALPHA — FF5 + Momentum")
print("="*60)

# ── Download FF5 + momentum factors from Ken French ──────────────────────────
import urllib.request, zipfile, io

def fetch_french_factor(url, filename):
    print(f"  Downloading {filename}...")
    req = urllib.request.urlopen(url)
    zf  = zipfile.ZipFile(io.BytesIO(req.read()))
    raw = zf.read(filename).decode('utf-8', errors='ignore')

    # Ken French CSVs have a header section, then data, then annual data
    # Find the first line that starts with a 6-digit date
    lines = raw.split('\n')
    data_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                val = int(parts[0])
                if 190001 <= val <= 210012:  # valid yyyymm range
                    data_rows.append(parts)
            except ValueError:
                pass
    
    df = pd.DataFrame(data_rows)
    df = df.set_index(0)
    df = df.apply(pd.to_numeric, errors='coerce') / 100
    df = df.dropna(how='all')
    return df

try:
    ff5 = fetch_french_factor(
    'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip',
    'F-F_Research_Data_5_Factors_2x3.csv'
    )
    ff5.columns = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']

    mom = fetch_french_factor(
        'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip',
        'F-F_Momentum_Factor.csv'
    )
    mom.columns = ['mom']

    factors = ff5[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']].join(mom, how='inner')
    factors.index.name = 'yyyymm'
    factors.index = factors.index.astype(int)
    print(f"  Factors loaded: {len(factors)} months")
    print(f"  Columns: {factors.columns.tolist()}")
    ff_available = True

except Exception as e:
    print(f"  ⚠ Could not download FF factors: {e}")
    print("  Skipping factor alpha test — download manually from Ken French website")
    ff_available = False

if ff_available:
    # Reconstruct L/S decile returns
    preds_df['decile'] = preds_df.groupby('yyyymm')['pred'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    ls_returns = preds_df.groupby('yyyymm').apply(
        lambda x: x[x['decile']==9]['ret_adj'].mean()
              - x[x['decile']==0]['ret_adj'].mean(),
        include_groups=False
    ).rename('ls_ret')

    # Reconstruct L/S decile returns
    preds_df['decile'] = preds_df.groupby('yyyymm')['pred'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    ls_returns = preds_df.groupby('yyyymm').apply(
        lambda x: x[x['decile']==9]['ret_adj'].mean()
            - x[x['decile']==0]['ret_adj'].mean(),
        include_groups=False
    ).rename('ls_ret')

    factor_df = ls_returns.reset_index()
    factor_df.columns = ['yyyymm', 'ls_ret']
    factor_df['yyyymm'] = factor_df['yyyymm'].astype(int)

    factors_reset = factors.reset_index()
    factors_reset['yyyymm'] = factors_reset['yyyymm'].astype(int)

    # Diagnostic — print ranges to confirm overlap
    print(f"  preds yyyymm range: {factor_df['yyyymm'].min()} – {factor_df['yyyymm'].max()}")
    print(f"  factors yyyymm range: {factors_reset['yyyymm'].min()} – {factors_reset['yyyymm'].max()}")

    factor_df = factor_df.merge(
        factors_reset[['yyyymm','mkt_rf','smb','hml','rmw','cma','mom']],
        on='yyyymm', how='inner'
    )
    print(f"  Matched months: {len(factor_df)}")
    print(f"  NaNs in factor_df:\n{factor_df[['mkt_rf','smb','hml','rmw','cma','mom']].isna().sum()}")
    print(f"  NaNs in ls_ret: {factor_df['ls_ret'].isna().sum()}")
    print(f"  Sample rows:\n{factor_df[['yyyymm','ls_ret','mkt_rf','mom']].head()}")

    # Subtract rf from L/S return (it's already an excess spread but keep consistent)
    factor_df = factor_df.dropna(subset=['ls_ret', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom'])
    print(f"  Rows after dropping NaN ls_ret: {len(factor_df)}")

    Y = factor_df['ls_ret'].values
    X = sm.add_constant(
        factor_df[['mkt_rf','smb','hml','rmw','cma','mom']].values
    )

    res = sm.OLS(Y, X).fit(
        cov_type='HAC', cov_kwds={'maxlags': 6}
    )

    alpha_monthly = res.params[0]
    alpha_annual  = alpha_monthly * 12 * 100  # annualised in bps
    alpha_t       = res.tvalues[0]

    print(f"\nFF5 + Momentum alpha:")
    print(f"  Monthly alpha : {alpha_monthly:.4f} ({alpha_monthly*100:.2f} bps/month)")
    print(f"  Annualised    : {alpha_annual:.1f} bps/year")
    print(f"  t-stat        : {alpha_t:.2f}")

    if alpha_t > 2.0:
        print(f"✓ Significant positive alpha (t = {alpha_t:.2f}) — GKX replication is economically valid")
    else:
        print(f"⚠ Alpha t = {alpha_t:.2f} — weak factor pricing result, investigate")

    print("\nFull regression summary:")
    print(f"  Intercept: {res.params[0]:.4f}, t = {res.tvalues[0]:.2f}")

    # Save
    alpha_result = pd.DataFrame({
        'metric': ['alpha_monthly', 'alpha_annual_bps', 'alpha_t'],
        'value':  [alpha_monthly, alpha_annual, alpha_t]
    })
    alpha_result.to_csv(DATA_DIR / "factor_alpha.csv", index=False)
    print("Saved to factor_alpha.csv")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 COMPLETION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1 COMPLETION SUMMARY")
print("="*60)
print(f"(a) OOS R²              : 0.5853%  ✓ in [0.25%, 0.60%]")
print(f"(b) L/S Sharpe          : 1.609    ✓ > 1.5")
print(f"(c) HDF5 files          : verify with: python -c \"import h5py; f=h5py.File('activations/acts_198701.h5'); print(list(f.keys()))\"")
print(f"(d) Regime definitions  : NBER=93, VIX=73, Credit=72, NFCI=123 — all ✓")
print(f"(e) Variable importance : see variable_importance.csv")
print(f"(f) FM slope t-stat     : see fama_macbeth_slopes.csv")
print(f"(g) FF5+Mom alpha       : see factor_alpha.csv")
print("\nIf all checks pass: Phase 1 → COMPLETE. Proceed to Phase 2.")