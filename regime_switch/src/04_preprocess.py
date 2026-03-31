import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ── Load all three datasets ───────────────────────────────────────────────────
print("Loading data...")
crsp    = pd.read_parquet(DATA_DIR / "crsp_monthly_raw.parquet")
signals = pd.read_parquet(DATA_DIR / "signals_raw.parquet")
comp    = pd.read_parquet(DATA_DIR / "compustat_chars.parquet")

print(f"CRSP:    {crsp.shape}")
print(f"Signals: {signals.shape}")
print(f"Comp:    {comp.shape}")

# ── Align date formats ────────────────────────────────────────────────────────
# CRSP uses datetime, convert to yyyymm int to match signals and comp
crsp['yyyymm'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month

# ── Merge CRSP with signals ───────────────────────────────────────────────────
print("Merging CRSP + signals...")
panel = crsp[['permno', 'yyyymm', 'ret_adj', 'prc', 'shrout', 'exchcd', 'shrcd', 'vol', 'siccd']].merge(
    signals, on=['permno', 'yyyymm'], how='left'
)


# ── Merge with Compustat characteristics ─────────────────────────────────────
print("Merging with Compustat characteristics...")
panel = panel.merge(comp, on=['permno', 'yyyymm'], how='left')

print(f"Panel shape after merge: {panel.shape}")

# ── Filter to 1963-2023 ───────────────────────────────────────────────────────
panel = panel[(panel['yyyymm'] >= 196301) & (panel['yyyymm'] <= 202312)]
print(f"Panel shape after date filter: {panel.shape}")

# ── Compute market cap (mvel1) from CRSP ─────────────────────────────────────
panel['mvel1'] = panel['prc'].abs() * panel['shrout']

# ── Construct CRSP-based signals ──────────────────────────────────────────────
# turn: share turnover = monthly volume / shares outstanding
panel['turn'] = panel['vol'] / panel['shrout']

# std_dolvol: rolling 36-month std of log dollar volume
# dollar volume = vol * abs(prc), then log, then rolling std
panel['dolvol_monthly'] = np.log(panel['vol'] * panel['prc'].abs() + 1)
panel = panel.sort_values(['permno', 'yyyymm'])
panel['std_dolvol'] = panel.groupby('permno')['dolvol_monthly'].transform(
    lambda x: x.rolling(36, min_periods=24).std()
)
panel = panel.drop(columns=['dolvol_monthly'])

# mve_ia: industry-adjusted size = mvel1 minus industry mean
# sic2 = 2-digit SIC from CRSP siccd
panel['sic2'] = (panel['siccd'] / 100).astype(int)
panel['mve_ia'] = panel.groupby(['yyyymm', 'sic2'])['mvel1'].transform(
    lambda x: x - x.mean()
)

# betasq: square of market beta — BetaSquared.csv not available in OAP release
# Construct directly from beta signal which is downloaded
if 'beta' in panel.columns:
    panel['betasq'] = panel['beta'] ** 2
else:
    panel['betasq'] = 0.0
    print("  Warning: beta not found — betasq set to 0")

# ── Industry-adjusted signals (raw signal minus industry mean) ────────────────

# chpmia: OAP doesn't have ChPM downloaded; approximate as pchgm_pchsale proxy
# or set to 0 pending proper construction
# For now construct bm_ia, cfp_ia, pchcapx_ia from their raw OAP counterparts:
for ia_col, raw_col in [('bm_ia', 'bm'), ('cfp_ia', 'cfp'), ('pchcapx_ia', 'grCAPX'), ('chempia', 'hire')]:
    if raw_col in panel.columns:
        panel[ia_col] = (
            panel[raw_col]
            - panel.groupby(['yyyymm', 'sic2'])[raw_col].transform('mean')
        )
    else:
        panel[ia_col] = 0.0
        print(f"  Warning: {raw_col} not found — {ia_col} set to 0")

# aeavol: requires IBES announcement dates — set to 0 (missing) for now
# Document as deviation in preprocessing_notes.md
panel['aeavol'] = 0.0   # requires IBES — documented deviation
panel['chpmia']  = 0.0  # requires quarterly data — documented deviation

# ── Define characteristic columns ────────────────────────────────────────────
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'prc', 'shrout', 'exchcd', 'shrcd', 'date', 'vol','siccd', 'sic2']
CHAR_COLS = [c for c in panel.columns if c not in SKIP_COLS]

print(f"Characteristic columns: {len(CHAR_COLS)}")

# ── GKX cross-sectional preprocessing ────────────────────────────────────────
# Applied each month:
# (a) winsorise at 1st/99th percentile
# (b) rank-transform to [-1, +1]
# (c) set missing values to 0

def preprocess_month(df):
    result = df.copy()
    for col in CHAR_COLS:
        if col not in result.columns:
            continue
        x = result[col].astype(float)
        # (a) winsorise
        p1  = x.quantile(0.01)
        p99 = x.quantile(0.99)
        x = x.clip(lower=p1, upper=p99)
        # (b) rank-transform to [-1, +1]
        ranked = x.rank(method='average', na_option='keep')
        n = ranked.notna().sum()
        if n > 0:
            x = 2 * (ranked - 1) / (n - 1) - 1 if n > 1 else ranked * 0
        # (c) missing to 0
        x = x.fillna(0)
        result[col] = x
    return result

print("Applying GKX preprocessing (this may take a few minutes)...")
yyyymm_vals = panel['yyyymm'].values
panel = panel.groupby('yyyymm', group_keys=False).apply(
    preprocess_month, include_groups=False
).reset_index(drop=True)
panel['yyyymm'] = yyyymm_vals
print("Preprocessing done.")

# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"\nFinal panel shape: {panel.shape}")
print(f"Date range: {panel['yyyymm'].min()} to {panel['yyyymm'].max()}")
print(f"Unique permnos: {panel['permno'].nunique():,}")
print(f"Avg stock-months per month: {len(panel) / panel['yyyymm'].nunique():,.0f}")
print(f"Mean ret_adj: {panel['ret_adj'].mean():.4f}")
print(f"Char value range (should be [-1,1]): {panel[CHAR_COLS].min().min():.2f} to {panel[CHAR_COLS].max().max():.2f}")
print(panel.head())

# ── Save ──────────────────────────────────────────────────────────────────────
today = pd.Timestamp.today().strftime('%Y%m%d')
out = DATA_DIR / f"panel_v1_{today}.parquet"
panel.to_parquet(out, index=False)
print(f"\nSaved to {out}")