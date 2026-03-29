import wrds
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Connect to WRDS ──────────────────────────────────────────────────────────
db = wrds.Connection(wrds_username='shifty_nifty')


# ── Pull CRSP monthly stock file ─────────────────────────────────────────────
print("Pulling CRSP msf...")

crsp = db.raw_sql("""
    SELECT a.permno, a.date, a.ret, a.retx,
           b.shrcd, b.exchcd,
           a.prc, a.shrout
    FROM crsp.msf AS a
    JOIN crsp.msenames AS b
        ON a.permno = b.permno
        AND a.date BETWEEN b.namedt AND b.nameendt
    WHERE b.shrcd IN (10, 11)
      AND b.exchcd IN (1, 2, 3)
      AND a.date BETWEEN '1963-01-01' AND '2023-12-31'
""", date_cols=['date'])

# ── Pull delisting returns ────────────────────────────────────────────────────
print("Pulling delisting returns...")
delist = db.raw_sql("""
    SELECT permno, dlstdt AS date, dlret, dlstcd
    FROM crsp.msedelist
""", date_cols=['date'])

db.close()

# ── Align delist dates to month-end and merge ─────────────────────────────────
delist['date'] = delist['date'] + pd.offsets.MonthEnd(0)

crsp = crsp.merge(
    delist[['permno', 'date', 'dlret', 'dlstcd']],
    on=['permno', 'date'],
    how='left',
    suffixes=('', '_delist')
)

# ── Shumway (1997) imputation for missing dlret ───────────────────────────────
def impute_dlret(row):
    if pd.notna(row['dlret']):
        return row['dlret']
    if pd.notna(row['dlstcd']):
        if row['exchcd'] in (1, 2):   # NYSE / AMEX
            return -0.30
        elif row['exchcd'] == 3:       # NASDAQ
            return -0.55
    return None

crsp['dlret_adj'] = crsp.apply(impute_dlret, axis=1)

# ── Compute delisting-adjusted return ─────────────────────────────────────────
def adjusted_return(row):
    r = row['ret']
    d = row['dlret_adj']
    if pd.notna(d):
        if pd.notna(r):
            return (1 + r) * (1 + d) - 1
        else:
            return d
    return r

crsp['ret_adj'] = crsp.apply(adjusted_return, axis=1)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"Rows: {len(crsp):,}")
print(f"Unique permnos: {crsp['permno'].nunique():,}")
print(f"Date range: {crsp['date'].min()} to {crsp['date'].max()}")
print(f"Mean monthly ret_adj: {crsp['ret_adj'].mean():.4f}")
print(f"Stock-months per month (avg): {len(crsp) / crsp['date'].nunique():,.0f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = DATA_DIR / "crsp_monthly_raw.parquet"
crsp.to_parquet(out, index=False)
print(f"Saved to {out}")