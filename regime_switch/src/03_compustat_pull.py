import wrds
import pandas as pd
import numpy as np
from pathlib import Path



# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Connect to WRDS ──────────────────────────────────────────────────────────
db = wrds.Connection(wrds_username='shifty_nifty')

# ── Pull Compustat annual fundamentals ───────────────────────────────────────
print("Pulling Compustat annuals...")
comp = db.raw_sql("""
    SELECT gvkey, datadate, fyear,
           at, lt, seq, ceq, txditc, txdb, itcb,
           act, lct, che, dlc, dltt, dd1,
           sale, cogs, xsga, xrd, xad,
           ib, ni, oancf, dp, capx, invt,
           rect, ppegt, ppent, gdwl, intan,
           dv, csho, prcc_f, ajex,
           dd2, dd3, dd4, dd5,
           pstk, pstkrv, pstkl, txp, mib
    FROM comp.funda
    WHERE indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
      AND datadate BETWEEN '1960-01-01' AND '2023-12-31'
""", date_cols=['datadate'])

print(f"Compustat shape: {comp.shape}")

# ── Pull CCM link table ───────────────────────────────────────────────────────
print("Pulling CCM link table...")
ccm = db.raw_sql("""
    SELECT gvkey, lpermno AS permno, linktype, linkprim,
           linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
""", date_cols=['linkdt', 'linkenddt'])

db.close()

# ── Merge Compustat with CCM ──────────────────────────────────────────────────
print("Merging with CCM...")
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
comp = comp.merge(ccm, on='gvkey', how='left')

# Keep only observations where datadate falls within link window
comp = comp[
    (comp['datadate'] >= comp['linkdt']) &
    (comp['datadate'] <= comp['linkenddt'])
]

# ── Apply 6-month reporting lag ───────────────────────────────────────────────
# Accounting data available 6 months after fiscal year end
comp['public_date'] = comp['datadate'] + pd.DateOffset(months=6)
comp['yyyymm'] = comp['public_date'].dt.year * 100 + comp['public_date'].dt.month

# ── Construct the 22 missing GKX characteristics ─────────────────────────────
print("Constructing characteristics...")

# Helper: lag variable by 1 year within firm
def lag(df, col, n=1):
    return df.groupby('permno')[col].shift(n)

# Sort for lagging
comp = comp.sort_values(['permno', 'datadate'])

# Book equity
comp['be'] = comp['seq'].fillna(comp['ceq'] + comp['pstk'].fillna(0)) + \
             comp['txditc'].fillna(comp['txdb'].fillna(0) + comp['itcb'].fillna(0)) - \
             comp['pstkrv'].fillna(comp['pstkl'].fillna(comp['pstk'].fillna(0)))
comp['be'] = comp['be'].where(comp['be'] > 0)

# mvel1: size = market cap = price * shares (will merge with CRSP prc later)
# Using Compustat fiscal year end price as proxy
comp['mvel1'] = comp['prcc_f'] * comp['csho']

# cashdebt: (ib + dp) / lt
comp['cashdebt'] = (comp['ib'] + comp['dp'].fillna(0)) / comp['lt']

# currat: act / lct
comp['currat'] = comp['act'] / comp['lct']

# depr: dp / ppent
comp['depr'] = comp['dp'] / comp['ppent']

# invest: (capx + invt change) / at_lag
comp['at_lag'] = lag(comp, 'at')
comp['invt_lag'] = lag(comp, 'invt')
comp['invest'] = (comp['capx'] + comp['invt'] - comp['invt_lag'].fillna(0)) / comp['at_lag']

# quick: (act - invt) / lct
comp['quick'] = (comp['act'] - comp['invt']) / comp['lct']

# rd_sale: xrd / sale
comp['rd_sale'] = comp['xrd'].fillna(0) / comp['sale']

# roeq: ib / be_lag
comp['be_lag'] = lag(comp, 'be')
comp['roeq'] = comp['ib'] / comp['be_lag']

# roic: (ib + xint - txp) / (ceq + lt - che)
comp['roic'] = (comp['ib'] + comp['dp'].fillna(0)) / \
               (comp['ceq'] + comp['dltt'].fillna(0))

# salecash: sale / che
comp['salecash'] = comp['sale'] / comp['che']

# saleinv: sale / invt
comp['saleinv'] = comp['sale'] / comp['invt']

# salerec: sale / rect
comp['salerec'] = comp['sale'] / comp['rect']

# secured: dd1 + dd2 + dd3 + dd4 + dd5 + dr (secured debt)
comp['secured'] = comp[['dd1','dd2','dd3','dd4','dd5']].fillna(0).sum(axis=1)

# securedind: indicator for any secured debt
comp['securedind'] = (comp['secured'] > 0).astype(float)

# sgr: sale / sale_lag - 1
comp['sale_lag'] = lag(comp, 'sale')
comp['sgr'] = comp['sale'] / comp['sale_lag'] - 1

# stdacc: std of accruals = std(ib - oancf) / avg(at) over rolling 5 years
comp['accruals'] = comp['ib'] - comp['oancf'].fillna(0)
comp['stdacc'] = comp.groupby('permno')['accruals'].transform(
    lambda x: x.rolling(5, min_periods=3).std()
) / comp.groupby('permno')['at'].transform(
    lambda x: x.rolling(5, min_periods=3).mean()
)

# pchcurrat: % change in currat
comp['currat_lag'] = lag(comp, 'currat')
comp['pchcurrat'] = comp['currat'] / comp['currat_lag'] - 1

# pchdepr: % change in depr
comp['depr_lag'] = lag(comp, 'depr')
comp['pchdepr'] = comp['depr'] / comp['depr_lag'] - 1

# pchgm_pchsale: % change gross margin - % change sales
comp['gm'] = (comp['sale'] - comp['cogs']) / comp['sale']
comp['gm_lag'] = lag(comp, 'gm')
comp['pchgm'] = comp['gm'] / comp['gm_lag'] - 1
comp['pchsale'] = comp['sale'] / comp['sale_lag'] - 1
comp['pchgm_pchsale'] = comp['pchgm'] - comp['pchsale']

# pchquick: % change in quick ratio
comp['quick_lag'] = lag(comp, 'quick')
comp['pchquick'] = comp['quick'] / comp['quick_lag'] - 1

# pchsaleinv: % change in sale/invt
comp['saleinv_lag'] = lag(comp, 'saleinv')
comp['pchsaleinv'] = comp['saleinv'] / comp['saleinv_lag'] - 1

# roavol: std of roaq over 8 quarters — needs quarterly data, approximate annually
# Use std of ib/at over rolling 5 years as annual proxy
comp['roa'] = comp['ib'] / comp['at']
comp['roavol'] = comp.groupby('permno')['roa'].transform(
    lambda x: x.rolling(5, min_periods=3).std()
)

# ── Select output columns ─────────────────────────────────────────────────────
CHARS_22 = [
    'cashdebt', 'currat', 'depr', 'invest', 'mvel1', 'pchcurrat', 'pchdepr',
    'pchgm_pchsale', 'pchquick', 'pchsaleinv', 'quick', 'rd_sale', 'roavol',
    'roeq', 'roic', 'salecash', 'saleinv', 'salerec', 'secured', 'securedind',
    'sgr', 'stdacc'
]

out_cols = ['permno', 'yyyymm'] + CHARS_22
comp_out = comp[out_cols].dropna(subset=['permno']).copy()
comp_out['permno'] = comp_out['permno'].astype(int)
comp_out['yyyymm'] = comp_out['yyyymm'].astype(int)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"Output shape: {comp_out.shape}")
print(f"Date range: {comp_out['yyyymm'].min()} to {comp_out['yyyymm'].max()}")
print(comp_out.head())

# ── Save ──────────────────────────────────────────────────────────────────────
out = DATA_DIR / "compustat_chars.parquet"
comp_out.to_parquet(out, index=False)
print(f"\nSaved to {out}")