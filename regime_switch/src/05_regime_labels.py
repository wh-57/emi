import pandas as pd
import numpy as np
from fredapi import Fred
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ── FRED connection ───────────────────────────────────────────────────────────
fred = Fred(api_key='f567e3eeba77b70588031a6329c4e5bb')

# ── Date spine from panel ─────────────────────────────────────────────────────
print("Loading panel date spine...")
panel = pd.read_parquet(DATA_DIR / "panel_v1_20260328.parquet", columns=['yyyymm'])
dates = panel['yyyymm'].drop_duplicates().sort_values()
date_index = pd.to_datetime(dates.astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
date_df = pd.DataFrame({'date': date_index, 'yyyymm': dates.values})

print(f"Date spine: {date_df['date'].min()} to {date_df['date'].max()}, {len(date_df)} months")

# ── Helper: fetch FRED series and resample to month-end ──────────────────────
def get_fred(series_id, start='1962-01-01', end='2023-12-31'):
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s

# ── 1. NBER recession indicator ───────────────────────────────────────────────
print("Fetching NBER recession dates...")
nber = get_fred('USRECM')
nber.name = 'nber'

# ── 2. VIX ───────────────────────────────────────────────────────────────────
print("Fetching VIX...")
vix_raw = get_fred('VIXCLS')
vix = vix_raw.resample('ME').mean()
vix.name = 'vix'

# ── 3. Baa and Aaa yields ────────────────────────────────────────────────────
print("Fetching credit spreads...")
baa = get_fred('BAA').resample('ME').mean()
aaa = get_fred('AAA').resample('ME').mean()
credit_spread = (baa - aaa)
credit_spread.name = 'credit_spread'

# ── 4. CFNAI-MA3 ─────────────────────────────────────────────────────────────
print("Fetching CFNAI...")
cfnai = get_fred('CFNAI').resample('ME').mean()
cfnai_ma3 = cfnai.rolling(3).mean()
cfnai_ma3.name = 'cfnai_ma3'

# ── 5. NFCI ──────────────────────────────────────────────────────────────────
print("Fetching NFCI...")
nfci = get_fred('NFCI').resample('ME').mean()
nfci.name = 'nfci'

# ── 6. Macro alignment variables ─────────────────────────────────────────────
print("Fetching macro variables...")

# INDPRO growth (yoy)
indpro = get_fred('INDPRO').resample('ME').last()
indpro_growth = indpro.pct_change(12)
indpro_growth.name = 'indpro_growth'

# Real PCE growth (yoy)
pce = get_fred('PCEC96').resample('ME').last()
pce_growth = pce.pct_change(12)
pce_growth.name = 'pce_growth'

# Unemployment change (mom)
unrate = get_fred('UNRATE').resample('ME').last()
unrate_chg = unrate.diff(1)
unrate_chg.name = 'unrate_chg'

# 10yr - 3mo yield spread
t10y = get_fred('GS10').resample('ME').mean()
t3m  = get_fred('TB3MS').resample('ME').mean()
yield_spread = t10y - t3m
yield_spread.name = 'yield_spread'

# ADS proxy (CFNAI used as proxy — replace with Philadelphia Fed ADS if needed)
ads_proxy = cfnai.copy()
ads_proxy.name = 'ads_proxy'

# Baker-Bloom-Davis EPU
print("Fetching EPU...")
try:
    epu = get_fred('USEPUINDXD').resample('ME').mean()
    epu.name = 'epu'
except:
    print("  EPU not available on FRED, skipping...")
    epu = pd.Series(dtype=float, name='epu')

# ── 7. He-Kelly-Manela intermediary capital ratio ─────────────────────────────
print("Loading HKM data...")
hkm_raw = pd.read_csv(DATA_DIR / "He_Kelly_Manela_Factors_quarterly_250627.csv")
hkm_raw.columns = hkm_raw.columns.str.strip()

def yyyyq_to_date(yyyyq):
    y = int(str(yyyyq)[:4])
    q = int(str(yyyyq)[4])
    month = q * 3
    return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(0)

hkm_raw['date'] = hkm_raw['yyyyq'].apply(yyyyq_to_date)
hkm_raw = hkm_raw[['date', 'intermediary_capital_ratio']].rename(
    columns={'intermediary_capital_ratio': 'hkm_cap_ratio'}
)

# Forward-fill quarterly to monthly
hkm_monthly = date_df[['date']].merge(hkm_raw, on='date', how='left')
hkm_monthly['hkm_cap_ratio'] = hkm_monthly['hkm_cap_ratio'].ffill()
hkm_median = hkm_monthly['hkm_cap_ratio'].median()
hkm_monthly['hkm_low'] = (hkm_monthly['hkm_cap_ratio'] < hkm_median).astype(float)
hkm_series = hkm_monthly.set_index('date')['hkm_cap_ratio']
hkm_series.name = 'hkm_cap_ratio'

# ── Combine all series ────────────────────────────────────────────────────────
print("Combining series...")
all_series = [nber, vix, credit_spread, cfnai_ma3, nfci,
              indpro_growth, pce_growth, unrate_chg, yield_spread,
              ads_proxy, epu, hkm_series]

macro = pd.concat(all_series, axis=1)
macro.index.name = 'date'
macro = macro.reset_index()

# ── Merge onto panel date spine ───────────────────────────────────────────────
regimes = date_df.merge(macro, on='date', how='left')
regimes = regimes.merge(hkm_monthly[['date', 'hkm_low']], on='date', how='left')

# ── Construct binary regime indicators ───────────────────────────────────────
regimes['vix25']     = (regimes['vix'] > 25).astype(float)
regimes['credit15']  = (regimes['credit_spread'] > 1.5).astype(float)
regimes['cfnai_neg'] = (regimes['cfnai_ma3'] < -0.70).astype(float)
regimes['nfci05']    = (regimes['nfci'] > 0.5).astype(float)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\n── Regime month counts ──────────────────────────────────────────")
print(f"NBER recession months:       {regimes['nber'].sum():.0f}")
print(f"VIX > 25 months:             {regimes['vix25'].sum():.0f}")
print(f"Credit spread > 1.5% months: {regimes['credit15'].sum():.0f}")
print(f"CFNAI-MA3 < -0.70 months:   {regimes['cfnai_neg'].sum():.0f}")
print(f"NFCI > 0.5 months:           {regimes['nfci05'].sum():.0f}")
print(f"HKM below median months:     {regimes['hkm_low'].sum():.0f}")
print(f"\nTotal months in sample: {len(regimes)}")
print(f"Missing NBER: {regimes['nber'].isna().sum()}")

# ── Power check ───────────────────────────────────────────────────────────────
print("\n── Power check ──────────────────────────────────────────────────")
for col, label in [('nber','NBER'), ('vix25','VIX>25'),
                   ('credit15','Credit>1.5%'), ('cfnai_neg','CFNAI<-0.70'),
                   ('nfci05','NFCI>0.5'), ('hkm_low','HKM below median')]:
    count = regimes[col].sum()
    flag = " ✓" if count >= 60 else " ✗ POWER CONCERN"
    print(f"{label}: {count:.0f} months{flag}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = DATA_DIR / "monthly_regimes.csv"
regimes.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(regimes.head())