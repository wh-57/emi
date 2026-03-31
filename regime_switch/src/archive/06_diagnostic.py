import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r'C:\Users\willi\Desktop\emi\regime_switch\data')

panel = pd.read_parquet(DATA_DIR / 'panel_v1_20260328.parquet')
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'prc', 'shrout', 'exchcd', 'shrcd']
CHAR_COLS = [c for c in panel.columns if c not in SKIP_COLS]

panel = panel.dropna(subset=['ret_adj'])

# ── Test 1: correlation of each characteristic with same-month return ─────────
print("=== Test 1: Cross-sectional correlation of characteristics with ret_adj ===")
print("(High correlation = characteristic contains same-month return info)\n")

results = []
for col in CHAR_COLS:
    monthly_corr = panel.groupby('yyyymm').apply(
        lambda x: x[col].corr(x['ret_adj']), include_groups=False
    )
    results.append({'char': col, 'mean_corr': monthly_corr.mean(), 'std_corr': monthly_corr.std()})

results_df = pd.DataFrame(results).sort_values('mean_corr', ascending=False)
print("Top 20 characteristics by correlation with same-month return:")
print(results_df.head(20).to_string(index=False))
print("\nBottom 10:")
print(results_df.tail(10).to_string(index=False))

# ── Test 2: correlation with NEXT month return ────────────────────────────────
print("\n=== Test 2: Cross-sectional correlation with NEXT month return ===")
print("(This is what the model should be predicting)\n")

panel['ret_next'] = panel.groupby('permno')['ret_adj'].shift(-1)
results2 = []
for col in CHAR_COLS:
    monthly_corr = panel.dropna(subset=['ret_next']).groupby('yyyymm').apply(
        lambda x: x[col].corr(x['ret_next']), include_groups=False
    )
    results2.append({'char': col, 'mean_corr_next': monthly_corr.mean()})

results2_df = pd.DataFrame(results2).sort_values('mean_corr_next', ascending=False)
print("Top 20 characteristics by correlation with NEXT month return:")
print(results2_df.head(20).to_string(index=False))