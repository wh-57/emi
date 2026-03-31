import openassetpricing as oap
import pandas as pd
from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── GKX 94 characteristics (Table A.6 acronyms) ──────────────────────────────
GKX_94 = [
    'absacc', 'acc', 'aeavol', 'age', 'agr', 'baspread', 'beta', 'betasq',
    'bm', 'bm_ia', 'cash', 'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia',
    'chcsho', 'chempia', 'chinv', 'chmom', 'chpmia', 'chtx', 'cinvest',
    'convind', 'currat', 'depr', 'divi', 'divo', 'dolvol', 'dy', 'ear',
    'egr', 'ep', 'gma', 'grCAPX', 'grltnoa', 'herf', 'hire', 'idiovol',
    'ill', 'indmom', 'invest', 'lev', 'lgr', 'maxret', 'mom12m', 'mom1m',
    'mom36m', 'mom6m', 'ms', 'mvel1', 'mve_ia', 'nincr', 'operprof',
    'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale',
    'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga',
    'pchsaleinv', 'pctacc', 'pricedelay', 'ps', 'quick', 'rd', 'rd_mve',
    'rd_sale', 'realestate', 'retvol', 'roaq', 'roavol', 'roeq', 'roic',
    'rsup', 'salecash', 'saleinv', 'salerec', 'secured', 'securedind',
    'sgr', 'sin', 'sp', 'std_dolvol', 'std_turn', 'stdacc', 'stdcf',
    'tang', 'tb', 'turn', 'zerotrade'
]

# ── Signals to construct from Compustat in 03_compustat_pull.py ──────────────
CONSTRUCT_IN_03 = [
    'aeavol', 'bm_ia', 'cashdebt', 'cfp_ia', 'chempia', 'chpmia', 'currat', 'depr',
    'invest', 'mvel1', 'mve_ia', 'pchcapx_ia', 'pchcurrat', 'pchdepr',
    'pchgm_pchsale', 'pchquick', 'pchsaleinv', 'quick', 'rd_sale', 'roavol',
    'roeq', 'roic', 'salecash', 'saleinv', 'salerec', 'secured', 'securedind',
    'sgr', 'stdacc', 'std_dolvol', 'turn'
]

# ── OAP name mapping ──────────────────────────────────────────────────────────
MANUAL_MAP = {
    'absacc':           'AbnormalAccruals',
    'acc':              'Accruals',
    # 'aeavol':           'AnnouncementReturn',
    'age':              'FirmAge',
    'agr':              'AssetGrowth',
    'baspread':         'BidAskSpread',
    'betasq':           'BetaSquared',
    # 'bm_ia':            'BMdec',
    'cashpr':           'CashProd',
    # 'cfp_ia':           'cfp',
    'chatoia':          'ChAssetTurnover',
    'chcsho':           'CompEquIss',
    # 'chempia':          'hire',
    'chmom':            'MomRev',
    # 'chpmia':           'ChForecastAccrual',
    'chtx':             'ChTax',
    'cinvest':          'Investment',
    'convind':          'ConvDebt',
    'divi':             'DivInit',
    'divo':             'DivOmit',
    'dy':               'DivYieldST',
    'ear':              'EarningsSurprise',
    'egr':              'ChEQ',
    'gma':              'GP',
    'idiovol':          'IdioVol3F',
    'ill':              'Illiquidity',
    'lev':              'Leverage',
    'lgr':              'CompositeDebtIssuance',
    'mom1m':            'MRreversal',
    'mom36m':           'LRreversal',
    'nincr':            'EarningsStreak',
    # 'pchcapx_ia':       'grcapx',
    'pchsale_pchinvt':  'ChInv',
    'pchsale_pchrect':  'ChNWC',
    'pchsale_pchxsga':  'GrAdExp',
    'pricedelay':       'PriceDelayRsq',
    'rd_mve':           'RDS',
    'retvol':           'RealizedVol',
    'rsup':             'RevenueSurprise',
    'sin':              'sinAlgo',
    # 'std_dolvol':       'VolSD',
    'stdcf':            'VarCF',
    'tb':               'Tax',
    # 'turn':             'ShareVol',
    'zerotrade':        'zerotrade6M',
}

# ── Build full matched dict (auto + manual, skip construct-in-03) ─────────────
ap = oap.OpenAP(release_year=202510)
doc = ap.dl_signal_doc(df_backend='pandas')
oap_acronyms = doc['Acronym'].tolist()
oap_lower = {a.lower().replace('_', '').replace('-', ''): a for a in oap_acronyms}

matched = {}
for gkx in GKX_94:
    if gkx in CONSTRUCT_IN_03:
        continue
    key = gkx.lower().replace('_', '').replace('-', '')
    if key in oap_lower:
        matched[gkx] = oap_lower[key]

for gkx, oap_name in MANUAL_MAP.items():
    if gkx not in CONSTRUCT_IN_03:
        matched[gkx] = oap_name

print(f"Matched from OAP: {len(matched)}/{94 - len(CONSTRUCT_IN_03)}")
print(f"To construct in 03: {len(CONSTRUCT_IN_03)}")

# ── Load signals from local unzipped files ────────────────────────────────────
signals_dir = DATA_DIR / "signals_raw"
dfs = []
missing = []

for gkx_name, oap_name in matched.items():
    path = signals_dir / f"{oap_name}.csv"
    if path.exists():
        df = pd.read_csv(path, usecols=['permno', 'yyyymm', oap_name])
        df = df.rename(columns={oap_name: gkx_name})
        dfs.append(df)
    else:
        missing.append(oap_name)

print(f"Loaded: {len(dfs)} signal files")
if missing:
    print(f"Missing files ({len(missing)}):")
    for m in missing:
        print(f"  {m}.csv")

# ── Merge all signals on permno + yyyymm ─────────────────────────────────────
if dfs:
    from functools import reduce
    print("Merging signals...")
    signals = reduce(lambda l, r: pd.merge(l, r, on=['permno', 'yyyymm'], how='outer'), dfs)
    print(f"Shape: {signals.shape}")
    print(signals.head())

    # ── Save ──────────────────────────────────────────────────────────────────
    out = DATA_DIR / "signals_raw.parquet"
    signals.to_parquet(out, index=False)
    print(f"\nSaved to {out}")

print("Done")