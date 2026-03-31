\# Preprocessing Notes

\## Step 1A: CRSP/Compustat Data Pipeline



\### CRSP Monthly Pull (`01\_crsp\_pull.py`)

\- Universe: shrcd ∈ {10,11}, exchcd ∈ {1,2,3}, date 1963–2023

\- Delisting returns: pulled separately from `crsp.msedelist`, merged on permno + month-end date

\- Shumway (1997) imputation for missing dlret: -0.30 NYSE/AMEX (exchcd 1,2), -0.55 NASDAQ (exchcd 3)

\- ret\_adj: compounded ret and dlret\_adj in final delisting month

\- Output: `crsp\_monthly\_raw.parquet`



\### Signal Download (`02\_signals\_download.py`)

\- Source: Open Source Asset Pricing (Chen \& Zimmermann 2022), release 202510

\- 72 of 94 GKX characteristics downloaded directly from OAP `PredictorsIndiv.zip`

\- OAP acronyms differ from GKX — full mapping maintained in MANUAL\_MAP dict in script

\- Key mapping decisions:

&#x20; - `betasq` → OAP `Beta` (no separate BetaSquared available)

&#x20; - `mom1m` → OAP `MRreversal` (1-month reversal)

&#x20; - `stdacc` mapped to OAP `Accruals` (AccrualQuality not available in release)

\- 22 characteristics not available in OAP, constructed from Compustat (see below)

\- Output: `signals\_raw.parquet`



\### Compustat Pull \& Characteristic Construction (`03\_compustat\_pull.py`)

\- Source: `comp.funda`, filters: indfmt=INDL, datafmt=STD, popsrc=D, consol=C

\- CCM merge: `crsp.ccmxpf\_lnkhist`, linktype ∈ {LU, LC}, linkprim ∈ {P, C}

\- \*\*6-month reporting lag\*\*: public\_date = datadate + 6 months (single most common look-ahead bug)

\- 22 characteristics constructed from raw Compustat items:

&#x20; - `cashdebt`: (ib + dp) / lt

&#x20; - `currat`: act / lct

&#x20; - `depr`: dp / ppent

&#x20; - `invest`: (capx + Δinvt) / at\_lag

&#x20; - `mvel1`: prcc\_f \* csho (fiscal year-end market cap proxy; overwritten by CRSP in 04)

&#x20; - `pchcurrat`: % change in currat

&#x20; - `pchdepr`: % change in depr

&#x20; - `pchgm\_pchsale`: % change gross margin minus % change sales

&#x20; - `pchquick`: % change in quick ratio

&#x20; - `pchsaleinv`: % change in sale/invt

&#x20; - `quick`: (act - invt) / lct

&#x20; - `rd\_sale`: xrd / sale

&#x20; - `roavol`: rolling 5-year std of ib/at (annual proxy for quarterly roavol)

&#x20; - `roeq`: ib / be\_lag

&#x20; - `roic`: (ib + dp) / (ceq + dltt)

&#x20; - `salecash`: sale / che

&#x20; - `saleinv`: sale / invt

&#x20; - `salerec`: sale / rect

&#x20; - `secured`: sum of dd1–dd5 (secured debt tranches)

&#x20; - `securedind`: indicator for any secured debt

&#x20; - `sgr`: sale / sale\_lag - 1

&#x20; - `stdacc`: rolling 5-year std of (ib - oancf) / avg(at)

\- Output: `compustat\_chars.parquet`



\### Panel Assembly \& Preprocessing (`04\_preprocess.py`)

\- Merge: CRSP (left) ← signals (left join) ← Compustat chars (left join), on permno + yyyymm

\- `mvel1` overwritten with CRSP-based market cap: abs(prc) \* shrout

\- Date filter: 1963-01 to 2023-12

\- Cross-sectional preprocessing applied each month (GKX convention):

&#x20; - (a) Winsorise at 1st/99th percentile

&#x20; - (b) Rank-transform to \[-1, +1]

&#x20; - (c) Set missing values to 0

\- Final output: `panel\_v1\_YYYYMMDD.parquet` — never overwrite, always create new version

\- Sanity checks passed: \~4,610 stock-months/month, mean ret\_adj \~1.1%, char range \[-1, 1]


## Known Deviations from GKX (2020)

### roavol
GKX construct roavol as the rolling standard deviation of quarterly ROA over 8 quarters 
(minimum 2 years) using comp.fundq. This pipeline uses an annual proxy: rolling 5-year 
standard deviation of annual ib/at (minimum 3 years) from comp.funda. The annual version 
is coarser and will differ from GKX's signal, particularly for firms with volatile 
intra-year earnings. This is a documented deviation.
### stdacc
GKX construct stdacc as the rolling standard deviation of quarterly working-capital 
accruals (Sloan 1996 definition: ΔCA - ΔCash - ΔCL + ΔSTD + ΔTP - DEP) scaled by 
average assets, over 16 quarters. This pipeline uses annual (ib - oancf) / avg(at) 
over a 5-year rolling window. Both measure accrual volatility but use different accrual 
definitions and frequencies. This is a documented deviation.

### invest
Originally constructed incorrectly as (capx + Δinvt) / at_lag. Corrected in v2 to 
(at - at_lag) / at_lag following Titman, Wei & Xie (2004). Panel regenerated from 
scratch after this fix.