"""
06_test_1987.py — Look-Ahead Bias Diagnostic for GKX Replication
=================================================================
Trains on 1963–1986, tests on 1987.
Compares three alignment strategies:
  - Current  : no shift (biased baseline)
  - Option A : shift all characteristics back 1 month (blanket lag)
  - Option C : shift target forward 1 month (forward return)

Key outputs
-----------
1. maxret same-month vs next-month correlation  ← primary falsification test
2. Ridge R² on 1987 for each variant            ← should be ~0.1–0.4% if unbiased
3. Long-short decile Sharpe (1987)              ← should be ~2–3 annualized if unbiased
4. Side-by-side summary table

Fixes vs DeepSeek's draft
--------------------------
- Missing characteristics imputed with cross-sectional median (not dropped)
- Option C ret_norm uses same standardization as Option A for comparability
- Kimi's maxret diagnostic added as first output
- Kimi's 4th "shift_merge" variant omitted (mathematically identical to A)
- Kimi's tiny NN omitted for this diagnostic (isolate data alignment, not model)

Expected runtime: ~5–10 minutes
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PANEL_PATH = DATA_DIR / "panel_v1_20260328.parquet"

# ── Sample windows ───────────────────────────────────────────────────────────
TRAIN_START = 196301
TRAIN_END   = 198612
TEST_START  = 198701
TEST_END    = 198712


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_panel():
    print(f"Loading panel from {PANEL_PATH} ...")
    panel = pd.read_parquet(PANEL_PATH)

    SKIP = {"permno", "yyyymm", "ret_adj", "prc", "shrout",
            "exchcd", "shrcd", "date"}
    char_cols = [c for c in panel.columns if c not in SKIP]

    print(f"  Observations : {len(panel):,}")
    print(f"  Period       : {panel['yyyymm'].min()} – {panel['yyyymm'].max()}")
    print(f"  Characteristics: {len(char_cols)}")

    # Restrict to training + test window for speed
    panel = panel[panel["yyyymm"].between(TRAIN_START, TEST_END)].copy()
    print(f"  After date filter: {len(panel):,} rows\n")

    return panel, char_cols


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def rank_transform_month(df: pd.DataFrame, char_cols: list) -> pd.DataFrame:
    """
    Cross-sectional rank → [-1, 1] following GKX.
    Missing values are filled with 0 (cross-sectional median of the
    rank-transformed distribution).
    """
    out = df.copy()
    for col in char_cols:
        if col not in out.columns:
            continue
        x = out[col].astype(float)
        # Winsorize at 1st / 99th percentile
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        x = x.clip(lo, hi)
        # Rank → [-1, 1]
        ranked = x.rank(method="average", na_option="keep")
        n = ranked.notna().sum()
        if n > 1:
            x = 2 * (ranked - 1) / (n - 1) - 1
        else:
            x = ranked * 0.0
        x = x.fillna(0.0)   # missing → median of transformed (= 0)
        out[col] = x
    return out


def impute_cross_sectional_median(panel: pd.DataFrame,
                                  char_cols: list) -> pd.DataFrame:
    """
    Fill NaNs with cross-sectional median for each (yyyymm, col).
    Applied AFTER shifting so imputation uses contemporaneous peers.
    """
    for col in char_cols:
        panel[col] = panel[col].fillna(
            panel.groupby("yyyymm")[col].transform("median")
        )
    return panel


def normalize_returns(panel: pd.DataFrame,
                      ret_col: str = "ret_adj") -> pd.DataFrame:
    """Standardize returns cross-sectionally within each month."""
    panel["ret_norm"] = panel.groupby("yyyymm")[ret_col].transform(
        lambda x: x / (x.std() + 1e-8)
    )
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# 3.  VARIANT CONSTRUCTORS
# ═══════════════════════════════════════════════════════════════════════════

def prepare_current(panel: pd.DataFrame, char_cols: list) -> pd.DataFrame:
    """
    Biased baseline — signals at t predict returns at t (look-ahead).
    """
    print("Building: Current (biased) ...")
    p = panel.copy().sort_values(["permno", "yyyymm"])
    p = normalize_returns(p, "ret_adj")
    p = p.groupby("yyyymm", group_keys=False).apply(
        lambda df: rank_transform_month(df, char_cols)
    )
    p = p.dropna(subset=["ret_norm"])
    print(f"  Rows: {len(p):,}")
    return p


def prepare_option_a(panel: pd.DataFrame, char_cols: list) -> pd.DataFrame:
    """
    Option A — shift ALL characteristics back 1 month.
    Signal at (permno, t) → predict return at t+1.
    Missing after shift imputed with cross-sectional median, then ranked.
    """
    print("Building: Option A (blanket shift features) ...")
    p = panel.copy().sort_values(["permno", "yyyymm"])

    # Shift characteristics
    for col in char_cols:
        p[col] = p.groupby("permno")[col].shift(1)

    # Impute (median, not drop — avoids survivorship tilt)
    p = impute_cross_sectional_median(p, char_cols)

    # Normalize same-month returns (month t+1 from the signal's perspective)
    p = normalize_returns(p, "ret_adj")

    # Rank-transform characteristics
    p = p.groupby("yyyymm", group_keys=False).apply(
        lambda df: rank_transform_month(df, char_cols)
    )

    p = p.dropna(subset=["ret_norm"])
    print(f"  Rows: {len(p):,}")
    return p


def prepare_option_c(panel: pd.DataFrame, char_cols: list) -> pd.DataFrame:
    """
    Option C — keep signals at t, shift TARGET forward 1 month.
    ret_fwd at (permno, t) = ret_adj at (permno, t+1).
    Normalized the same way as Option A for comparability.
    """
    print("Building: Option C (shift target forward) ...")
    p = panel.copy().sort_values(["permno", "yyyymm"])

    # Forward return
    p["ret_fwd"] = p.groupby("permno")["ret_adj"].shift(-1)

    # Impute characteristics (no shift)
    p = impute_cross_sectional_median(p, char_cols)

    # Normalize FORWARD returns cross-sectionally — same recipe as Option A
    p["ret_norm"] = p.groupby("yyyymm")["ret_fwd"].transform(
        lambda x: x / (x.std() + 1e-8)
    )

    # Rank-transform characteristics
    p = p.groupby("yyyymm", group_keys=False).apply(
        lambda df: rank_transform_month(df, char_cols)
    )

    p = p.dropna(subset=["ret_norm"])
    print(f"  Rows: {len(p):,}")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# 4.  KIMI'S MAXRET DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def maxret_diagnostic(panel_raw: pd.DataFrame,
                      variants: dict) -> None:
    """
    For each variant, compute average cross-sectional correlation between
    maxret and SAME-month vs NEXT-month raw returns.

    Correct alignment  → same-month corr ≈ 0,  next-month corr < 0
                         (maxret = prior-month max daily ret, predicts reversal)
    Biased alignment   → same-month corr >> 0  (contemporaneous contamination)
    """
    print("\n" + "=" * 65)
    print("MAXRET DIAGNOSTIC  (Kimi's falsification test)")
    print("  Correct fix  → same-month corr ≈ 0")
    print("  Biased       → same-month corr >> 0")
    print("=" * 65)

    # We need the raw (unranked) maxret aligned to 1987 months
    raw87 = panel_raw[panel_raw["yyyymm"].between(TEST_START, TEST_END)].copy()
    raw87 = raw87.sort_values(["permno", "yyyymm"])
    raw87["ret_next"] = raw87.groupby("permno")["ret_adj"].shift(-1)

    for name, data in variants.items():
        test = data[data["yyyymm"].between(TEST_START, TEST_END)]

        # Pull maxret from this variant (already rank-transformed, just use sign)
        if "maxret" not in test.columns:
            print(f"  {name}: maxret not found, skip")
            continue

        # Merge with raw returns to get same- and next-month ret_adj
        merged = test[["permno", "yyyymm", "maxret"]].merge(
            raw87[["permno", "yyyymm", "ret_adj", "ret_next"]],
            on=["permno", "yyyymm"], how="inner"
        )

        same_corr = merged.groupby("yyyymm").apply(
            lambda x: x["maxret"].corr(x["ret_adj"])
        ).mean()

        next_corr = merged.groupby("yyyymm").apply(
            lambda x: x["maxret"].corr(x["ret_next"])
        ).mean()

        print(f"  {name:<35}  same={same_corr:+.4f}  next={next_corr:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MODEL EVALUATION  (Ridge + time-series CV)
# ═══════════════════════════════════════════════════════════════════════════

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def fit_ridge(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # Time-series CV to pick alpha
    tscv = TimeSeriesSplit(n_splits=3)
    best_alpha, best_score = 1.0, -np.inf
    for alpha in RIDGE_ALPHAS:
        scores = []
        for tr_idx, va_idx in tscv.split(Xtr):
            m = Ridge(alpha=alpha).fit(Xtr[tr_idx], y_train[tr_idx])
            scores.append(m.score(Xtr[va_idx], y_train[va_idx]))
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = alpha

    model = Ridge(alpha=best_alpha).fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(y_pred, y_test)[0, 1] if len(y_test) > 1 else np.nan

    return r2, corr, best_alpha, y_pred


def evaluate_variant(name: str, data: pd.DataFrame,
                     char_cols: list) -> dict:
    train = data[data["yyyymm"] < TEST_START].dropna(subset=["ret_norm"])
    test  = data[data["yyyymm"].between(TEST_START, TEST_END)].dropna(subset=["ret_norm"])

    print(f"\n  {name}")
    print(f"    Train: {train['yyyymm'].min()}–{train['yyyymm'].max()}  ({len(train):,} obs)")
    print(f"    Test : {test['yyyymm'].min()}–{test['yyyymm'].max()}  ({len(test):,} obs)")

    if len(train) == 0 or len(test) == 0:
        return {"r2": np.nan, "corr": np.nan, "sharpe": np.nan,
                "n_train": 0, "n_test": 0, "alpha": np.nan}

    X_train = train[char_cols].values.astype(np.float32)
    y_train = train["ret_norm"].values.astype(np.float32)
    X_test  = test[char_cols].values.astype(np.float32)
    y_test  = test["ret_norm"].values.astype(np.float32)

    r2, corr, alpha, y_pred = fit_ridge(X_train, y_train, X_test, y_test)
    print(f"    Ridge alpha: {alpha}  |  R²: {r2:.4%}  |  Corr: {corr:.4f}")

    # Long-short Sharpe on 1987
    test = test.copy()
    test["y_pred"] = y_pred
    sharpe = long_short_sharpe(test, data, char_cols)
    print(f"    L/S Sharpe (1987, annualized): {sharpe:.2f}")

    return {"r2": r2, "corr": corr, "sharpe": sharpe,
            "n_train": len(train), "n_test": len(test), "alpha": alpha}


# ═══════════════════════════════════════════════════════════════════════════
# 6.  LONG-SHORT SHARPE
# ═══════════════════════════════════════════════════════════════════════════

def long_short_sharpe(test_data: pd.DataFrame,
                      full_data: pd.DataFrame,
                      char_cols: list) -> float:
    """
    Simple equal-weight top-decile minus bottom-decile portfolio.
    Uses raw ret_adj for actual returns.
    """
    if "y_pred" not in test_data.columns:
        return np.nan

    monthly_returns = []
    for yyyymm, grp in test_data.groupby("yyyymm"):
        grp = grp.copy()
        grp["decile"] = pd.qcut(grp["y_pred"], 10,
                                labels=False, duplicates="drop")
        # Map back to raw ret_adj
        raw = full_data[full_data["yyyymm"] == yyyymm][["permno", "ret_adj"]]
        grp = grp.merge(raw, on="permno", how="left", suffixes=("", "_raw"))
        ret_col = "ret_adj_raw" if "ret_adj_raw" in grp.columns else "ret_adj"

        long_ret  = grp[grp["decile"] == 9][ret_col].mean()
        short_ret = grp[grp["decile"] == 0][ret_col].mean()
        monthly_returns.append(long_ret - short_ret)

    if len(monthly_returns) < 2:
        return np.nan

    r = np.array(monthly_returns)
    return (r.mean() / (r.std() + 1e-10)) * np.sqrt(12)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("LOOK-AHEAD BIAS DIAGNOSTIC  —  GKX Replication")
    print("Train 1963–1986  |  Test 1987")
    print("=" * 65 + "\n")

    panel, char_cols = load_panel()

    # ── Build variants ──────────────────────────────────────────────────
    variants = {
        "Current (biased)"          : prepare_current(panel, char_cols),
        "Option A (shift features)" : prepare_option_a(panel, char_cols),
        "Option C (shift target)"   : prepare_option_c(panel, char_cols),
    }

    # ── Kimi's maxret diagnostic ────────────────────────────────────────
    maxret_diagnostic(panel, variants)

    # ── Model evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RIDGE REGRESSION RESULTS")
    print("=" * 65)

    results = {}
    for name, data in variants.items():
        results[name] = evaluate_variant(name, data, char_cols)

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    hdr = f"{'Variant':<35} {'R²':>8} {'Corr':>8} {'Sharpe':>8} {'N_train':>9}"
    print(hdr)
    print("-" * 65)
    for name, m in results.items():
        print(
            f"{name:<35} "
            f"{m['r2']:>8.4%} "
            f"{m['corr']:>8.4f} "
            f"{m['sharpe']:>8.2f} "
            f"{m['n_train']:>9,}"
        )

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)

    cur_r2 = results["Current (biased)"]["r2"]
    a_r2   = results["Option A (shift features)"]["r2"]
    c_r2   = results["Option C (shift target)"]["r2"]

    print(f"Current R²  : {cur_r2:.4%}  (expect >>5% if biased)")
    print(f"Option A R² : {a_r2:.4%}   (expect 0.1–0.6% if correct)")
    print(f"Option C R² : {c_r2:.4%}   (expect 0.1–0.6% if correct)")

    if cur_r2 > 0.05:
        print("\n✓ Severe look-ahead bias confirmed in current pipeline.")
    else:
        print("\n⚠ Current R² unexpectedly low — check data or sample filter.")

    gap = abs(a_r2 - c_r2)
    if gap < 0.001:
        print("✓ Option A ≈ Option C — alignment is the only issue.")
        print("  → Use Option A (shift features) to match GKX convention.")
    else:
        print(f"⚠ Option A and C differ by {gap:.4%} — investigate pipeline.")

    print("\nTarget R² range for unbiased GKX replication: 0.10–0.60%")
    print("Target annualized Sharpe (L/S decile, 1987): 1.5–3.0")
    print("=" * 65)


if __name__ == "__main__":
    main()
