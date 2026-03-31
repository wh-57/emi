"""
07c_probe_seed123.py — Add seed 123 to Jaccard screen
======================================================
Seed 123 activations already exist in activations/seed_123/.
This script just runs the linear probe and recomputes Jaccard
across all three seeds: 42, 123, 456.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ACTS_DIR = Path(__file__).resolve().parents[1] / "activations"

TOP_N      = 10
FIRST_TEST = 198701

# ── Load regime labels ────────────────────────────────────────────────────────
regimes = pd.read_csv(DATA_DIR / "monthly_regimes.csv")[['yyyymm', 'nber']]
regimes['yyyymm'] = regimes['yyyymm'].astype(int)

# ── Load panel date spine for test_months ────────────────────────────────────
import glob
panel_files = sorted(glob.glob(str(DATA_DIR / "panel_v1_*.parquet")))
dates = pd.read_parquet(panel_files[-1], columns=['yyyymm'])['yyyymm'].drop_duplicates().sort_values()
all_months  = sorted(dates.tolist())
test_months = [m for m in all_months if m >= FIRST_TEST]
print(f"Test months: {len(test_months)} ({test_months[0]} – {test_months[-1]})")


# ── Probe function ────────────────────────────────────────────────────────────
def probe_neurons(acts_dir, seed):
    print(f"\n  Running linear probe for seed {seed}...")
    layer_sizes = {'layer0': 32, 'layer1': 16, 'layer2': 8}
    monthly_means = {f'{l}_{n}': [] for l, sz in layer_sizes.items()
                     for n in range(sz)}
    months_list = []

    for yyyymm in test_months:
        path = acts_dir / f"acts_{yyyymm}.h5"
        if not path.exists():
            continue
        with h5py.File(path, 'r') as f:
            for layer, sz in layer_sizes.items():
                if layer not in f:
                    continue
                acts  = f[layer][:]
                means = acts.mean(axis=0)
                for n in range(sz):
                    monthly_means[f'{layer}_{n}'].append(means[n])
        months_list.append(yyyymm)

    months_df = pd.DataFrame({'yyyymm': months_list})
    months_df = months_df.merge(regimes, on='yyyymm', how='left')
    nber      = months_df['nber'].fillna(0).values

    tstats = {}
    for key, vals in monthly_means.items():
        if len(vals) != len(nber):
            continue
        arr = np.array(vals)
        if arr.std() < 1e-8:
            tstats[key] = 0.0
            continue
        slope, _, _, _, se = stats.linregress(nber, arr)
        tstats[key] = slope / se if se > 1e-10 else 0.0

    return tstats


def top_n(tstats, n=TOP_N):
    return set(k for k, _ in sorted(tstats.items(),
                                    key=lambda x: abs(x[1]),
                                    reverse=True)[:n])


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ── Run probes ────────────────────────────────────────────────────────────────
seeds = {
    42:  ACTS_DIR,                    # seed 42 activations in root
    123: ACTS_DIR / "seed_123",
    456: ACTS_DIR / "seed_456",
}

all_top10  = {}
all_tstats = {}

for seed, acts_dir in seeds.items():
    tstats = probe_neurons(acts_dir, seed)
    top10  = top_n(tstats)
    all_top10[seed]  = top10
    all_tstats[seed] = tstats
    print(f"  Seed {seed} top-10: {sorted(top10)}")


# ── Jaccard across all 3 seed pairs ──────────────────────────────────────────
print(f"\n{'='*60}")
print("JACCARD OVERLAP — All 3 Seeds")
print(f"{'='*60}")

seed_list  = list(seeds.keys())
pairs      = [(seed_list[i], seed_list[j])
              for i in range(len(seed_list))
              for j in range(i+1, len(seed_list))]

scores = []
for s1, s2 in pairs:
    j = jaccard(all_top10[s1], all_top10[s2])
    scores.append(j)
    print(f"  Seeds {s1} vs {s2}: Jaccard = {j:.3f}  "
          f"(overlap: {sorted(all_top10[s1] & all_top10[s2])})")

mean_j = np.mean(scores)
print(f"\nMean Jaccard (3 seeds): {mean_j:.3f}")

# ── Decision ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DECISION")
print(f"{'='*60}")
if mean_j >= 0.30:
    print(f"✓ Mean Jaccard = {mean_j:.3f} >= 0.30")
    print("  → Proceed NEURON-FIRST in Phase 2 (2A → 2B → 2C → 2D → 2E)")
elif mean_j >= 0.20:
    print(f"⚠ Mean Jaccard = {mean_j:.3f} (borderline 0.20–0.30)")
    print("  → Proceed with caution. Run 2A then 2D (SAE) in parallel with 2B.")
else:
    print(f"✗ Mean Jaccard = {mean_j:.3f} < 0.20")
    print("  → Pivot to SAE-FIRST in Phase 2 (2A → 2D → 2B → 2C → 2E)")

# ── Save updated results ──────────────────────────────────────────────────────
rows = []
for seed, tstats in all_tstats.items():
    for rank, (neuron, tstat) in enumerate(
            sorted(tstats.items(), key=lambda x: abs(x[1]), reverse=True)):
        rows.append({'seed': seed, 'neuron': neuron,
                     'tstat_nber': tstat, 'rank': rank + 1})

pd.DataFrame(rows).to_csv(DATA_DIR / "seed_screen_results.csv", index=False)
pd.DataFrame({'seed_pair': [f"{s1}_vs_{s2}" for s1, s2 in pairs],
              'jaccard': scores}).to_csv(DATA_DIR / "seed_screen_jaccard.csv",
                                         index=False)
print(f"\nUpdated seed_screen_results.csv and seed_screen_jaccard.csv")
print(f"{'='*60}")