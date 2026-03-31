"""
07b_seed_screen.py — Early Seed Stability Screen (flexmap v2, step 1D)
=======================================================================
Trains GKX MLP with seeds 123 and 456, runs a quick linear probe on NBER
regime for all 56 neurons, computes Jaccard overlap of top-10 crisis neurons
against seed 42 baseline.

Decision rule:
  mean Jaccard >= 0.30 → proceed neuron-first in Phase 2 (2B before 2D)
  mean Jaccard < 0.20  → do SAE (2D) before linear probing (2B/2C)

Expected runtime: ~2 x 90 min training + ~10 min probing = ~3 hrs total
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import warnings
import gc
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS_TO_RUN  = [456]
BASELINE_SEED = 42
HIDDEN        = [32, 16, 8]
LR            = 0.001
BATCH_SIZE    = 10_000
EPOCHS        = 100
PATIENCE      = 5
L1_LAMBDA     = 0.001
DROPOUT       = 0.5
FIRST_TEST    = 198701
VAL_YEARS     = 3
TOP_N         = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ACTS_DIR  = Path(__file__).resolve().parents[1] / "activations"

# ── GKX MLP ───────────────────────────────────────────────────────────────────
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

    def register_hooks(self):
        relu_count = 0
        for module in self.net:
            if isinstance(module, nn.ReLU):
                name = f'layer{relu_count}'
                module.register_forward_hook(
                    lambda m, inp, out, n=name:
                    self.activations.update({n: out.detach().cpu()})
                )
                relu_count += 1

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Training function ─────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0
    n = len(X_train)

    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            xb = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            mse  = nn.MSELoss()(pred, yb)
            output_layer = [m for m in model.net if isinstance(m, nn.Linear)][-1]
            l1 = L1_LAMBDA * output_layer.weight.abs().sum()
            (mse + l1).backward()
            optimizer.step()
            del xb, yb
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            val_preds = []
            for vs in range(0, len(X_val), BATCH_SIZE):
                xvb = torch.tensor(
                    X_val[vs:vs+BATCH_SIZE], dtype=torch.float32
                ).to(DEVICE)
                val_preds.append(model(xvb).cpu())
                del xvb
            val_pred_t = torch.cat(val_preds)
            y_v_t      = torch.tensor(y_val, dtype=torch.float32)
            val_loss   = nn.MSELoss()(val_pred_t, y_v_t).item()
            del val_pred_t, y_v_t, val_preds
        torch.cuda.empty_cache()
        model.train()

        if np.isnan(val_loss):
            raise ValueError(f"val_loss NaN at epoch {epoch}")
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    torch.cuda.empty_cache()
    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Load and preprocess panel ─────────────────────────────────────────────────
print("\nLoading panel...")
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'ret_norm', 'prc',
             'shrout', 'exchcd', 'shrcd', 'vol', 'siccd', 'sic2']
panel = pd.read_parquet(DATA_DIR / "panel_v1_20260330.parquet")
CHAR_COLS = [c for c in panel.columns if c not in SKIP_COLS]
print(f"Panel: {panel.shape}, characteristics: {len(CHAR_COLS)}")

panel = panel.dropna(subset=['ret_adj'])
panel = panel.sort_values(['permno', 'yyyymm'])
for col in CHAR_COLS:
    panel[col] = panel.groupby('permno')[col].shift(1)
for col in CHAR_COLS:
    panel[col] = panel[col].fillna(
        panel.groupby('yyyymm')[col].transform('median')
    ).fillna(0)

assert not panel[CHAR_COLS].isna().any().any()

panel['ret_norm'] = panel.groupby('yyyymm')['ret_adj'].transform(
    lambda x: x / (x.std() + 1e-8)
)
p1  = panel['ret_norm'].quantile(0.01)
p99 = panel['ret_norm'].quantile(0.99)
panel['ret_norm'] = panel['ret_norm'].clip(lower=p1, upper=p99)

# ── Load regime labels ────────────────────────────────────────────────────────
regimes = pd.read_csv(DATA_DIR / "monthly_regimes.csv")[['yyyymm', 'nber']]
regimes['yyyymm'] = regimes['yyyymm'].astype(int)

all_months     = sorted(panel['yyyymm'].unique())
test_months    = [m for m in all_months if m >= FIRST_TEST]
retrain_months = [m for m in test_months if str(m)[-2:] == '01']


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Train new seeds and save activations
# ══════════════════════════════════════════════════════════════════════════════
for SEED in SEEDS_TO_RUN:
    print(f"\n{'='*60}")
    print(f"Training seed {SEED}...")
    print(f"{'='*60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    acts_dir_seed = ACTS_DIR / f"seed_{SEED}"
    acts_dir_seed.mkdir(exist_ok=True)

    model = None
    for t_idx, yyyymm in enumerate(test_months):
        if yyyymm in retrain_months:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()

            train_data = panel[panel['yyyymm'] < yyyymm].copy()
            val_cutoff = sorted(train_data['yyyymm'].unique())[-(VAL_YEARS * 12)]
            val_data   = train_data[train_data['yyyymm'] >= val_cutoff]
            train_data = train_data[train_data['yyyymm'] <  val_cutoff]

            X_train = train_data[CHAR_COLS].values.astype(np.float32)
            y_train = train_data['ret_norm'].values.astype(np.float32)
            X_val   = val_data[CHAR_COLS].values.astype(np.float32)
            y_val   = val_data['ret_norm'].values.astype(np.float32)

            model = GKXMLP(n_chars=len(CHAR_COLS), hidden=HIDDEN,
                           dropout=DROPOUT).to(DEVICE)
            model.register_hooks()
            model = train_model(model, X_train, y_train, X_val, y_val)

            ckpt_path = MODEL_DIR / f"checkpoint_{yyyymm}_seed{SEED}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [{yyyymm}] Retrained and saved")

        month_data = panel[panel['yyyymm'] == yyyymm].copy()
        if len(month_data) == 0 or model is None:
            continue

        X_test  = month_data[CHAR_COLS].values.astype(np.float32)
        permnos = month_data['permno'].values

        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            _   = model(X_t).cpu().numpy()
            del X_t
        torch.cuda.empty_cache()

        path = acts_dir_seed / f"acts_{yyyymm}.h5"
        with h5py.File(path, 'w') as f:
            f['permno'] = permnos
            for k, v in model.activations.items():
                f[k] = v.numpy()

        if t_idx % 12 == 0:
            print(f"  Processed {yyyymm} ({t_idx+1}/{len(test_months)})")

    print(f"Seed {SEED} training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Quick linear probe: regress mean monthly activation on NBER
# ══════════════════════════════════════════════════════════════════════════════
def probe_neurons(acts_dir, seed, regimes, test_months):
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
                acts = f[layer][:]
                means = acts.mean(axis=0)
                for n in range(sz):
                    monthly_means[f'{layer}_{n}'].append(means[n])
        months_list.append(yyyymm)

    months_df = pd.DataFrame({'yyyymm': months_list})
    months_df = months_df.merge(regimes, on='yyyymm', how='left')
    nber = months_df['nber'].fillna(0).values

    tstats = {}
    for key, vals in monthly_means.items():
        if len(vals) != len(nber):
            continue
        arr = np.array(vals)
        if arr.std() < 1e-8:
            tstats[key] = 0.0
            continue
        slope, intercept, r, p, se = stats.linregress(nber, arr)
        tstats[key] = slope / se if se > 1e-10 else 0.0

    return tstats


def top_n_neurons(tstats, n=TOP_N):
    sorted_neurons = sorted(tstats.items(),
                            key=lambda x: abs(x[1]), reverse=True)
    return set(k for k, _ in sorted_neurons[:n])


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── Probe seed 42 ─────────────────────────────────────────────────────────────
acts_dir_42 = ACTS_DIR
tstats_42   = probe_neurons(acts_dir_42, BASELINE_SEED, regimes, test_months)
top10_42    = top_n_neurons(tstats_42)
print(f"\nSeed 42 top-10 crisis neurons: {sorted(top10_42)}")

# ── Probe new seeds ───────────────────────────────────────────────────────────
all_top10  = {BASELINE_SEED: top10_42}
all_tstats = {BASELINE_SEED: tstats_42}

for SEED in SEEDS_TO_RUN:
    acts_dir_seed = ACTS_DIR / f"seed_{SEED}"
    tstats  = probe_neurons(acts_dir_seed, SEED, regimes, test_months)
    top10   = top_n_neurons(tstats)
    all_top10[SEED]  = top10
    all_tstats[SEED] = tstats
    print(f"Seed {SEED} top-10 crisis neurons: {sorted(top10)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Jaccard overlap
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("JACCARD OVERLAP — Top-10 Crisis Neurons")
print(f"{'='*60}")

all_seeds   = [BASELINE_SEED] + SEEDS_TO_RUN
seed_pairs  = [(all_seeds[i], all_seeds[j])
               for i in range(len(all_seeds))
               for j in range(i+1, len(all_seeds))]

jaccard_scores = []
for s1, s2 in seed_pairs:
    j = jaccard(all_top10[s1], all_top10[s2])
    jaccard_scores.append(j)
    print(f"  Seeds {s1} vs {s2}: Jaccard = {j:.3f}  "
          f"(overlap: {sorted(all_top10[s1] & all_top10[s2])})")

mean_jaccard = np.mean(jaccard_scores)
print(f"\nMean Jaccard: {mean_jaccard:.3f}")

# ── Decision ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DECISION")
print(f"{'='*60}")
if mean_jaccard >= 0.30:
    print(f"✓ Mean Jaccard = {mean_jaccard:.3f} >= 0.30")
    print("  → Proceed NEURON-FIRST in Phase 2 (2A → 2B → 2C → 2D → 2E)")
elif mean_jaccard >= 0.20:
    print(f"⚠ Mean Jaccard = {mean_jaccard:.3f} (0.20–0.30, borderline)")
    print("  → Proceed with caution. Run 2A then 2D (SAE) in parallel with 2B.")
else:
    print(f"✗ Mean Jaccard = {mean_jaccard:.3f} < 0.20")
    print("  → Pivot to SAE-FIRST in Phase 2 (2A → 2D → 2B → 2C → 2E)")

# ── Save results ──────────────────────────────────────────────────────────────
rows = []
for seed, tstats in all_tstats.items():
    sorted_neurons = sorted(tstats.items(),
                            key=lambda x: abs(x[1]), reverse=True)
    for rank, (neuron, tstat) in enumerate(sorted_neurons):
        rows.append({
            'seed': seed, 'neuron': neuron,
            'tstat_nber': tstat, 'rank': rank + 1
        })

results_df = pd.DataFrame(rows)
results_df.to_csv(DATA_DIR / "seed_screen_results.csv", index=False)

summary_df = pd.DataFrame({
    'seed_pair': [f"{s1}_vs_{s2}" for s1, s2 in seed_pairs],
    'jaccard':   jaccard_scores
})
summary_df.to_csv(DATA_DIR / "seed_screen_jaccard.csv", index=False)
print(f"\nSaved seed_screen_results.csv and seed_screen_jaccard.csv")
print(f"{'='*60}")