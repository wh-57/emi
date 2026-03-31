import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import warnings
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ACTS_DIR  = Path(__file__).resolve().parents[1] / "activations"
MODEL_DIR.mkdir(exist_ok=True)
ACTS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN     = [32, 16, 8]
LR         = 0.001
BATCH_SIZE = 10_000
EPOCHS     = 100
PATIENCE   = 5
L1_LAMBDA  = 0.001
DROPOUT    = 0.5

# ── Load panel ────────────────────────────────────────────────────────────────
print("Loading panel...")
SKIP_COLS = ['permno', 'yyyymm', 'ret_adj', 'ret_norm', 'prc', 'shrout',
             'exchcd', 'shrcd', 'vol', 'siccd', 'sic2']
panel = pd.read_parquet(DATA_DIR / "panel_v1_20260330.parquet")
CHAR_COLS = [c for c in panel.columns if c not in SKIP_COLS]
print(f"Panel shape: {panel.shape}, characteristics: {len(CHAR_COLS)}")

panel = panel.dropna(subset=['ret_adj'])

# ── Option A: shift all characteristics back 1 month (GKX alignment) ─────────
# Signal at (permno, t) predicts return at t+1.
# Impute NaNs with cross-sectional median, then 0 as fallback (GKX convention:
# 0 = no information in the [-1, 1] rank-transformed space).
print("Applying Option A: shifting all characteristics by 1 month...")
panel = panel.sort_values(['permno', 'yyyymm'])
for col in CHAR_COLS:
    panel[col] = panel.groupby('permno')[col].shift(1)
for col in CHAR_COLS:
    panel[col] = panel[col].fillna(
        panel.groupby('yyyymm')[col].transform('median')
    ).fillna(0)  # fallback: entire column NaN for a month → 0

# ── Hard NaN guard ────────────────────────────────────────────────────────────
assert not panel[CHAR_COLS].isna().any().any(), "NaNs remain in characteristics after imputation — abort."
assert not panel['ret_adj'].isna().any(), "NaNs in ret_adj after dropna — abort."
print("Shift complete. No NaNs in features.")
# ─────────────────────────────────────────────────────────────────────────────

panel['ret_norm'] = panel.groupby('yyyymm')['ret_adj'].transform(
    lambda x: x / (x.std() + 1e-8)
)
p1  = panel['ret_norm'].quantile(0.01)
p99 = panel['ret_norm'].quantile(0.99)
panel['ret_norm'] = panel['ret_norm'].clip(lower=p1, upper=p99)

assert not panel['ret_norm'].isna().any(), "NaNs in ret_norm — abort."

# ── GKX MLP ───────────────────────────────────────────────────────────────────
class GKXMLP(nn.Module):
    def __init__(self, n_chars=94, hidden=[32, 16, 8], dropout=0.5):
        super().__init__()
        self.activations = {}
        layers = []
        prev = n_chars
        for i, h in enumerate(hidden):
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
        epoch_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            xb = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            mse  = nn.MSELoss()(pred, yb)

            output_layer = [m for m in model.net if isinstance(m, nn.Linear)][-1]
            l1 = L1_LAMBDA * output_layer.weight.abs().sum()

            loss = mse + l1
            loss.backward()
            optimizer.step()

            del xb, yb
            torch.cuda.empty_cache()
            epoch_loss += mse.item()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_preds = []
            for vs in range(0, len(X_val), BATCH_SIZE):
                xvb = torch.tensor(
                    X_val[vs:vs + BATCH_SIZE], dtype=torch.float32
                ).to(DEVICE)
                val_preds.append(model(xvb).cpu())
                del xvb
            val_pred_t = torch.cat(val_preds)
            y_v_t      = torch.tensor(y_val, dtype=torch.float32)
            val_loss   = nn.MSELoss()(val_pred_t, y_v_t).item()
            del val_pred_t, y_v_t, val_preds
        torch.cuda.empty_cache()
        model.train()

        # ── NaN guard on val_loss ─────────────────────────────────────────────
        if np.isnan(val_loss):
            raise ValueError(
                f"val_loss is NaN at epoch {epoch} — NaN inputs or exploding gradients."
            )

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

# ── Save activations ──────────────────────────────────────────────────────────
def save_activations(model, X, permnos, yyyymm):
    model.eval()
    all_acts = {f'layer{i}': [] for i in range(len(HIDDEN))}
    n = len(X)
    for start in range(0, n, BATCH_SIZE):
        X_batch = torch.tensor(
            X[start:start + BATCH_SIZE], dtype=torch.float32
        ).to(DEVICE)
        with torch.no_grad():
            _ = model(X_batch)
        for k, v in model.activations.items():
            all_acts[k].append(v.numpy())
        del X_batch
    torch.cuda.empty_cache()

    path = ACTS_DIR / f"acts_{yyyymm}.h5"
    with h5py.File(path, 'w') as f:
        f['permno'] = permnos
        for k, v_list in all_acts.items():
            f[k] = np.concatenate(v_list, axis=0)

# ── Expanding window training loop ────────────────────────────────────────────
all_months     = sorted(panel['yyyymm'].unique())
FIRST_TEST     = 198701
VAL_YEARS      = 3

predictions    = []
test_months    = [m for m in all_months if m >= FIRST_TEST]
retrain_months = [m for m in test_months if str(m)[-2:] == '01']

print(f"Total test months  : {len(test_months)}")
print(f"Retrain points     : {len(retrain_months)}")
print(f"Starting training loop...\n")

split_log = []
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

        split_log.append({
        'retrain_yyyymm': yyyymm,
        'train_start':    int(train_data['yyyymm'].min()),
        'train_end':      int(train_data['yyyymm'].max()),
        'val_start':      int(val_data['yyyymm'].min()),
        'val_cutoff':     int(val_cutoff),
        'val_end':        int(val_data['yyyymm'].max()),
        'test_start':     int(yyyymm),
            })
        
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
        print(f"  [{yyyymm}] Retrained and saved checkpoint")

    month_data = panel[panel['yyyymm'] == yyyymm].copy()
    if len(month_data) == 0 or model is None:
        continue

    X_test  = month_data[CHAR_COLS].values.astype(np.float32)
    permnos = month_data['permno'].values

    model.eval()
    with torch.no_grad():
        X_t  = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pred = model(X_t).cpu().numpy()
        del X_t
    torch.cuda.empty_cache()

    # ── NaN guard on predictions ──────────────────────────────────────────────
    if np.isnan(pred).any():
        raise ValueError(f"NaN predictions at {yyyymm} — investigate immediately.")

    month_data['pred'] = pred
    predictions.append(month_data[['permno', 'yyyymm', 'ret_adj', 'pred']])
    save_activations(model, X_test, permnos, yyyymm)

    if t_idx % 12 == 0:
        print(f"  Processed {yyyymm} ({t_idx+1}/{len(test_months)} months)")

# ── Compile predictions ───────────────────────────────────────────────────────
pd.DataFrame(split_log).to_csv(DATA_DIR / "split_log.csv", index=False)
print("Saved split log.")
print("\nCompiling predictions...")
preds_df = pd.concat(predictions, ignore_index=True)

# ── Out-of-sample R² ─────────────────────────────────────────────────────────
preds_df['ret_norm'] = preds_df.groupby('yyyymm')['ret_adj'].transform(
    lambda x: x / (x.std() + 1e-8)
)
p1  = preds_df['ret_norm'].quantile(0.01)
p99 = preds_df['ret_norm'].quantile(0.99)
preds_df['ret_norm'] = preds_df['ret_norm'].clip(lower=p1, upper=p99)

SS_res = ((preds_df['ret_norm'] - preds_df['pred']) ** 2).sum()
SS_tot = (preds_df['ret_norm'] ** 2).sum()
r2_oos = 1 - SS_res / SS_tot
print(f"\nOut-of-sample R²: {r2_oos:.4%}")
print(f"Target range: 0.25% – 0.60%")
if r2_oos < 0.002:
    print("WARNING: R² below 0.20% — debug before proceeding to MI")
elif r2_oos <= 0.006:
    print("✓ R² in acceptable range")
else:
    print("R² above 0.60% — may indicate overfitting, check")

# ── Long-short decile Sharpe ──────────────────────────────────────────────────
preds_df['decile'] = preds_df.groupby('yyyymm')['pred'].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
)
ls_returns = preds_df.groupby('yyyymm').apply(
    lambda x: x[x['decile'] == 9]['ret_adj'].mean() - x[x['decile'] == 0]['ret_adj'].mean(),
    include_groups=False
)
sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(12)
print(f"Long-short decile Sharpe: {sharpe:.3f}")
print(f"Target: > 1.5")

# ── Save predictions ──────────────────────────────────────────────────────────
out = DATA_DIR / f"predictions_seed{SEED}.parquet"
preds_df.to_parquet(out, index=False)
print(f"\nSaved predictions to {out}")
print(f"Activation files saved to {ACTS_DIR}")