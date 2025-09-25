import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, matthews_corrcoef,
    mean_absolute_error
)
from statsmodels.stats.weightstats import ttest_ind


# only used for naming output files and finding input files
window = 20
test_size = 0.5

# Toggle: feed pre-shifted exog into the decoder
FEED_FUTURE_EXOG = True


# === Your pre-split data ===
setting = "nointerest"
csv_src_train = Path(f"~/Dropbox/elvas-thesis/timesNet/data/excluding-exogs/timesnet_energy_var6_window{window}_testsize{test_size}_{setting}_train.csv").expanduser()
csv_src_test  = Path(f"~/Dropbox/elvas-thesis/timesNet/data/excluding-exogs/timesnet_energy_var6_window{window}_testsize{test_size}_{setting}_test.csv").expanduser()
#checkpoint_dir = f"short_term_forecast_energy_rollin{window}_testsize{test_size}_TimesNet_custom_ftM_sl48_ll0_pl20_dm64_nh2_el1_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
checkpoint_dir = f"short_term_forecast_energy_{setting}_TimesNet_custom_ftM_sl48_ll0_pl20_dm64_nh2_el1_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"

#out_dir = Path(f"plots_timesnet_block_var6_window{window}_testsize{test_size}"); out_dir.mkdir(exist_ok=True, parents=True)
out_dir = Path(f"plots_timesnet_block_{setting}"); out_dir.mkdir(exist_ok=True, parents=True)

# only used for volatility regime later:
csv_path_regime = Path("~/Dropbox/elvas-thesis/varx/data/dataset_stationary_with_us10y_final.csv").expanduser()
regime_path = "./plots_timesnet_block"
# train test
df_train = pd.read_csv(csv_src_train, parse_dates=["date"]).sort_values("date").set_index("date")
df_test  = pd.read_csv(csv_src_test,  parse_dates=["date"]).sort_values("date").set_index("date")

# Full timeline (useful if you ever want future exog); not strictly needed when we zero x_dec
df_all = pd.concat([df_train, df_test], axis=0)

# === TimesNet imports (repo stays untouched) ===
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.timefeatures import time_features as build_time_features

torch.set_grad_enabled(False)

# === Config (match your training run) ===
target_col = "ENEL_logret"
feature_cols = [
    "ENEL_logret", "NKT_logret", "VWS_logret",
    #"brent_price_logret", "utilities_close_logret", "us10y",
]
seq_len       = 48
label_len     = 0
block_pred_len= 20           # you trained with --pred_len 20
horizon       = len(df_test) # forecast exactly the test span
stride        = 20           # use 1 for max stability; 20 to consume full blocks

# Build a small args object the Exp class expects (match your hyperparams)
class Dot: pass

args = Dot()
args.task_name     = "short_term_forecast"
args.is_training   = 0
args.model_id      = "TN_p20"
args.model         = "TimesNet"
args.data          = "custom"
args.root_path     = str(csv_src_train.parent)
args.data_path     = csv_src_train.name
args.features      = "M"
args.target        = target_col
args.freq          = "b"  # business days
args.seq_len       = seq_len
args.label_len     = label_len
args.pred_len      = block_pred_len
args.enc_in        = len(feature_cols)
args.dec_in        = len(feature_cols)
args.c_out         = len(feature_cols)  # you trained with --c_out 6
args.d_model       = 64
args.d_ff          = 128
args.e_layers      = 1
args.n_heads       = 2
args.top_k         = 3
args.num_kernels   = 3
args.dropout       = 0.1
args.batch_size    = 8
args.train_epochs  = 10
args.patience      = 3
args.learning_rate = 5e-4
args.num_workers   = 0
args.use_gpu       = True
args.gpu           = 0
args.gpu_type      = "cuda"
args.use_multi_gpu = False
args.devices       = "0"
args.seasonal_patterns = "Monthly"
args.embed       = "timeF"   # <—  missing attribute (time features encoding)
args.activation  = "gelu"
args.moving_avg  = block_pred_len
args.factor      = 1
args.distil      = True
args.decomp_method = "moving_avg"
args.use_norm    = 1
args.down_sampling_layers  = 0
args.down_sampling_window  = 1
args.down_sampling_method  = None
args.seg_len     = 96

device = torch.device(f"cuda:{args.gpu}") if (args.use_gpu and torch.cuda.is_available()) else torch.device("cpu")

# Init experiment + model (weights will be loaded below if needed)
exp = Exp_Short_Term_Forecast(args)
exp.model.to(device)

# load explicit checkpoint if your Exp class doesn't auto-load it
state = torch.load("checkpoints/" + checkpoint_dir + "/checkpoint.pth", map_location=device)
exp.model.load_state_dict(state)
exp.model.eval()


# With c_out=6 (M), channel 0 corresponds to ENEL_logret
predict_cols = ["ENEL_logret", "NKT_logret", "VWS_logret"]
pred_idx = [feature_cols.index(c) for c in predict_cols]  # map to output channels
exog_cols = [c for c in feature_cols if c not in predict_cols]
exog_idx  = [feature_cols.index(c) for c in exog_cols]

def time_mark(idx: pd.DatetimeIndex):
    tm = build_time_features(idx, freq=args.freq)  # [Tm, L]
    return torch.tensor(tm.T, dtype=torch.float32) # [L, Tm]

# Train-split normalization (fixes flatness)
mu  = df_train[feature_cols].mean()
std = df_train[feature_cols].std(ddof=0).replace(0, 1.0)

def scale_frame(frame: pd.DataFrame) -> np.ndarray:
    """Return z-scored feature matrix using TRAIN stats, shape [L, C]."""
    return ((frame[feature_cols] - mu) / std).to_numpy(dtype=np.float32)

def inverse_targets(y_block: np.ndarray) -> np.ndarray:
    """Inverse-transform the three target channels back to original scale."""
    inv = y_block.copy()
    for j, col in enumerate(predict_cols):
        inv[:, j] = inv[:, j] * float(std[col]) + float(mu[col])
    return inv



# === Single model call that mimics test() without leakage ===


def predict_block_fn(ctx_df: pd.DataFrame, future_index: pd.DatetimeIndex) -> np.ndarray:
    # ----- Encoder (scale with TRAIN stats) -----
    x_enc_np   = scale_frame(ctx_df)                 # [L_enc, C] scaled
    x_enc      = torch.from_numpy(x_enc_np)
    x_mark_enc = time_mark(ctx_df.index)             # [L_enc, Tm]
    
    # ----- Decoder (future): zeros + optional scaled exog, targets remain zero -----
    x_dec_np = np.zeros((len(future_index), len(feature_cols)), dtype=np.float32)
    if FEED_FUTURE_EXOG and exog_idx:
        fut_exog = df_test.loc[future_index, exog_cols]
        x_dec_np[:, exog_idx] = ((fut_exog - mu[exog_cols]) / std[exog_cols]).to_numpy(np.float32)
    
    x_dec      = torch.from_numpy(x_dec_np)          # [pred_len, C] scaled
    x_mark_dec = time_mark(future_index)             # [pred_len, Tm]
    
    # ----- Batch + device -----
    x_enc      = x_enc.unsqueeze(0).to(device)
    x_mark_enc = x_mark_enc.unsqueeze(0).to(device)
    x_dec      = x_dec.unsqueeze(0).to(device)
    x_mark_dec = x_mark_dec.unsqueeze(0).to(device)
    
    # ----- Forward -----
    with torch.no_grad():
        y = exp.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [1, pred_len, c_out] or [1, pred_len]
    y = y.squeeze(0).detach().cpu().numpy()                  # [pred_len, c_out] or [pred_len]
    if y.ndim == 1:
        y = y[:, None]
    
    # ----- Pick our 3 channels and inverse-transform back to data space -----
    y3_scaled = y[:, pred_idx]               # [pred_len, 3] in normalized space
    y3 = inverse_targets(y3_scaled)          # back to original scale
    return y3                                 # [pred_len, 3]





# def predict_block_fn(ctx_df: pd.DataFrame, future_index: pd.DatetimeIndex) -> np.ndarray:
#     # encoder
#     x_enc = torch.tensor(ctx_df[feature_cols].values, dtype=torch.float32)
#     x_mark_enc = time_mark(ctx_df.index)
    
#     # decoder: start with zeros, then fill exog columns from df_test (known in backtest)
#     x_dec_np = np.zeros((len(future_index), len(feature_cols)), dtype=np.float32)
#     if len(exog_idx):
#         x_dec_np[:, exog_idx] = df_test.loc[future_index, exog_cols].to_numpy(np.float32)
    
#     x_dec = torch.from_numpy(x_dec_np)             # [pred_len, C]
#     x_mark_dec = time_mark(future_index)
    
#     # batchify
#     x_enc      = x_enc.unsqueeze(0).to(device)
#     x_mark_enc = x_mark_enc.unsqueeze(0).to(device)
#     x_dec      = x_dec.unsqueeze(0).to(device)
#     x_mark_dec = x_mark_dec.unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         y = exp.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [1, pred_len, c_out]
    
#     y = y.squeeze(0).detach().cpu().numpy()
#     if y.ndim == 1:  # safety if c_out==1
#         y = y[:, None]
    
#     return y[:, pred_idx]   # (pred_len, 3)


def rolling_forecast_from_presplit(df_train, df_test,feature_cols, predict_cols,seq_len, label_len,block_pred_len, stride):
    """
    Returns a DataFrame with columns:
      <col>_hat and <col>_true for each col in predict_cols,
    indexed by df_test.index.
    """
    need = seq_len + label_len
    ntest = len(df_test)
    preds = {c: [] for c in predict_cols}
    
    s = 0  # start index of current block within df_test
    while s < ntest:
        e = min(s + block_pred_len, ntest)          # end of the current block (exclusive)
        fut_idx = df_test.index[s:e]                # dates to forecast in this block
        # context is TRAIN + realized TEST up to block start (truth-fed at block boundaries)
        ctx_base = pd.concat([df_train, df_test.iloc[:s]], axis=0)
        
        if len(ctx_base) < need:
            raise ValueError(f"Need {need} rows of history, have {len(ctx_base)}")
        
        ctx_df = ctx_base.iloc[-need:]             # last seq_len (+label_len) rows
        
        # one model call for the whole block -> shape (block_len, 3)
        block_pred = predict_block_fn(ctx_df, fut_idx)
        if block_pred.ndim == 1:
            block_pred = block_pred[:, None]
        # make sure lengths match (tail block may be shorter)
        block_pred = block_pred[:len(fut_idx), :]
        
        # collect per series
        for j, col in enumerate(predict_cols):
            preds[col].extend(block_pred[:, j].tolist())
        
        # advance to next block; on next iteration, the ground truth for this block
        # will be included automatically via df_test.iloc[:s] when s increases.
        s = e
    
    # assemble output aligned to df_test.index
    out = pd.DataFrame(index=df_test.index)
    for col in predict_cols:
        out[f"{col}_hat"]  = preds[col][:ntest]
        out[f"{col}_true"] = df_test[col].values[:ntest]
    return out

# Run
out = rolling_forecast_from_presplit(
    df_train=df_train,
    df_test=df_test,
    feature_cols=feature_cols,
    predict_cols=predict_cols,   # ["ENEL_logret","NKT_logret","VWS_logret"]
    seq_len=seq_len,
    label_len=label_len,
    block_pred_len=block_pred_len,  # 20
    stride=stride,                  # 1 or 20
)

out.to_csv(f"roll20_three_stocks_{len(df_test)}.csv")
print("Saved:", f"roll20_three_stocks_{len(df_test)}.csv")
print(out.head(), "\n...\n", out.tail())



# # Metrics and Plotting

# # --- config / paths ---
# endog = ["ENEL_logret", "NKT_logret", "VWS_logret"]               # the 3 companies
# img_path = Path("plots_timesnet"); img_path.mkdir(exist_ok=True, parents=True)

# # Optional: use your run settings in filenames
# window = seq_len                      # you used seq_len as your history window
# test_size = len(out)                  # number of test points
# stride_tag = f"stride{stride}"
# model_tag  = f"tn_p{block_pred_len}"  # pred_len = 20

# # --- build A (actuals) and P (preds) from `out` ---
# A = out[[f"{c}_true" for c in endog]].rename(columns=lambda s: s.replace("_true",""))
# P = out[[f"{c}_hat"  for c in endog]].rename(columns=lambda s: s.replace("_hat",""))

# # ---- plots (one figure per series) ----
# for col in endog:
#     plt.figure(figsize=(8,3))
#     plt.plot(A.index, A[col], label="Actual")
#     plt.plot(P.index, P[col], label="Forecast (TimesNet rolling)", alpha=0.9)
#     plt.legend()
#     plt.title(col)
#     outfile = img_path / f"{col}_win{window}_{model_tag}_{stride_tag}_testsize{test_size}.png"
#     plt.savefig(outfile, dpi=300, bbox_inches="tight")
#     plt.close()

# # ---- metrics ----
def dir_num_metrics(a: pd.Series, p: pd.Series):
    a, p = a.align(p, join="inner")
    m = np.isfinite(a) & np.isfinite(p)
    if not m.any():
        return dict(n=0, DA=np.nan, BA=np.nan, P=np.nan, R=np.nan, F1=np.nan, MCC=np.nan, MAE=np.nan)
    yt = (a[m] > 0).astype(int)
    yp = (p[m] > 0).astype(int)
    P_, R_, F1_, _ = precision_recall_fscore_support(
        yt, yp, average="binary", pos_label=1, zero_division=0
    )
    return dict(
        n=int(m.sum()),
        DA=accuracy_score(yt, yp),
        BA=balanced_accuracy_score(yt, yp),
        P=P_, R=R_, F1=F1_,
        MCC=matthews_corrcoef(yt, yp),
        MAE=mean_absolute_error(a[m], p[m]),
    )

def metrics_table(A: pd.DataFrame, P: pd.DataFrame, digits=3):
    rows = [{"stock": c, **dir_num_metrics(A[c], P[c])} for c in A.columns]
    # pooled/stacked across all series:
    rows.append({"stock": "ALL", **dir_num_metrics(A.stack(), P.stack())})
    return pd.DataFrame(rows).set_index("stock").round(digits)

# mtable = metrics_table(A, P)
# print(mtable)

# # save metrics
# metrics_csv = img_path / f"metrics_{model_tag}_win{window}_{stride_tag}_testsize{test_size}.csv"
# mtable.to_csv(metrics_csv, float_format="%.6f")

# # ---- baseline (fraction of nonnegative returns) ----
# # align to the same test index; equivalent to your VARX baseline line
# baseline = (A.reindex(out.index) >= 0).mean()
# print("\nBaseline (share of >=0 moves):\n", baseline)


# -------- Block-end evaluation (window=20) --------
endog = ["ENEL_logret", "NKT_logret", "VWS_logret"]   # the 3 companies
block_len = 20
n = len(out)
n_blocks = n // block_len
O = out.iloc[: n_blocks*block_len].copy()  # trim to full blocks

end_dates = O.index[block_len-1::block_len]
A_end = pd.DataFrame(index=end_dates)
P_end = pd.DataFrame(index=end_dates)

for c in endog:
    y  = O[f"{c}_true"].to_numpy().reshape(n_blocks, block_len)
    yh = O[f"{c}_hat" ].to_numpy().reshape(n_blocks, block_len)
    A_end[c] = y[:, -1]
    P_end[c] = yh[:, -1]

# plots for (h=20)
for c in endog:
    plt.figure(figsize=(8,3))
    plt.plot(A_end.index, A_end[c], label="Actual")
    plt.plot(P_end.index, P_end[c], label="Forecast (h=20, monthly)", alpha=0.9)
    plt.title(c); plt.legend()
    plt.savefig(out_dir / f"{c}_block_h20.png", dpi=300, bbox_inches="tight")
    plt.close()

mtable_h20 = metrics_table(A_end, P_end)
print("\nBlock-anchored (h=20) metrics:\n", mtable_h20)
mtable_h20.to_csv(out_dir / "metrics_block_h20.csv", float_format="%.6f")

# -------- Per-horizon metrics (h=1..20) on disjoint blocks --------
from sklearn.metrics import precision_recall_fscore_support  # (already imported above, but safe)

def per_horizon_metrics(out_block: pd.DataFrame, cols, block_len=20, digits=3):
    n = len(out_block)
    n_blocks = n // block_len
    idx = out_block.index[: n_blocks*block_len]
    rows = []
    for c in cols:
        y  = out_block.loc[idx, f"{c}_true"].to_numpy().reshape(n_blocks, block_len)
        yh = out_block.loc[idx, f"{c}_hat" ].to_numpy().reshape(n_blocks, block_len)
        for h in range(block_len):  # 0..19 -> horizon = h+1
            a = pd.Series(y[:, h]); p = pd.Series(yh[:, h])
            m = np.isfinite(a) & np.isfinite(p)
            if not m.any(): continue
            yt, yp = (a[m] > 0).astype(int), (p[m] > 0).astype(int)
            P_, R_, F1_, _ = precision_recall_fscore_support(yt, yp, average="binary", pos_label=1, zero_division=0)
            rows.append({
                "horizon": h+1, "stock": c, "n": int(m.sum()),
                "DA": accuracy_score(yt, yp),
                "BA": balanced_accuracy_score(yt, yp),
                "P": P_, "R": R_, "F1": F1_,
                "MCC": matthews_corrcoef(yt, yp),
                "MAE": mean_absolute_error(a[m], p[m]),
            })
    return pd.DataFrame(rows).round(digits)

hm = per_horizon_metrics(O, endog, block_len=block_len)
print("\nPer-horizon metrics (block-anchored):\n", hm.head())
hm.to_csv(out_dir / "metrics_per_horizon.csv", index=False, float_format="%.6f")

# optional: MAE vs horizon plots
for c in endog:
    sub = hm[hm["stock"]==c]
    plt.figure(figsize=(7,3))
    plt.plot(sub["horizon"], sub["MAE"], marker="o")
    plt.title(f"{c}: MAE by horizon (block-anchored)")
    plt.xlabel("Horizon (days ahead)"); plt.ylabel("MAE"); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{c}_MAE_by_horizon.png", dpi=300, bbox_inches="tight")
    plt.close()

# -------------- high vs low volatility regime and t test -----------

A = A_end
P = P_end

rows_reg = []
df_data = pd.read_csv(csv_path_regime, delimiter=";")
df_data["Date"] = pd.to_datetime(df_data["Date"])
df_data = df_data.sort_values("Date").set_index("Date")  # index by date
reg = df_data.loc[A.index, 'regime']  # 'high'/'low'
for s in A.columns:
    corr = ((A[s] > 0).astype(int) == (P[s] > 0).astype(int)).astype(float)
    hi, lo = corr[reg.eq('high')].dropna(), corr[reg.eq('low')].dropna()
    t,p,_ = ttest_ind(hi, lo, usevar='unequal')
    #print(f"{s}: DA_high={hi.mean():.3f} (n={len(hi)}), DA_low={lo.mean():.3f} (n={len(lo)}), Δ={hi.mean()-lo.mean():.3f}, t={t:.2f}, p={p:.4f}")
    rows_reg.append({
        'stock': s,
        'DA_high': hi.mean(),
        'n_high': int(hi.shape[0]),
        'DA_low': lo.mean(),
        'n_low': int(lo.shape[0]),
        'delta': hi.mean() - lo.mean(),
        't_stat': float(t),
        'p_value': float(p),
    })

df_reg = pd.DataFrame(rows_reg).set_index('stock').sort_index()
df_reg.to_csv(out_dir + f"plots_timesnet_block/volregime_win{window}_testsize{test_size}.csv", float_format='%.6f')
print(df_reg)
