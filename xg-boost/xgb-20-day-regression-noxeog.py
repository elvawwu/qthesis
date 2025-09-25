from xgboost import XGBRegressor, XGBClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, matthews_corrcoef,
    mean_absolute_error
)
import matplotlib.dates as mdates



isbinary = False
use_exog = True
window = 20
p = 3
test_size=0.5
# --- paths ---
csv_path = Path("~/Dropbox/elvas-thesis/xg-boost/data/dataset_stationary_with_us10y_final.csv").expanduser()
csv_path_regime = Path("~/Dropbox/elvas-thesis/xg-boost/data/dataset_stationary_with_us10y_final.csv").expanduser()

df_data = pd.read_csv(csv_path, parse_dates=["Date"], delimiter=";").sort_values("Date").set_index("Date")

endog_cols = ["ENEL_logret","NKT_logret","VWS_logret"] # first, access log returns, independently of classification or regression
endog_cols_binary_names = ["ENEL_bin","NKT_bin","VWS_bin"] # only for presentation of results

exog_cols  = ["utilities_close_logret"] #  , "us10y", "brent_price_logret", "utilities_close_logret", "brent_price_logret"

# smoothening: moving average over rolling window. 
df = df_data[endog_cols+exog_cols].rolling(window=window, min_periods=window).sum()
df[exog_cols]  = df[exog_cols].shift(window)   # uses info known by t-window (avoid look-ahead)
df = df.iloc[2*window:]  # drop first row to align with shifted exog # this also drops the nas

if isbinary:
    # for classification, replace accumulated log returns by their sign. 
    # we read 1 for "positive" and 0 for "nonpositive"
    df[endog_cols] = (np.sign(df[endog_cols])>0).astype(int) 
    df[exog_cols] = (np.sign(df[exog_cols])>0).astype(int) 
    outdir = Path(f"plots_xgb_block_bin/win{window}_p{p}_testsize{test_size}_exog{use_exog}"); outdir.mkdir(exist_ok=True, parents=True)
else:
    endog_cols = ["ENEL_logret","NKT_logret","VWS_logret"]
    outdir = Path(f"plots_xgb_block/win{window}_p{p}_testsize{test_size}_onlybrentoil"); outdir.mkdir(exist_ok=True, parents=True)

# === features: lags of endog + exog (no future info) ===
lagY = [df[endog_cols].shift(l).add_prefix(f"L{l}_") for l in range(1, p+1)]
lagX = [df[exog_cols].shift(l).add_prefix(f"X{l}_")  for l in range(1, p+1)]
if use_exog:
    X_all = pd.concat(lagY + lagX, axis=1)
else:
    X_all = pd.concat(lagY, axis=1)

Y_all = df[endog_cols].copy()

XY = pd.concat([Y_all, X_all], axis=1).dropna()
Y = XY[endog_cols]
X = XY.drop(columns=endog_cols)

# === chronological split (on the aligned frame) ===
mid = int(np.floor((1-test_size)*len(X)))
X_tr, X_te = X.iloc[:mid], X.iloc[mid:]
Y_tr, Y_te = Y.iloc[:mid], Y.iloc[mid:]

# === train once ===
if isbinary:
    reg = MultiOutputClassifier(XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1
    ))
else:
    reg = MultiOutputRegressor(XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    ))

reg.fit(X_tr, Y_tr)

# === block-anchored forecast (no refit inside blocks) ===
# block_len = 20
# idx_all = X.index
# pos = {d:i for i,d in enumerate(idx_all)}
# test_idx = X_te.index

# n-day ahead rolling forecast (no refits; only state updates)
preds = []
idx = []
row = X_te.loc[X_te.index[0]].copy()
pos = {d:i for i,d in enumerate(X_te.index)}
for i in np.arange(len(X_te.index)):
    if (window !=1) and (i%window !=0):
        continue
    
    t_start = X_te.index[i]
    if window > len(X_te.index)-i:
        break
    
    print(i)
    t_end = X_te.index[i+window-1]
    x_t = Y_te.loc[t_start:t_end]
    pm = reg.predict(X_te.loc[t_start:t_end])
    
    # append prediction to endog for next window to enable next day prediction
    block = X_te.index[i:i+window]
    yhat_map = {}
    rows = []
    for t in block:
        r = X.loc[t].copy(); tp = pos[t]
        for c in endog_cols:
            for l in range(1, p+1):
                col = f"L{l}_{c}"
                if col not in r: continue
                lp = tp - l
                if lp < 0: break
                ld = X_te.index[lp]
                if ld in yhat_map: r[col] = yhat_map[ld][c]
        rows.append(r)
        # one-step predict for this day and store for subsequent lags in this block
        yv = reg.predict(r.to_frame().T)[0]                 # shape (3,)
        yhat_map[t] = dict(zip(endog_cols, yv))
    
    # vectorized predict for the whole block (matches your pm[...] style)
    R  = pd.DataFrame(rows, index=block)
    pm = reg.predict(R)                                     # shape (window, 3)
    preds.append(pm[window-1]) # next prediction
    idx.append(t_end)


P = pd.DataFrame(preds, index=idx, columns=endog_cols)
A = Y_te.loc[P.index, endog_cols]


# convert back to 1/-1 and rename columns 
if isbinary:
    P_trend = 2*P-1
    A_trend = 2*A-1
    binary_dict = dict(zip(endog_cols, endog_cols_binary_names))
    P = P_trend.rename(columns=binary_dict)
    A = A_trend.rename(columns=binary_dict)


# === metrics & a quick plot per stock (end-of-block) ===
def dir_num_metrics(a, p):
    a, p = a.align(p, join="inner")
    m = np.isfinite(a) & np.isfinite(p)
    if not m.any(): return dict(n=0, DA=np.nan, BA=np.nan, P=np.nan, R=np.nan, F1=np.nan, MCC=np.nan, MAE=np.nan)
    yt, yp = (a[m] > 0).astype(int), (p[m] > 0).astype(int)
    P_, R_, F1_, _ = precision_recall_fscore_support(yt, yp, average="binary", pos_label=1, zero_division=0)
    return dict(n=m.sum(), DA=accuracy_score(yt, yp), BA=balanced_accuracy_score(yt, yp),
                P=P_, R=R_, F1=F1_, MCC=matthews_corrcoef(yt, yp), MAE=mean_absolute_error(a[m], p[m]))

def metrics_table(A, P, digits=3):
    rows = [{ "stock": c, **dir_num_metrics(A[c], P[c]) } for c in A.columns]
    rows.append({ "stock": "ALL", **dir_num_metrics(A.stack(), P.stack()) })
    return pd.DataFrame(rows).set_index("stock").round(digits)

mt_h20 = metrics_table(A, P)


# plots
def plot_bin_scatter(idx, a01, p01, title, outfile):
    fig, ax = plt.subplots(figsize=(9,3))
    ax.scatter(idx, a01, s=10, label="Actual", alpha=0.8)
    ax.scatter(idx, np.clip(p01 + 0.06, 0, 1.12), s=10, marker="x", label="Pred", alpha=0.8)
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0,1]); ax.set_yticklabels(["0","1"])
    ax.set_title(title); ax.set_ylabel("Class")
    
    loc = mdates.AutoDateLocator(); ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight"); plt.close()

if isbinary:
    for c in endog_cols_binary_names:
        plot_bin_scatter(idx=A.index,a01=A[c].astype(int),p01=P[c].astype(int),title=c,outfile=outdir / f"{c}_xgb_bin_win20_testsize{test_size}.png")
else:
    for c in endog_cols:
        plt.figure(figsize=(8,3))
        plt.plot(A.index, A[c], label="Actual")
        plt.plot(P.index, P[c], label="XGB (rolling 1-step)")
        plt.legend(); plt.title(c)
        plt.savefig(outdir / f"{c}_xgb_win20_testsize{test_size}.png", dpi=300, bbox_inches="tight"); plt.close()





# (optional) save CSVs
mt_h20.to_csv(outdir / f"metrics_xgb_p{p}_win20_testsize{test_size}_p{p}.csv", float_format="%.6f")
A.to_csv(outdir / f"A_{window}d_p{p}_testsize{test_size}.csv"); P.to_csv(outdir / f"P_{window}d_p{p}_testsize{test_size}.csv")

# ========= Volatility-regime split for 20-day (end-of-block) evaluation =========
from statsmodels.stats.weightstats import ttest_ind
from pathlib import Path

def regime_table(A: pd.DataFrame, P: pd.DataFrame, csv_path_regime: Path, out_csv_path: Path):
    """
    Welch t-test comparing direction accuracy in 'high' vs 'low' volatility
    at the evaluation dates in A.index (end-of-block dates).
    A: truth (DataFrame), P: preds (DataFrame) with same index/columns.
    """
    # try to load explicit regime file
    reg = None
    try:
        df_reg = pd.read_csv(csv_path_regime, delimiter=";")
        df_reg["Date"] = pd.to_datetime(df_reg["Date"])
        df_reg = df_reg.sort_values("Date").set_index("Date")
        reg_raw = df_reg["regime"].astype(str).str.lower()
        reg = reg_raw.replace({"1":"high","0":"low","hi":"high","lo":"low"}).reindex(A.index)
        print(f"Loaded regime from {csv_path_regime}")
    except Exception as e:
        print(f"[Info] Regime file not available/usable ({e}). Building proxy from data…")
        # simple fallback: median split of 20d rolling cross-sectional |returns|
        vol = df[endog_cols].abs().mean(axis=1).rolling(20, min_periods=10).mean()
        v_ = vol.reindex(A.index)
        thr = v_.median()
        reg = pd.Series(np.where(v_ >= thr, "high", "low"), index=A.index)
        tmp = out_csv_path.with_name("vol_regime_built_from_data.csv")
        pd.DataFrame({"Date": A.index, "regime": reg.values}).to_csv(tmp, index=False, date_format="%Y-%m-%d")
        print(f"[Info] Wrote derived regime to {tmp}")
    
    rows = []
    for s in A.columns:
        # direction accuracy at end-of-block dates
        corr = ((A[s] > 0).astype(int) == (P[s] > 0).astype(int)).astype(float)
        hi = corr[reg.eq("high")].dropna()
        lo = corr[reg.eq("low")].dropna()
        if len(hi)==0 or len(lo)==0:
            t = p = np.nan
        else:
            t, p, _ = ttest_ind(hi, lo, usevar="unequal")
        rows.append({
            "stock": s,
            "DA_high": hi.mean() if len(hi) else np.nan,
            "n_high": int(len(hi)),
            "DA_low":  lo.mean() if len(lo) else np.nan,
            "n_low":  int(len(lo)),
            "delta": (hi.mean() - lo.mean()) if (len(hi) and len(lo)) else np.nan,
            "t_stat": float(t) if np.isfinite(t) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })
    
    out = pd.DataFrame(rows).set_index("stock").sort_index()
    out.to_csv(out_csv_path, float_format="%.6f")
    print("\nVolatility-regime split (DA):\n", out)
    return out

print("\nXGB (regression) — block-anchored h=20, 50/50 split:\n", mt_h20)
regime_table(A, P, csv_path_regime, outdir / f"volregime_xgb_20d_p{p}_testsize{test_size}.csv" )
