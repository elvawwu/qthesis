import pandas as pd, numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, mean_absolute_error
from statsmodels.stats.weightstats import ttest_ind
import matplotlib.dates as mdates


isbinary = True
window = 20
p = 1
test_size=0.2
# --- paths ---
csv_path = Path("~/Dropbox/elvas-thesis/xg-boost/data/dataset_stationary_with_us10y_final.csv").expanduser()
img_path = Path("~/Dropbox/elvas-thesis/xg-boost/img").expanduser(); img_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path, parse_dates=["Date"], delimiter=";").sort_values("Date").set_index("Date")
if isbinary:
    endog_cols = ["ENEL_bin","NKT_bin","VWS_bin"]
else:
    endog_cols = ["ENEL_logret","NKT_logret","VWS_logret"]

    
exog_cols  = ["brent_price_logret","utilities_close_logret","us10y"]
endog, exog = df[endog_cols].astype(float), df[exog_cols].astype(float)

# ---- features: lags of endog and exog (use ONLY past info) ----
X_parts = [endog.shift(l).add_prefix(f"L{l}_") for l in range(1, p+1)]
X_parts += [exog.shift(l).add_prefix(f"X{l}_") for l in range(1, p+1)]
X = pd.concat(X_parts, axis=1)
Y = endog.copy()
XY = pd.concat([X, Y], axis=1).dropna()
X, Y = XY.drop(columns=endog_cols), XY[endog_cols]

# ---- 80/20 split (chronological) ----
n = len(Y); split = int(0.8*n)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

# ---- train once (fixed parameters) ----
if isbinary:
    model = MultiOutputClassifier(XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1
    ))
else:
    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    ))

model.fit(X_tr, Y_tr)

# ---- 1-step rolling predictions (no refits; uses actual lags already in X_te) ----
Y_pred = pd.DataFrame(model.predict(X_te), index=Y_te.index, columns=Y_te.columns)

A = Y_te.copy()
P = Y_pred.copy()
if isbinary: # convert back to 1 / -1 labels
    P_trend = 2*P-1
    A_trend = 2*A-1

# ---- plots ----
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
    for c in ["ENEL_bin","NKT_bin","VWS_bin"]:
        plot_bin_scatter(idx=Y_te.index,a01=A,p01=P,title=c,outfile=img_path / f"{c}_xgb_bin_win1_testsize0.2_p{p}.png")
else:
    for c in endog_cols:
        plt.figure(figsize=(8,3))
        plt.plot(Y_te.index, Y_te[c], label="Actual")
        plt.plot(Y_pred.index, Y_pred[c], label="XGB (rolling 1-step)")
        plt.legend(); plt.title(c)
        plt.savefig(img_path / f"{c}_xgb_win1_testsize0.2_p{p}.png", dpi=300, bbox_inches="tight"); plt.close()



# ---- metrics (same style you used) ----
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

tbl = metrics_table(A, P)
tbl.to_csv(img_path / f"metrics_xgb_p{p}_win1_testsize0.2_p{p}.csv", float_format="%.6f")
print(tbl)


# ========= Volatility-regime split (Welch t-test on direction accuracy) =========
# Set this to your regime file (must contain a 'Date' column and a 'regime' column)
csv_path_regime = Path("~/Dropbox/elvas-thesis/xg-boost/data/dataset_stationary_with_us10y_final.csv").expanduser()

def regime_table(A: pd.DataFrame, P: pd.DataFrame, csv_path_regime: Path, out_csv_path: Path):
    """
    A, P: DataFrames aligned on target dates (same index), columns = stocks.
          A are truths, P are predictions for those dates.
    csv_path_regime: CSV with columns ['Date', 'regime'] where regime in {'high','low'} or {1,0}.
    """
    # load and align regime
    df_reg = pd.read_csv(csv_path_regime, delimiter=";")
    if "Date" not in df_reg.columns:
        raise ValueError("Regime CSV must have a 'Date' column.")
    if "regime" not in df_reg.columns:
        raise ValueError("Regime CSV must have a 'regime' column.")
    df_reg["Date"] = pd.to_datetime(df_reg["Date"])
    df_reg = df_reg.sort_values("Date").set_index("Date")
    
    # normalize labels to 'high'/'low'
    reg_raw = df_reg["regime"].astype(str).str.lower()
    reg_norm = reg_raw.replace({"1":"high","0":"low","hi":"high","lo":"low"})
    reg = reg_norm.reindex(A.index)   # align to evaluation dates
    
    rows = []
    for s in A.columns:
        # 1 if sign correct, 0 otherwise
        corr = ((A[s] > 0).astype(int) == (P[s] > 0).astype(int)).astype(float)
        hi = corr[reg.eq("high")].dropna()
        lo = corr[reg.eq("low")].dropna()
        if len(hi) == 0 or len(lo) == 0:
            t, p = np.nan, np.nan
        else:
            t, p, _ = ttest_ind(hi, lo, usevar='unequal')  # Welch
        rows.append({
            "stock": s,
            "DA_high": hi.mean() if len(hi) else np.nan,
            "n_high": int(len(hi)),
            "DA_low": lo.mean() if len(lo) else np.nan,
            "n_low": int(len(lo)),
            "delta": (hi.mean() - lo.mean()) if (len(hi) and len(lo)) else np.nan,
            "t_stat": float(t) if np.isfinite(t) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })
    
    out = pd.DataFrame(rows).set_index("stock").sort_index()
    out.to_csv(out_csv_path, float_format="%.6f")
    print("\nVolatility-regime split (DA):\n", out)
    return out

# --- call for NEXT-DAY evaluation (this file) ---
reg_out_csv = img_path / f"volregime_xgb_nextday_p{p}_testsize0.2_bin.csv"
_ = regime_table(A, P, csv_path_regime, reg_out_csv)
