import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from pathlib import Path
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, mean_absolute_error, mean_squared_error
from statsmodels.stats.weightstats import ttest_ind

# load data
csv_path = Path("~/Dropbox/elvas-thesis/varx/data/dataset_stationary_with_us10y_final.csv").expanduser()
img_path = Path("~/Dropbox/elvas-thesis/varx/img").expanduser()
img_path.mkdir(parents=True, exist_ok=True)  

# endog and exog vars
endog = ["ENEL_logret", "NKT_logret", "VWS_logret"]
exog  = ["brent_price_logret", "utilities_close_logret", "us10y"] 

# model parameter
#window=20
#test_size=0.5
window=1
test_size=0.2
order = (1, 7) # 20-day forecast were trained with (3, 0) # next-day forecasts were trained with (1, 0)

df_data = pd.read_csv(csv_path, delimiter=";")
df_data["Date"] = pd.to_datetime(df_data["Date"])
df_data = df_data.sort_values("Date").set_index("Date")  # index by date

# smoothening: moving average over rolling window. 
df = df_data[endog+exog].rolling(window=window, min_periods=window).sum()


# split endog and exog
df_endog = df[endog]
df_exog  = df[exog]
df_exog  = df_exog.shift(window)   # uses info known by t-window (avoid look-ahead)
df_endog = df_endog.iloc[2*window:]  # drop first row to align with shifted exog # this also drops the nas
df_exog  = df_exog.iloc[2*window:]

# train-test split

endog_train, endog_test, exog_train, exog_test = train_test_split(df_endog, df_exog, test_size=test_size, shuffle=False)

model = VARMAX(endog=endog_train, exog=exog_train, order=order)
result = model.fit(disp=False)
print(result.summary())

# predictions (fixed-origin multi-step; uses future exog you provided)
# steps = len(endog_test)
# fc_crystal_ball = result.get_forecast(steps=steps, exog=exog_test.iloc[:steps])

# n-day ahead rolling forecast (no refits; only state updates)
preds = []
res_rolling = result
for i in np.arange(len(endog_test.index)):
    if (window !=1) and (i%window !=0):
        continue
    
    t_start = endog_test.index[i]
    if window > len(endog_test.index)-i:
        break
    
    print(i)
    t_end = endog_test.index[i+window-1]
    x_t = exog_test.loc[t_start:t_end]
    pm  = res_rolling.get_forecast(steps=window, exog=x_t).predicted_mean  # <-- label-free
    yhat_t = pm.iloc[window-1]
    yhat_t.name = t_end                             # <-- stamp with the date
    preds.append(yhat_t)
    # append values to avoid index-extension checks
    res_rolling = res_rolling.append(endog_test.loc[t_start:t_end].values, exog=x_t.values, refit=False)


fc_rolling = pd.DataFrame(preds, index=[p.name for p in preds], columns=endog_test.columns)

# ---- Plots and Metrics ---- #

# plots
A = endog_test.reindex([p.name for p in preds])
P = fc_rolling

for col in endog:
    plt.figure(figsize=(8,3))
    plt.plot(A.index, A[col], label="Actual")
    plt.plot(fc_rolling.index, fc_rolling[col], label="Forecast (rolling 1-step)", alpha=0.9)
    plt.legend(); plt.title(col)
    outfile = img_path / f"{col}_win{window}_testsize{test_size}_p{order[0]}q{order[1]}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# metrics
def dir_num_metrics(a, p):
    a, p = a.align(p, join="inner")
    m = np.isfinite(a) & np.isfinite(p)
    if not m.any():
        return dict(n=0, DA=np.nan, BA=np.nan, P=np.nan, R=np.nan, F1=np.nan, MCC=np.nan, MAE=np.nan,RMSE=np.nan)
    yt, yp = (a[m] > 0).astype(int), (p[m] > 0).astype(int)
    P_, R_, F1_, _ = precision_recall_fscore_support(yt, yp, average="binary", pos_label=1, zero_division=0)
    return dict(
        n=m.sum(),
        DA=accuracy_score(yt, yp),
        BA=balanced_accuracy_score(yt, yp),
        P=P_, R=R_, F1=F1_,
        MCC=matthews_corrcoef(yt, yp),
        MAE=mean_absolute_error(a[m], p[m]),
        #RMSE=mean_squared_error(a[m], p[m])
    )

def metrics_table(A, P, digits=3):
    rows = [{ "stock": c, **dir_num_metrics(A[c], P[c]) } for c in A.columns]
    rows.append({ "stock": "ALL", **dir_num_metrics(A.stack(), P.stack()) })
    df = pd.DataFrame(rows).set_index("stock")
    # enforce column order so RMSE is visible
    order = ["n", "MAE", "RMSE", "DA", "BA", "P", "R", "F1", "MCC"]
    df = df.reindex(columns=[c for c in order if c in df.columns])
    return pd.DataFrame(rows).set_index("stock").round(digits)


mtable = metrics_table(A, P)
print(mtable)
mtable.to_csv(img_path / f"metrics_win{window}_testsize{test_size}_p{order[0]}q{order[1]}.csv", float_format="%.6f")

# baseline
(df[endog].reindex([p.name for p in preds])>0).mean()



# -------------- high vs low volatility regime and t test -----------

A = endog_test.reindex([p.name for p in preds])
P = fc_rolling.reindex(A.index)
df_data.reindex(A.index)['regime']

# daily DA = share of correctly predicted directions across stocks
# da_daily = ((A>0).astype(int) == (P>0).astype(int)).mean(axis=1)

rows_reg = []
reg = df_data.loc[A.index, 'regime']  # 'high'/'low'
for s in endog_test.columns:
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
df_reg.to_csv(img_path / f"volregime_win{window}_testsize{test_size}_p{order[0]}q{order[1]}.csv", float_format='%.6f')
print(df_reg)