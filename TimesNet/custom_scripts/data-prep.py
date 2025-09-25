import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from pathlib import Path
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, mean_absolute_error
from statsmodels.stats.weightstats import ttest_ind

# load data
window=20
test_size=0.5

csv_path = Path("~/Dropbox/elvas-thesis/varx/data/dataset_stationary_with_us10y_final.csv").expanduser()
csv_outpath = f"./timesnet_energy_var6_window{window}_testsize{test_size}"

df_data = pd.read_csv(csv_path, delimiter=";")
df_data["Date"] = pd.to_datetime(df_data["Date"])
df_data = df_data.sort_values("Date").set_index("Date")  # index by date

# endog and exog vars
endog = ["ENEL_logret", "NKT_logret", "VWS_logret"]
exog  = ["brent_price_logret", "utilities_close_logret", "us10y"] 


# smoothening: moving average over rolling window. 
df = df_data[endog+exog].rolling(window=window, min_periods=window).sum()
df[exog] = df[exog].shift(window) 
df = df.iloc[2*window:]
df.to_csv(Path(csv_outpath + ".csv").expanduser(), float_format="%.6f")

df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)
df_train.to_csv(Path(csv_outpath + "_train.csv").expanduser(), float_format="%.6f")
df_test.to_csv(Path(csv_outpath + "_test.csv").expanduser(), float_format="%.6f")
