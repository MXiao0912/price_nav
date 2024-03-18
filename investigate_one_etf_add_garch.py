import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace import tools
import warnings
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import ADFTest
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf
from workalendar.europe import UnitedKingdom
from scipy.linalg import block_diag
from datetime import date
import random
import re
idx = pd.IndexSlice
cal = UnitedKingdom()

px_nav = pd.read_csv("px_nav.csv")
px_nav = px_nav.sort_values("date")
px_nav = px_nav[["isin","date","price","nav"]]
px_nav["price"] = np.log(px_nav["price"])
px_nav["nav"] = np.log(px_nav["nav"])

isin_list = pd.unique(px_nav["isin"])
i = np.where(isin_list=="IE00BG13YK79")[0][0]

df = px_nav.loc[px_nav["isin"]==isin_list[i],["price","nav"]]
df.index = px_nav.loc[px_nav["isin"]==isin_list[i], "date"]
df.index = pd.to_datetime(df.index)
df_complete = pd.DataFrame(index=[d for d in pd.bdate_range(start=max(date(2018,1,3),min(df.index.date)), 
                                            end=min(max(df.index.date), date(2022,10,22))) if cal.is_working_day(d)])
df_complete = df_complete.join(df)
df_complete = df_complete.fillna(method="ffill")
df_complete.dropna(inplace=True)

# starting param
df_start = df_complete.diff().dropna().copy()
q1_p = df_start["price"].quantile(0.25)
q3_p = df_start["price"].quantile(0.75)
iqr_p = q3_p-q1_p
low_p = q1_p-1.5*iqr_p
up_p = q3_p+1.5*iqr_p

q1_n = df_start["nav"].quantile(0.25)
q3_n = df_start["nav"].quantile(0.75)
iqr_n = q3_n-q1_n
low_n = q1_n-1.5*iqr_n
up_n = q3_n+1.5*iqr_n

df_start.loc[:,"price"] = df_start["price"].where((df_start["price"] > low_p) & (df_start["price"] < up_p), np.nan)
df_start.loc[:,"nav"] = df_start["nav"].where((df_start["nav"] > low_n) & (df_start["nav"] < up_n), np.nan)
df_start['price'] = df_start['price'].interpolate(method='linear')
df_start['nav'] = df_start['nav'].interpolate(method='linear')
df_start.dropna(inplace=True)

price_auto = auto_arima(df_start["price"], start_p=1, start_q=1, max_p=10, max_q=10)
price_auto = auto_arima(df_start.loc["2020-04-01":,"price"], start_p=1, start_q=1, max_p=10, max_q=10)

nav_auto = auto_arima(df_start["nav"], start_p=1, start_q=1, max_p=10, max_q=10)




lag = 2
psi_res = ARIMA(df_start["price"], order=(lag,0,lag)).fit()


for i in range(len(isin_list)):
    df = px_nav.loc[px_nav["isin"]==isin_list[i],["price","nav"]]
    df.index = px_nav.loc[px_nav["isin"]==isin_list[i], "date"]
    df.index = pd.to_datetime(df.index)
    df_complete = pd.DataFrame(index=[d for d in pd.bdate_range(start=max(date(2018,1,3),min(df.index.date)), 
                                                end=min(max(df.index.date), date(2022,10,22))) if cal.is_working_day(d)])
    df_complete = df_complete.join(df)
    df_complete = df_complete.fillna(method="ffill")
    df_complete.dropna(inplace=True)

    df_start = df_complete.diff().dropna().copy()
    # df_start = df_start.loc[:"2020-02-01",:]
    if df_start.shape[0]!=0:
        price_auto = auto_arima(df_start["price"], start_p=1, start_q=1, max_p=10, max_q=10)
        nav_auto = auto_arima(df_start["nav"], start_p=1, start_q=1, max_p=10, max_q=10)

        print(isin_list[i])
        print(price_auto)
        print(nav_auto)
