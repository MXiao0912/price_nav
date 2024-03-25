import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
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
import os
idx = pd.IndexSlice
cal = UnitedKingdom()
os.chdir("/home/azureuser/price_nav")

px_nav = pd.read_csv("px_nav.csv")
px_nav = px_nav.sort_values("date")
px_nav = px_nav[["isin","date","price","nav"]]
px_nav["price"] = np.log(px_nav["price"])
px_nav["nav"] = np.log(px_nav["nav"])
# px_nav = na.omit(px_nav)


class SSM_anylag(sm.tsa.statespace.MLEModel):
    

    def __init__(self, endog, params0, lag):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=lag, trim='both', original='in')
        self.ind_all = endog.index[lag:]
        endog = X[:, :2]
        self.lagged_endog = X[:, 2:]
        self._start_params = params0
        self.lag = lag
        self._param_names = [f'psi_{i}' for i in range(1,lag+1)] + [f'phi_{i}' for i in range(1,lag+1)] + ['sigma2_r', 'sigma_e', 'sigma_w',
                    "obs_err_cov_lower"]
        
        # Initialize the state space model
        # Note: because your state process is nonstationary,
        # you can't use stationary initialization
        super().__init__(endog, k_states=lag+1, k_posdef=1,
                         initialization='diffuse')

        # Setup the fixed components of the state space representation
        design = np.vstack((np.ones((1,lag+1)),np.zeros((1, lag+1))))
        design[1,0] = 1
        self['design'] = design
        transition = np.zeros((1, lag+1))
        transition[0,0] = 1
        transition = np.vstack((transition, np.hstack((np.eye(lag),np.zeros((lag,1))))))
        self['transition'] = transition
        selection = np.zeros((lag+1,1))
        selection[0,0] = 1
        self['selection'] = selection

    # For the parameter estimation part, it is
    # helpful to use parameter transformations to
    # keep the parameters in the valid domain. Here
    # I assumed that you wanted phi and psi to be
    # between (-1, 1).
        
    def transform_params(self, params):
        params = params.copy()
        for i in [0, lag]:
            params[i:i + lag] = tools.constrain_stationary_univariate(params[i:i + lag])
        params[(2*lag)] = params[(2*lag)]**2
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        for i in [0, lag]:
            params[i:i + lag] = tools.unconstrain_stationary_univariate(params[i:i + lag])
        params[(2*lag)] = params[(2*lag)]**0.5
        return params

    # Describe how parameters enter the model
    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        # Here is where we're putting the lagged endog, multiplied
        # by phi and psi into the intercept term
        # self['obs_intercept'] = self.lagged_endog.T * params[:2, None]
        intercept_p = []
        intercept_n = []
        for param in params[:lag]:
            intercept_p.extend([param, 0])
        for param in params[lag:2*lag]:
            intercept_n.extend([0, param])

        self['obs_intercept'] = np.array([intercept_p, intercept_n]) @ self.lagged_endog.T
        self['state_intercept'] = np.tile(np.vstack((0, np.zeros((lag,1)))), self.lagged_endog.shape[0])
        # self['state_intercept'] = self['state_intercept'] + np.vstack((np.hstack(([[params[2*lag+7]]], np.zeros((1,lag)))), np.zeros((lag,lag+1)))) @ np.tile((self.ind_all.month == 3) & (self.ind_all.year == 2020), lag+1).reshape(lag+1, -1).astype(int)
        for i in range(1, lag+1):
            self['design', 0, i] = -params[i-1]
        params_acc = 0
        for i in range(lag, 2*lag):
            params_acc += params[i]
        self['design', 1, 0]  = 1 - params_acc
        self['state_cov', 0, 0] = params[2*lag]
        lower = np.array([[params[2*lag+1], 0], [params[2*lag+3], params[2*lag+2]]])
        self['obs_cov'] = lower@(lower.T)



def fit_param(df, params0, lag, method="lbfgs",maxit=200):
  mod = SSM_anylag(df,params0,lag)
  res = mod.fit(disp=False,method=method, maxiter=maxit)
  result_param = res.params
  result_state = res.smoothed_state[0,:]
  result_state_var = res.smoothed_state_cov[0,0]
  res.plot_diagnostics(variable=0, acf_kwargs = {'zero':False})
  plt.savefig("./diagnostics/" + "price" + str(isin_list[i]) + ".jpg")
  plt.close()
  res.plot_diagnostics(variable=1, acf_kwargs = {'zero':False})
  plt.savefig("./diagnostics/" + "nav" + str(isin_list[i]) + ".jpg")
  plt.close()
  return list([result_param, result_state, result_state_var])

isin_list = pd.unique(px_nav["isin"])
res_list = [None]*len(isin_list)
state_list = [None]*len(isin_list)
isin_list = np.unique(px_nav[["isin"]])
px_nav = pd.DataFrame(px_nav)
px_nav["date"] = px_nav["date"].astype(str)


for i in range(len(isin_list)):
# for i in range(5):
    print(str(i)+isin_list[i])
    # i = 97
    df = px_nav.loc[px_nav["isin"]==isin_list[i],["price","nav"]]
    df.index = px_nav.loc[px_nav["isin"]==isin_list[i], "date"]
    df.index = pd.to_datetime(df.index)
    df_complete = pd.DataFrame(index=[d for d in pd.bdate_range(start=max(date(2018,1,3),min(df.index.date)), 
                                                end=min(max(df.index.date), date(2022,10,22))) if cal.is_working_day(d)])
    df_complete = df_complete.join(df)
    df_complete = df_complete.fillna(method="ffill")
    df_complete.dropna(inplace=True)

    df_start = df_complete.diff().copy()
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

    # price_auto = auto_arima(df_start["price"], start_p=1, start_q=1, max_p=10, d = 0, max_q=10)
    # nav_auto = auto_arima(df_start["nav"], start_p=1, d = 0,
    #                     start_q=1, max_p=10, max_q=10)
    # lag = max(len(price_auto.arparams()), len(price_auto.maparams()), len(nav_auto.arparams()), len(nav_auto.maparams()), 1)
    # print(lag)
    lag = 1

    warnings.filterwarnings("ignore")
    psi_res = ARIMA(df_start["price"], order=(lag,0,lag)).fit()
    phi_res = ARIMA(df_start["nav"], order=(lag,0,1)).fit()
    warnings.filterwarnings("default")

    # result_param, result_state, result_state_var = fit_param(df_complete, psi_res.arparams.tolist() + phi_res.arparams.tolist() + [0.1, 1, 1, 0, 0, 0, 0], lag=lag)

    print("fit1 start")
    use_alternative_model = False
    # Catch ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.filterwarnings("always")
        try:
            result_param, result_state, result_state_var = fit_param(df_complete, psi_res.arparams.tolist() + phi_res.arparams.tolist() + [0.1, 1, 1, 0, 0, 0, 0], lag=lag)
            if caught_warnings:  # If any warnings were caught
                use_alternative_model = True
        except:
            use_alternative_model = True
    # if use_alternative_model:
    print("fit2 start")
    result_param, result_state, result_state_var = fit_param(df_complete, result_param, method="nm", maxit=3000, lag=lag)

    res_list[i] = np.append(result_param, isin_list[i])
    result_state_df = df_complete.iloc[:-lag,].copy()
    result_state_df["state"] = result_state
    result_state_df["isin"] = isin_list[i]
    result_state_df["var"] = result_state_var
    state_list[i] = result_state_df


# tot_res1 = pd.DataFrame(res_list, columns = ['phi', 'psi', 'sigma2_r', 'sigma2_e', 'sigma2_w', "intercept_obs_1",
# "intercept_obs_2", "intercept_state_1", "isin"]).reset_index()
# tot_state1 = pd.concat(state_list,axis=0).reset_index(drop=True)
# tot_res.to_csv("ssm_res.csv")
# tot_state.to_csv("ssm_state.csv")



# for i in range(110,len(isin_list)):
#     df = px_nav.loc[px_nav["isin"]==isin_list[i],["price","nav"]]
#     df.index = px_nav.loc[px_nav["isin"]==isin_list[i], "date"]
#     df.index = pd.to_datetime(df.index)
#     df_complete = pd.DataFrame(index=[d for d in pd.bdate_range(start=max(date(2018,1,3),min(df.index.date)), 
#                                                 end=min(max(df.index.date), date(2022,10,22))) if cal.is_working_day(d)])
#     df_complete = df_complete.join(df)
#     df_complete = df_complete.fillna(method="ffill")
#     df_complete.dropna(inplace=True)
#     df_complete.loc["2020-01-01":"2020-06-01",:].plot(title=str(i)+isin_list[i])
#     plt.savefig("./px_nav_plot/" + str(i)+isin_list[i] + ".jpg")
    

isin_info = pd.read_excel("AUG_INCEPT_MKTCAP.xlsx")
# for i in range(len(isin_list)):
for i in range(len(isin_list)):
    isin_ = state_list[i].loc[state_list[i].index[0], "isin"]
    name_ = isin_info.loc[isin_info["isin"]==isin_, "Name"]
    if len(name_):
        state_list[i].loc["2020-01-01":"2020-06-01",["price","nav","state"]].plot(title=name_.iloc[0])
        plt.savefig("./px_nav_fitted_plot/" + str(i)+isin_list[i] + ".jpg")


for i in range(len(isin_list)):
    state_list[i].loc[:,["price","nav","state"]].plot()
    plt.savefig("./px_nav_fitted_plot/" + str(i)+isin_list[i] + ".jpg")

tot_res = pd.DataFrame(res_list[:5], columns = ['phi', 'psi', 'sigma2_r', 'sigma2_e', 'sigma2_w', "intercept_obs_1",
"intercept_obs_2", "intercept_state_1", "obs_err_cov_lower", "isin"]).reset_index()