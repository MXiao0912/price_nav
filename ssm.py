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

px_nav = pd.read_csv("px_nav.csv")
px_nav = px_nav.sort_values("date")
px_nav = px_nav[["isin","date","price","nav"]]
px_nav["price"] = np.log(px_nav["price"])
px_nav["nav"] = np.log(px_nav["nav"])
# px_nav = na.omit(px_nav)



class SSM(sm.tsa.statespace.MLEModel):
    

    # If you use _param_names and _start_params then the model
    # will automatically pick them up
    _param_names = ['psi', 'phi', 'sigma2_r', 'sigma2_e', 'sigma2_w',
                       "intercept_obs_1", "intercept_obs_2", "intercept_state_1", "covid"]
    # _start_params = [-0.33, -0.44, 0.1, 1, 1, 0, 0, 0, 0, -1]

    def __init__(self, endog, params0):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=1, trim='both', original='in')
        self.ind_all = endog.index[1:]
        endog = X[:, :2]
        self.lagged_endog = X[:, 2:]
        self._start_params = params0

        # Initialize the state space model
        # Note: because your state process is nonstationary,
        # you can't use stationary initialization
        super().__init__(endog, k_states=2, k_posdef=1,
                         initialization='diffuse')

        # Setup the fixed components of the state space representation
        self['design'] = [[1., 1.],
                          [1., 0]]
        self['transition'] = [[1., 0],
                              [1., 0]]
        self['selection'] = [[1],
                             [0]]

    # For the parameter estimation part, it is
    # helpful to use parameter transformations to
    # keep the parameters in the valid domain. Here
    # I assumed that you wanted phi and psi to be
    # between (-1, 1).
        
    def transform_params(self, params):
        params = params.copy()
        for i in range(2):
            params[i:i + 1] = tools.constrain_stationary_univariate(params[i:i + 1])
        params[2:5] = params[2:5]**2
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        for i in range(2):
            params[i:i + 1] = tools.unconstrain_stationary_univariate(params[i:i + 1])
        params[2:5] = params[2:5]**0.5
        return params

    # Describe how parameters enter the model
    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        # Here is where we're putting the lagged endog, multiplied
        # by phi and psi into the intercept term
        # self['obs_intercept'] = self.lagged_endog.T * params[:2, None]
        self['obs_intercept'] = self.lagged_endog.T * params[:2, None]+ np.tile([[params[5]],[params[6]]],self.lagged_endog.shape[0])
        self['state_intercept'] = np.tile([[params[7]],[0]],self.lagged_endog.shape[0])
        self['state_intercept'] = self['state_intercept'] + np.array([[params[8], 0], [0, 0]]) @ np.tile((self.ind_all.month == 3) & (self.ind_all.year == 2020), 2).reshape(2, -1).astype(int)
        self['design', 0, 1] = -params[0]
        self['design', 1, 0] = 1 - params[1]
        self['state_cov', 0, 0] = params[2]
        self['obs_cov'] = np.diag([params[3],params[4]])


class SSM_two_lags(sm.tsa.statespace.MLEModel):
    

    # If you use _param_names and _start_params then the model
    # will automatically pick them up
    _param_names = ['psi_1', 'psi_2', 'phi_1', 'phi_2', 'sigma2_r', 'sigma2_e', 'sigma2_w', 
    "intercept_obs_1", "intercept_obs_2", "intercept_state_1", "covid"]
    # _start_params = [0.5, 0.5, 0.5, 0.5, 0.1, 1, 1, 0, 0, 0]

    def __init__(self, endog, params0):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=2, trim='both', original='in')
        self.ind_all = endog.index[2:]
        endog = X[:, :2]
        self.lagged_endog = X[:, 2:]
        self._start_params = params0

        # Initialize the state space model
        # Note: because your state process is nonstationary,
        # you can't use stationary initialization
        super().__init__(endog, k_states=3, k_posdef=1,
                         initialization='diffuse')

        # Setup the fixed components of the state space representation
        self['design'] = [[1., 1., 1.],
                          [1., 0, 0]]
        self['transition'] = [[1., 0, 0],
                              [1., 0, 0],
                              [0,  1.,0]]
        self['selection'] = [[1],
                             [0],
                             [0]]

    # For the parameter estimation part, it is
    # helpful to use parameter transformations to
    # keep the parameters in the valid domain. Here
    # I assumed that you wanted phi and psi to be
    # between (-1, 1).
        
    def transform_params(self, params):
        params = params.copy()
        for i in [0,2]:
            params[i:i + 2] = tools.constrain_stationary_univariate(params[i:i + 2])
        params[4:7] = params[4:7]**2
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        for i in [0,2]:
            params[i:i + 2] = tools.unconstrain_stationary_univariate(params[i:i + 2])
        params[4:7] = params[4:7]**0.5
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

        self['obs_intercept'] = np.array([intercept_p, intercept_n]) @ self.lagged_endog.T+ np.tile([[params[7]],[params[8]]],self.lagged_endog.shape[0])
        self['state_intercept'] = np.tile([[params[9]],[0],[0]],self.lagged_endog.shape[0])
        self['state_intercept'] = self['state_intercept'] + np.array([[params[10], 0, 0], [0, 0, 0], [0,0,0]]) @ np.tile((self.ind_all.month == 3) & (self.ind_all.year == 2020), 3).reshape(3, -1).astype(int)
        self['design', 0, 1] = -params[0]
        self['design', 0, 2] = -params[1]
        self['design', 1, 0] = 1 - params[2] - params[3]
        self['state_cov', 0, 0] = params[4]
        self['obs_cov'] = np.diag(params[5:7])

        
class SSM_vec(sm.tsa.statespace.MLEModel):
    

    # If you use _param_names and _start_params then the model
    # will automatically pick them up
    _param_names = ['psi', 'phi', 'sigma2_r', 'sigma2_e', 'sigma2_w', 'obs_cov',
                       "intercept_obs_1", "intercept_obs_2", "intercept_state_1"]
    _start_params = [0.5*np.eye(137), 0.5*np.eye(137), 0.1*np.eye(), 1, 1, 0, 0, 0, 0]

    def __init__(self, endog):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=1, trim='both', original='in')
        self.ind_all = endog.index[1:]
        endog = X[:, :2]
        self.lagged_endog = X[:, 2:]

        # Initialize the state space model
        # Note: because your state process is nonstationary,
        # you can't use stationary initialization
        super().__init__(endog, k_states=2, k_posdef=1,
                         initialization='diffuse')

        # Setup the fixed components of the state space representation
        self['design'] = [[1., 1.],
                          [1., 0]]
        self['transition'] = [[1., 0],
                              [1., 0]]
        self['selection'] = [[1],
                             [0]]

    # For the parameter estimation part, it is
    # helpful to use parameter transformations to
    # keep the parameters in the valid domain. Here
    # I assumed that you wanted phi and psi to be
    # between (-1, 1).
        
    def transform_params(self, params):
        params = params.copy()
        for i in range(2):
            params[i:i + 1] = tools.constrain_stationary_univariate(params[i:i + 1])
        params[2:] = params[2:]**2
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        for i in range(2):
            params[i:i + 1] = tools.unconstrain_stationary_univariate(params[i:i + 1])
        params[2:] = params[2:]**0.5
        return params

    # Describe how parameters enter the model
    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        # Here is where we're putting the lagged endog, multiplied
        # by phi and psi into the intercept term
        # self['obs_intercept'] = self.lagged_endog.T * params[:2, None]
        self['obs_intercept'] = self.lagged_endog.T * params[:2, None]+ np.tile([[params[6]],[params[7]]],self.lagged_endog.shape[0])
        self['state_intercept'] = np.tile([[params[8]],[0]],self.lagged_endog.shape[0])
        # self['state_intercept'] = self['state_intercept'] + np.matrix([[params[8],0],[0,0]]) @ np.tile((self.ind_all.month==3) & (self.ind_all.year==2020),2).reshape(2,-1).astype(int)
        self['design', 0, 1] = -params[0]
        self['design', 1, 0] = 1 - params[1]
        self['state_cov', 0, 0] = params[2]
        self['obs_cov'] = np.matrix([[params[3],params[5]],[params[5],params[4]]])


class SSM_anylag(sm.tsa.statespace.MLEModel):
    

    def __init__(self, endog, params0, lag):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=lag, trim='both', original='in')
        self.ind_all = endog.index[lag:]
        endog = X[:, :2]
        self.lagged_endog = X[:, 2:]
        self._start_params = params0
        self.lag = lag
        self._param_names = [f'psi_{i}' for i in range(1,lag+1)] + [f'phi_{i}' for i in range(1,lag+1)] + ['sigma2_r', 'sigma2_e', 'sigma2_w',
                    "intercept_obs_1", "intercept_obs_2", "intercept_state_1", "covid"]
        
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
        params[(2*lag):(2*lag+3)] = params[(2*lag):(2*lag+3)]**2
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        for i in [0, lag]:
            params[i:i + lag] = tools.unconstrain_stationary_univariate(params[i:i + lag])
        params[(2*lag):(2*lag+3)] = params[(2*lag):(2*lag+3)]**0.5
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

        self['obs_intercept'] = np.array([intercept_p, intercept_n]) @ self.lagged_endog.T+ np.tile([[params[2*lag+3]],[params[2*lag+4]]],self.lagged_endog.shape[0])
        self['state_intercept'] = np.tile(np.vstack(([params[2*lag+5]], np.zeros((lag,1)))), self.lagged_endog.shape[0])
        self['state_intercept'] = self['state_intercept'] + np.vstack((np.hstack(([[params[2*lag+6]]], np.zeros((1,lag)))), np.zeros((lag,lag+1)))) @ np.tile((self.ind_all.month == 3) & (self.ind_all.year == 2020), lag+1).reshape(lag+1, -1).astype(int)
        for i in range(1, lag+1):
            self['design', 0, i] = -params[i-1]
        params_acc = 0
        for i in range(lag, 2*lag):
            params_acc += params[i]
        self['design', 1, 0]  = 1 - params_acc
        self['state_cov', 0, 0] = params[2*lag]
        self['obs_cov'] = np.diag([params[2*lag+1],params[2*lag+2]])


def fit_param(df, params0, lag, method="lbfgs",maxit=200):
  mod = SSM_anylag(df,params0,lag)
  res = mod.fit(disp=False, method=method, maxiter=maxit)
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
  print(str(i)+isin_list[i])
  df = px_nav.loc[px_nav["isin"]==isin_list[i],["price","nav"]]
  df.index = px_nav.loc[px_nav["isin"]==isin_list[i], "date"]
  df.index = pd.to_datetime(df.index)
  df_complete = pd.DataFrame(index=pd.bdate_range(start="01/03/2018", end="10/22/2022"))
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
  
  price_auto = auto_arima(df_start["price"], start_p=1, start_q=1)
  nav_auto = auto_arima(df_start["nav"], start_p=1, start_q=1)
  lag = max(len(price_auto.arparams()), len(price_auto.maparams()), len(nav_auto.arparams()), len(nav_auto.maparams()), 1)
  print(lag)

  warnings.filterwarnings("ignore")
  psi_res = ARIMA(df_start["price"], order=(lag,0,lag)).fit()
  phi_res = ARIMA(df_start["nav"], order=(lag,0,1)).fit()
  warnings.filterwarnings("default")

  covid_drop = df_complete.loc[(df_complete.index.year==2020)&(df_complete.index.month==3),].mean().mean()-df_complete.loc[(df_complete.index.year==2020)&(df_complete.index.month==2),].mean().mean()

  print("fit1 start")
  use_alternative_model = False
  # Catch ConvergenceWarning
  with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.filterwarnings("always")
    try:
        result_param, result_state, result_state_var = fit_param(df_complete, psi_res.arparams.tolist() + phi_res.arparams.tolist() + [0.1, 1, 1, 0, 0, 0, covid_drop], lag=lag)
        if caught_warnings:  # If any warnings were caught
            use_alternative_model = True
    except:
        use_alternative_model = True
  if use_alternative_model:
        print("fit2 start")
        result_param, result_state, result_state_var = fit_param(df_complete, result_param, method="nm", maxit=3000, lag=lag)

  res_list[i] = np.append(result_param, isin_list[i])
  result_state_df = df_complete.iloc[:-lag,].copy()
  result_state_df["state"] = result_state
  result_state_df["isin"] = isin_list[i]
  result_state_df["var"] = result_state_var
  state_list[i] = result_state_df

  
# non-converge: 9/137, 25/137

tot_res1 = pd.DataFrame(res_list, columns = ['phi', 'psi', 'sigma2_r', 'sigma2_e', 'sigma2_w', "intercept_obs_1",
"intercept_obs_2", "intercept_state_1", "isin"]).reset_index()
tot_state1 = pd.concat(state_list,axis=0).reset_index(drop=True)
tot_state1["date"] = tot_state1["date"].astype(str)
tot_res.to_csv("ssm_res.csv")
tot_state.to_csv("ssm_state.csv")




vec form -- better estimation of the mispricing var & cov as we take into account the cross-asset cov

lag flexible form -- check if the autocorr problem can be solved by higher lags
