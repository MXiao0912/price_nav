
class SSM_vec(sm.tsa.statespace.MLEModel):
    

    def __init__(self, endog, params0, lag):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=lag, trim='both', original='in')
        tot_col = endog.shape[1]
        num_etf = int(tot_col/2)
        self.num_etf = num_etf
        self.ind_all = endog.index[lag:]
        endog = X[:, :tot_col]
        self.lagged_endog = X[:, tot_col:]
        self._start_params = params0
        self.lag = lag
        self._param_names = [f'psi_{isin}_{i}' for isin in range(num_etf) for i in range(1,lag+1) ] +\
                            [f'phi_{isin}_{i}' for isin in range(num_etf) for i in range(1,lag+1) ] +\
                            [f'sigma_r_{i}_{j}' for i in range(num_etf) for j in range(i+1)] +\
                            [f'sigma_obs_{i}_{j}' for i in range(tot_col) for j in range(i+1)] +\
                            [f'intercept_obs_{i}' for i in range(tot_col)] +\
                            [f'intercept_state_{i}' for i in range(num_etf)]
                            # [f'covid_{i}' for i in range(num_etf)]
        
        # Initialize the state space model
        # Note: because your state process is nonstationary,
        # you can't use stationary initialization
        super().__init__(endog, k_states=(lag+1)*(num_etf), k_posdef=num_etf,
                         initialization='diffuse')

        # Setup the fixed components of the state space representation
        design = np.vstack((np.ones((1,lag+1)),np.zeros((1, lag+1))))
        design[1,0] = 1
        self['design'] = block_diag(*[design for _ in range(num_etf)])
        transition = np.zeros((1, lag+1))
        transition[0,0] = 1
        transition = np.vstack((transition, np.hstack((np.eye(lag),np.zeros((lag,1))))))
        self['transition'] = block_diag(*[transition for _ in range(num_etf)])
        selection = np.zeros((lag+1,1))
        selection[0,0] = 1
        self['selection'] = block_diag(*[selection for _ in range(num_etf)])

    # For the parameter estimation part, it is
    # helpful to use parameter transformations to
    # keep the parameters in the valid domain. Here
    # I assumed that you wanted phi and psi to be
    # between (-1, 1).
        
    def transform_params(self, params):
        params = params.copy()
        for i in range(0, 2*lag*self.num_etf, lag):
            params[i:i + lag] = tools.constrain_stationary_univariate(params[i:i + lag])
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        for i in range(0, 2*lag*self.num_etf, lag):
            params[i:i + lag] = tools.unconstrain_stationary_univariate(params[i:i + lag])
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

        etf_mat_list = []
        for e in range(num_etf):
            param_etf_position = [i for i, item in enumerate(param_names) if re.match(rf'^(psi|phi)_{e}_(\d+)$',item)]
            param_etf = [params[m] for m in param_etf_position]
            intercept_p = []
            intercept_n = []
            for param in param_etf[:lag]:
                intercept_p.extend([param, 0])
            for param in param_etf[lag:2*lag]:
                intercept_n.extend([0, param])
            etf_mat_list.append(np.array([intercept_p, intercept_n]))
        
        intercept_obs_postion = [i for i, item in enumerate(param_names) if re.match(rf'^intercept_obs_(\d+)$',item)]
        intercept_obs = [params[m] for m in intercept_obs_postion]

        intercept_state_postion = [i for i, item in enumerate(param_names) if re.match(rf'^intercept_state_(\d+)$',item)]
        intercept_state = [params[m] for m in intercept_state_postion]
        modified_intercept_state = []
        # Iterate over the original list
        for i, elem in enumerate(intercept_state):
            modified_intercept_state.append(elem)  # Add the current element
            modified_intercept_state.extend([0 for _ in range(lag)]) 

        self['obs_intercept'] = block_diag(etf_mat_list) @ self.lagged_endog.T+ np.tile(np.array(intercept_obs).reshape(-1,1),self.lagged_endog.shape[0])
        self['state_intercept'] = np.tile(np.array(modified_intercept_state).reshape(-1,1), self.lagged_endog.shape[0])
        # self['state_intercept'] = self['state_intercept'] + np.vstack((np.hstack(([[params[2*lag+7]]], np.zeros((1,lag)))), np.zeros((lag,lag+1)))) @ np.tile((self.ind_all.month == 3) & (self.ind_all.year == 2020), lag+1).reshape(lag+1, -1).astype(int)
        for e in range(num_etf):
            for i in range(lag*e+1, lag*e+lag+1):
                self['design', 0, i] = -params[i-1]
        params_acc = 0
        for i in range(lag, 2*lag):
            params_acc += params[i]
        self['design', 1, 0]  = 1 - params_acc
        self['state_cov', 0, 0] = params[2*lag]
        self['obs_cov'] = np.array([[params[2*lag+1],params[2*lag+6]],[params[2*lag+6],params[2*lag+2]]])
        lower = np.array([[params[2*lag+1], 0], [params[2*lag+6], params[2*lag+2]]])
        self['obs_cov'] = lower@(lower.T)



class SSM_single_price(sm.tsa.statespace.MLEModel):
    

    def __init__(self, endog, params0, lag):
        # Extract lagged endog
        X = sm.tsa.lagmat(endog, maxlag=lag, trim='both', original='in')
        # self.ind_all = endog.index[lag:]
        endog = X[:, :1]
        self.lagged_endog = X[:, 1:]
        self._start_params = params0
        self.lag = lag
        self._param_names = [f'psi_{i}' for i in range(1,lag+1)] + ['sigma2_r', 'sigma2_e',
                    "intercept_obs", "intercept_state"]
        
        # Initialize the state space model
        # Note: because your state process is nonstationary,
        # you can't use stationary initialization
        super().__init__(endog, k_states=lag+1, k_posdef=1,
                         initialization='diffuse')

        # Setup the fixed components of the state space representation
        design = np.ones((1,lag+1))
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
        params[:lag] = tools.constrain_stationary_univariate(params[:lag])
        params[lag:(lag+1)] = params[lag:(lag+1)]**2
        return params
    
    def untransform_params(self, params):
        params = params.copy()
        params[:lag] = tools.unconstrain_stationary_univariate(params[:lag])
        params[lag:(lag+1)] = params[lag:(lag+1)]**0.5
        return params

    # Describe how parameters enter the model
    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        # Here is where we're putting the lagged endog, multiplied
        # by phi and psi into the intercept term
        # self['obs_intercept'] = self.lagged_endog.T * params[:2, None]
        
        self['obs_intercept'] = np.array(params[:lag]) @ self.lagged_endog.T+ np.tile([params[lag+2]],self.lagged_endog.shape[0]).reshape(1,-1)
        self['state_intercept'] = np.tile(np.vstack(([params[lag+3]], np.zeros((lag,1)))), self.lagged_endog.shape[0])
        for i in range(1, lag+1):
            self['design', 0, i] = -params[i-1]
        self['state_cov', 0, 0] = params[lag]
        self['obs_cov'] = np.array([params[lag+1]])
