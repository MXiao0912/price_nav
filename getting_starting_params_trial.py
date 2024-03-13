  # getting starting parameters
#   psi0 = ARIMA(df_complete["price"], order=(1,1,1)).fit().arparams
#   phi0 = ARIMA(df_complete["nav"], order=(1,1,1)).fit().arparams
#   df_diff = df_complete.diff()
#   df_diff.loc[:,"price"] = df_diff["price"]-psi0 * df_diff["price"].shift(1)
#   df_diff.loc[:,"nav"] = df_diff["nav"]-phi0 * df_diff["nav"].shift(1)
#   df_diff.dropna(inplace=True)
#   mod0 = sm.tsa.VARMAX(df_diff, order=(0,1), trend="n")
#   res0 = mod0.fit(maxiter=200, disp=False)
#   sigma2_z = np.matrix([[res0.params["sqrt.var.price"]**2, res0.params["sqrt.cov.price.nav"]**2],
#                        [res0.params["sqrt.cov.price.nav"]**2, res0.params["sqrt.var.nav"]**2]])
#   theta = np.matrix([[res0.params["L1.e(price).price"], res0.params["L1.e(nav).price"]],
#                      [res0.params["L1.e(price).nav"], res0.params["L1.e(nav).nav"]]])
#   RHS = (sigma2_z+theta@sigma2_z@np.transpose(theta))
#   sigma2_r = RHS[0,1]/(1-phi0)
#   sigma2_e = RHS[0,0]-(1+psi0**2)*sigma2_r
#   sigma2_w = RHS[1,1]-((1-phi0)**2)*sigma2_r
  
#   result_param, result_state, result_state_var = fit_param(df_complete, [psi0[0],phi0[0],max(sigma2_r[0],0),max(sigma2_e[0],0),max(sigma2_w[0],0),0,0,0])