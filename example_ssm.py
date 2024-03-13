# rs = np.random.RandomState(12345)

# # Specify some parameters to simulate data and
# # check the model
# nobs = 10000
# params = np.r_[0.1, 0.1, 0.05, 0.05, 0.1, 1.5, 2.5, 0,0, 0]

# # Simulate data
# v = rs.normal(scale=params[4]**0.5, size=nobs + 2).cumsum()
# e = rs.normal(scale=params[5]**0.5, size=nobs + 2)
# w = rs.normal(scale=params[6]**0.5, size=nobs + 2)

# p = np.zeros(nobs + 2)
# n = np.zeros(nobs + 2)

# for t in range(2, nobs+2):
#     # print(p[t-1],n[t],v[t],w[t], e[t])
#     # print(params[1],params[3])
#     p[t] = params[0] * p[t - 1] + params[2] * p[t - 2] + v[t] - params[0] * v[t - 1] - params[2] * v[t - 2] + e[t]
#     n[t] = params[1] * n[t - 1] + params[3] * n[t - 2] + (1 - params[1] - params[3]) * v[t] + w[t]
    
# y = np.c_[p, n]

# # Run MLE routine on the fitted data

# mod = SSM_two_lags(y)
# res = mod.fit(disp=False, maxiter=500)
# res.smoothed_state[1,:]

# Check the estimated parameters
# print(pd.DataFrame({
#     'True': params,
#     'Estimated': res.params
# }, index=mod.param_names).round(2))