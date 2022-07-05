def sp_global():
    # eliminate constraints (1d) and (1f)

    from GlobalVariable import globalVariables as gbv
    import numpy as np

    gbv.X = np.maximum(0, np.sum(gbv.rho * gbv.XC - gbv.Mu, axis=2) / (gbv.rho * gbv.K))
    gbv.R = np.maximum(0, np.sum(gbv.rho * gbv.RC - gbv.Ksi, axis=2) / (gbv.rho * gbv.K))

    # Update dual variables
    for i in range(gbv.K):
        for ii in range(gbv.K):
            for j in range(gbv.M):
                for t in range(gbv.N):
                    gbv.Mu[i][ii][j][t] += gbv.rho * (gbv.X[i][j][t] - gbv.XC[i][j][ii][t])
    for a in range(gbv.A):
        for j in range(gbv.M):
            for i in range(gbv.K):
                for t in range(gbv.N):
                    gbv.Ksi[a][j][i][t] += gbv.rho*(gbv.R[a][j][t]-gbv.RC[a][j][i][t])
