def sp_global(type):
    # eliminate constraints (1d) and (1f)

    from GlobalVariable import globalVariables as gbv
    from gurobipy import gurobipy as gp
    import numpy as np

    if type == 'item':
        gbv.X = np.maximum(0, np.sum(gbv.rho * gbv.XC - gbv.Mu, axis=1)) / (gbv.rho * gbv.K)

        gbv.R = np.maximum(0, gbv.RC1 + gbv.RC2 - (gbv.Ksi + gbv.Eta) / gbv.rho) / 2

        if gbv.ite > 0:
            # Update dual variables
            for ii in range(gbv.K):
                gbv.Mu[:, ii, :, :] += gbv.rho * (gbv.X - gbv.XC[:, ii, :, :])

            gbv.Ksi += gbv.rho * (gbv.R - gbv.RC1)
            gbv.Eta += gbv.rho * (gbv.R - gbv.RC2)
                        
    if type == 'plant':
        prob = gp.Model("plant_global")

        u = prob.addMVar((gbv.K, gbv.N))    # u_{it}
        z = prob.addMVar((gbv.K, gbv.M, gbv.N))     # z_{ijt}

        prob.addConstrs((u[i][0] + gp.quicksum(z[i][j][0] for j in range(gbv.M)) == gbv.D[i][0] for i in range(gbv.K)),
                        name='11b+11c-1')
        prob.addConstrs((u[i][t] - u[i][t-1] + gp.quicksum(z[i][j][t] for j in range(gbv.M)) == gbv.D[i][t]
                         for i in range(gbv.K) for t in range(1, gbv.N)), name='11b+11c-2')

        prob.addConstrs((u[i][t] >= 0 for i in range(gbv.K) for t in range(gbv.N)), name='11d1')
        prob.addConstrs((z[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)),
                        name='11d2')

        prob.setObjective(gp.quicksum(gbv.Lam[i][jj][j][t] * z[i][jj][t] for i in range(gbv.K) for j in range(gbv.M)
                                      for jj in range(gbv.M)  for t in range(gbv.N)) + gbv.rho / 2 *
                          gp.quicksum((z[i][jj][t] - gbv.ZC[i][jj][j][t]) ** 2 for i in range(gbv.K)
                                      for j in range(gbv.M) for jj in range(gbv.M)  for t in range(gbv.N))
                          + gp.quicksum(gbv.P[i][t] * u[i][t] for i in range(gbv.K) for t in range(gbv.N)))

        prob.optimize()
        gbv.Z = z
        gbv.U = u

        gbv.S = np.maximum(0, np.sum(gbv.SC - gbv.Eta, axis=2) / (gbv.rho * gbv.M))

        if gbv.ite > 0:
            for i in range(gbv.K):
                for j in range(gbv.M):
                    for jj in range(gbv.M):
                        for t in range(gbv.N):
                            gbv.Lam[i][j][jj][t] += gbv.rho * (gbv.Z[i][j][t] - gbv.ZC[i][j][jj][t])
            for i in range(gbv.K):
                for ll in range(gbv.L):
                    for j in range(gbv.M):
                        for t in range(gbv.N):
                            gbv.Eta[i][ll][j][t] += gbv.rho * (gbv.S[i][ll][t] - gbv.SC[i][ll][j][t])

    if type == 'time':
        gbv.U = np.maximum(0, np.sum(gbv.rho * gbv.UC - gbv.Lam, axis=2) / (gbv.rho * gbv.N))
        gbv.S = np.maximum(0, np.sum(gbv.rho * gbv.SC - gbv.Mu, axis=3) / (gbv.rho * gbv.N))
        gbv.X = np.maximum(0, np.sum(gbv.rho * gbv.XC - gbv.Eta, axis=3) / (gbv.rho * gbv.N))

        tem = np.sum(gbv.rho * gbv.VC - gbv.Ksi, axis=3)
        for i in range(gbv.K):
            for j in range(gbv.M):
                for t in range(gbv.N):
                    gbv.V[i][j][t] = max(0, tem[i][j][t] - gbv.H[i][j]) / (gbv.rho * gbv.N)

        if gbv.ite > 0:
            for i in range(gbv.K):
                for t in range(gbv.N):
                    for tt in range(gbv.N):
                        gbv.Lam[i][t][tt] += gbv.rho * (gbv.U[i][t] - gbv.UC[i][t][tt])
            for i in range(gbv.K):
                for ll in range(gbv.L):
                    for t in range(gbv.N):
                        for tt in range(gbv.N):
                            gbv.Mu[i][ll][t][tt] += gbv.rho * (gbv.S[i][ll][t] - gbv.SC[i][ll][t][tt])
            for i in range(gbv.K):
                for j in range(gbv.M):
                    for t in range(gbv.N):
                        for tt in range(gbv.N):
                            gbv.Ksi[i][j][t][tt] += gbv.rho * (gbv.V[i][j][t] - gbv.VC[i][j][t][tt])
                            gbv.Eta[i][j][t][tt] += gbv.rho * (gbv.X[i][j][t] - gbv.XC[i][j][t][tt])

