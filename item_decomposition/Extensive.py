def solve_extensive():
    # (1)
    from GlobalVariable import globalVariables as gbv
    from gurobipy import gurobipy as gp
    from gurobipy import GRB

    prob = gp.Model("extensive")

    # variable
    v = prob.addMVar((gbv.K, gbv.M, gbv.N))  # (i,j,t)
    u = prob.addMVar((gbv.K, gbv.N))  # (i,t)
    z = prob.addMVar((gbv.K, gbv.M, gbv.N))  # (i,j,t)
    yUI = prob.addMVar((gbv.K, gbv.M, gbv.N))  # (i,j,t)
    yUO = prob.addMVar((gbv.K, gbv.M, gbv.N))  # (i,j,t)
    x = prob.addMVar((gbv.K, gbv.M, gbv.N))  # (i,j,t)
    r = prob.addMVar((gbv.A, gbv.M, gbv.N))  # (a,j,t)
    s = prob.addMVar((gbv.K, gbv.L, gbv.N))  # (i,l,t)
    #w = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.INTEGER)  # (i,j,t)
    #xUb = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.BINARY)  # (i,j,t)

    # constraint()
    # (1b)-(1c)
    prob.addConstrs((u[i][0] + gp.quicksum(z[i][j][0] for j in range(gbv.M)) == gbv.D[i][0] for i in range(gbv.K)),
                    name='1b-1c1')
    prob.addConstrs(
        (u[i][t] - u[i][t - 1] + gp.quicksum(z[i][j][t] for j in range(gbv.M)) == gbv.D[i][t] for i in range(gbv.K)
         for t in range(1, gbv.N)), name='1c2')

    # (1d)
    #prob.addConstrs(
    #    (gp.quicksum(u[i][tt] / D[i][tt] for i in range(K) for tt in range(t + 1) if D[i][tt] != 0) <= epsilonUD[t] for
    #     t in range(N)), name='1d')

    # (1e)
    prob.addConstrs((v[i][j][0] - yUI[i][j][0] + yUO[i][j][0] == gbv.v0[i][j] for i in range(gbv.K)
                     for j in range(gbv.M)), name='1e1')
    prob.addConstrs(
        (v[i][j][t] - v[i][j][t - 1] - yUI[i][j][t] + yUO[i][j][t] == 0 for i in range(gbv.K) for j in range(gbv.M)
         for t in range(1, gbv.N)), name='1e2')

    # (1f)
    # prob.addConstrs((gp.quicksum(
    #     360 / (12 * N) * v[i][j][tt] - epsilonUI * yUO[i][j][tt] for j in range(M) for tt in range(t + 1)) <= 0
    #     for i in range(K) for t in range(N)), name='1f')

    # (1g)
    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - int(gbv.dtUL[i][l])] for l in range(gbv.L) if gbv.cL[l][1] == j + 1
            and t - gbv.dtUL[i][l] >= 0) - x[i][j][t - int(gbv.dtUP[i][j])] == gbv.Q[i][j][t] for i in range(gbv.K)
         for j in range(gbv.M) for t in range(gbv.N) if t >= gbv.dtUP[i][j]),
        name='1g1')

    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - gbv.dtUL[i][l]] for l in range(gbv.L) if gbv.cL[l][1] == j + 1 and t - gbv.dtUL[i][l] >= 0) ==
         gbv.Q[i][j][t] for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N) if t < gbv.dtUP[i][j]),
        name='1g2')

    # (1h)
    prob.addConstrs((yUO[i][j][t] - gp.quicksum(s[i][l][t] for l in range(gbv.L) if gbv.cL[l][0] == j + 1) -
                     gp.quicksum(gbv.q[e] * x[gbv.cE[e][1] - 1][j][t] for e in range(gbv.E) if gbv.cE[e][0] == i + 1)
                     - z[i][j][t] - gp.quicksum(
        r[a][j][t] for a in range(gbv.A) if gbv.cA[a][0] == i + 1) + gp.quicksum(
        r[a][j][t] for a in range(gbv.A) if gbv.cA[a][1] == i + 1) == 0 for i in range(gbv.K)
                     for j in range(gbv.M) for t in range(gbv.N)),
                    name='1h')

    # (1i)
    prob.addConstrs(
        (gp.quicksum(gbv.nu[i][j] * x[i][j][t] for i in range(gbv.K)) <= gbv.C[j][t] for j in range(gbv.M)
         for t in range(gbv.N)), name='1i')

    # (1j)
    prob.addConstrs((r[a][j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N) for a in range(gbv.A)), name='1j1')
    prob.addConstrs(
        (v[i][j][t] - r[a][j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N) for i in range(gbv.K)
         for a in range(gbv.A) if gbv.cA[a][0] == i + 1),
        name='1j2')

    # (1k)
    prob.addConstrs((yUO[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)),
                    name='1k1')
    prob.addConstrs(
        (v[i][j][t] - yUO[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)),
        name='1k2')

    # (1l)
    prob.addConstrs((x[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)), name='1l1')
    prob.addConstrs((yUI[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)), name='1l2')
    prob.addConstrs((v[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)), name='1l3')
    prob.addConstrs((z[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)), name='1l4')

    # (1m)
    prob.addConstrs((u[i][t] >= 0 for i in range(gbv.K) for t in range(gbv.N)), name='1m')

    # (1n)
    prob.addConstrs((s[i][l][t] >= 0 for i in range(gbv.K) for l in range(gbv.L) for t in range(gbv.N)), name='1n')

    # (1o)
    #prob.addConstrs((x[i][j][t] - gbv.m[i] * w[i][j][t] == 0 for i in range(gbv.K)
    #                 for j in range(gbv.M)
    #                 for t in range(gbv.N)), name='1o')

    # (1p)
    #prob.addConstrs((xUb[i][j][t] * gbv.wLB[i][j] - w[i][j][t] <= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)),
    #                name='1p1')
    #prob.addConstrs((xUb[i][j][t] * gbv.wUB[i][j] - w[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)),
    #                name='1p2')

    # objective
    prob.setObjective(gp.quicksum(gbv.H[i][j]*v[i][j][t] for i in range(gbv.K) for j in range(gbv.M)
                                  for t in range(gbv.N))
                      + gp.quicksum(gbv.P[i][t] * u[i][t] for i in range(gbv.K) for t in range(gbv.N)))

    prob.optimize()

    gbv.X_E = x.X
    gbv.S_E = s.X
    gbv.Z_E = z.X
    gbv.R_E = r.X
    gbv.V_E = v.X
    gbv.U_E = u.X
    gbv.YUI_E = yUI.X
    gbv.YUO_E = yUO.X
    #gbv.W_E = w.X
    #gbv.XUb_E = xUb.X

    return prob.objVal