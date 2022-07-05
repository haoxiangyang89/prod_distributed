def sp_item(i):
    # eliminate constraints (1d) and (1f)

    from GlobalVariable import globalVariables as gbv
    from gurobipy import gurobipy as gp

    prob = gp.Model("item")

    # variable
    ui = prob.addMVar(gbv.N)  # u_{it} for t
    si = prob.addMVar((gbv.L, gbv.N))  # s_{ilt} for l,t
    zi = prob.addMVar((gbv.M, gbv.N))  # z_{ijt} for j,t
    vi = prob.addMVar((gbv.M, gbv.N))  # v_{ijt} for j,t
    yUIi = prob.addMVar((gbv.M, gbv.N))  # y^{I}_{ijt} for j,t
    yUOi = prob.addMVar((gbv.M, gbv.N))  # y^{O}_{ijt} for j,t
    xCi = prob.addMVar((gbv.K, gbv.M, gbv.N))  # x_{i'j(i)t} for i',j,t
    rCi = prob.addMVar((gbv.A, gbv.M, gbv.N))  # r_{aj(i)t} for a,j,t


    # Constraint
    # (3b)-(3c)
    prob.addConstr(ui[0] + gp.quicksum(zi[j][0] for j in range(gbv.M)) == gbv.D[i][0], name='3b+3c1')
    prob.addConstrs(
        (ui[t] - ui[t - 1] + gp.quicksum(zi[j][t] for j in range(gbv.M)) == gbv.D[i][t] for t in
         range(1, gbv.N)), name='3c2')

    # (3e)
    prob.addConstrs((vi[j][0] - yUIi[j][0] + yUOi[j][0] == gbv.v0[i][j] for j in range(gbv.M)),
                    name='3e1')
    prob.addConstrs(
        (vi[j][t] - vi[j][t - 1] - yUIi[j][t] + yUOi[j][t] == 0 for j in range(gbv.M) for t in
         range(1, gbv.N)), name='3e2')

    # (3g)
    prob.addConstrs(
        (yUIi[j][t] - gp.quicksum(
            si[ll][t - int(gbv.dtUL[i][ll])] for ll in range(gbv.L) if gbv.cL[ll][1] == j + 1 and t - gbv.dtUL[i][ll] >= 0) -
         xCi[i][j][
             t - int(gbv.dtUP[i][j])] == gbv.Q[i][j][t] for j in range(gbv.M) for t in range(gbv.N) if
         t >= gbv.dtUP[i][j]),
        name='3g1')
    prob.addConstrs(
        (yUIi[j][t] - gp.quicksum(
            si[ll][t - gbv.dtUL[i][ll]] for ll in range(gbv.L) if
            gbv.cL[ll][1] == j + 1 and t - gbv.dtUL[i][ll] >= 0) ==
         gbv.Q[i][j][t] for j in
         range(gbv.M) for t in range(gbv.N) if t < gbv.dtUP[i][j]),
        name='3g2')

    # (3h)
    prob.addConstrs((yUOi[j][t] - gp.quicksum(si[ll][t] for ll in range(gbv.L) if gbv.cL[ll][0] == j + 1) - gp.quicksum(
        gbv.q[e] * xCi[gbv.cE[e][1]][j][t] for e in range(gbv.E) if gbv.cE[e][0] == i + 1) - zi[j][t] - gp.quicksum(
        rCi[a][j][t] for a in range(gbv.A) if gbv.cA[a][0] == i + 1) + gp.quicksum(
        rCi[a][j][t] for a in range(gbv.A) if gbv.cA[a][1] == i + 1) == 0 for j in range(gbv.M) for t in range(gbv.N)),
                    name='3h')

    # (3i)
    prob.addConstrs(
        (gp.quicksum(gbv.nu[ii][j] * xCi[ii][j][t] for ii in range(gbv.K)) <= gbv.C[j][t] for j in range(gbv.M) for t in
         range(gbv.N)),
        name='3i')

    # (3j)
    prob.addConstrs(
        (rCi[a][j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N) for a in range(gbv.A) if gbv.cA[a][0] == i + 1),
        name='3j1')
    prob.addConstrs(
        (vi[j][t] - rCi[a][j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N) for a in range(gbv.A) if
         gbv.cA[a][0] == i + 1),
        name='3j2')

    # (3k)
    prob.addConstrs((yUOi[j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N)),
                    name='3k1')
    prob.addConstrs(
        (vi[j][t] - yUOi[j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N)),
        name='3k2')

    # (3l)
    prob.addConstrs((yUIi[j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N)), name='3l')
    prob.addConstrs((vi[j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N)), name='3l2')
    prob.addConstrs((zi[j][t] >= 0 for j in range(gbv.M) for t in range(gbv.N)), name='3l3')

    # (3m)
    prob.addConstrs((xCi[ii][j][t] >= 0 for ii in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N)), name='3m')

    # (3n)
    prob.addConstrs((ui[t] >= 0 for t in range(gbv.N)), name='3n')

    # (3o)
    prob.addConstrs((si[ll][t] >= 0 for ll in range(gbv.L) for t in range(gbv.N)), name='3o')

    # objective
    if gbv.ite == 0:
        prob.setObjective(gp.quicksum(gbv.H[i][j] * vi[j][t] for j in range(gbv.M) for t in range(gbv.N))
                          + gp.quicksum(gbv.P[i][t] * ui[t] for t in range(gbv.N))
                          - gp.quicksum(
            gbv.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(gbv.K) for j in
            range(gbv.M) for t in
            range(gbv.N))
                          - gp.quicksum(
            gbv.Ksi[a][j][i][t] * rCi[a][j][t] for a in range(gbv.A) for j in range(gbv.M) for t in range(gbv.N))
                          )
    else:
        prob.setObjective(gp.quicksum(gbv.H[i][j] * vi[j][t] for j in range(gbv.M) for t in range(gbv.N))
                          + gp.quicksum(gbv.P[i][t] * ui[t] for t in range(gbv.N))
                          - gp.quicksum(
            gbv.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(gbv.K) for j in
            range(gbv.M) for t in
            range(gbv.N))
                          - gp.quicksum(
            gbv.Ksi[a][j][i][t] * rCi[a][j][t] for a in range(gbv.A) for j in range(gbv.M) for t in range(gbv.N))
                          + gbv.rho / 2 * gp.quicksum(
            (gbv.X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(gbv.K) for j in range(gbv.M) for t in range(gbv.N))
                          + gbv.rho / 2 * gp.quicksum(
            (gbv.R[a][j][t] - rCi[a][j][t]) ** 2 for a in range(gbv.A) for j in range(gbv.M) for t in range(gbv.N))
                          )

    prob.optimize()
    gbv.U[i] = ui.X
    gbv.S[i] = si.X
    gbv.Z[i] = zi.X
    gbv.V[i] = vi.X
    gbv.YUI[i] = yUIi.X
    gbv.YUO[i] = yUOi.X
    gbv.XC[:, :, i, :] = xCi.X
    gbv.RC[:, :, i, :] = rCi.X
