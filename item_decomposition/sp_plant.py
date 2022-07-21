def sp_plant(j):

    from GlobalVariable import globalVariables as gbv
    from gurobipy import gurobipy as gp
    from gurobipy import GRB

    prob = gp.Model("plant")

    # variable
    vj = prob.addMVar((gbv.K, gbv.N))   # v_{ijt} for i,t
    xj = prob.addMVar((gbv.K, gbv.N))  # x_{ijt} for i,t
    rj = prob.addMVar((gbv.A, gbv.N))  # r_{ajt} for a,t
    yUIj = prob.addMVar((gbv.K, gbv.N))  # y^{I}_{ijt} for i,t
    yUOj = prob.addMVar((gbv.K, gbv.N))  # y^{O}_{ijt} for i,t
    sc1j = prob.addMVar((gbv.K, len(gbv.out_place[j]), gbv.N))   # s_{il(j)t} for i,t,l=(j,j')
    sc2j = prob.addMVar((gbv.K, len(gbv.in_place[j]), gbv.N))  # s_{il(j)t} for i,t,l=(j',j)
    zcj = prob.addMVar((gbv.K, gbv.N))    # z_{ij(j)t} for i,t
    #wj = prob.addMVar((gbv.K, gbv.N), vtype=GRB.INTEGER)  # w_{i,j,t} for i,t
    #xUbj = prob.addMVar((gbv.K, gbv.N), vtype=GRB.BINARY)

    # Constraint
    # (10b)
    prob.addConstrs((vj[i][0] - gbv.v0[i][j] - yUIj[i][0] + yUOj[i][0] == 0 for i in range(gbv.K)), name='10b1')
    prob.addConstrs((vj[i][t] - vj[i][t-1] - yUIj[i][t] + yUOj[i][t] == 0 for i in range(gbv.K)
                     for t in range(1, gbv.N)), name='10b2')

    # (10c)
    prob.addConstrs((yUIj[i][t] - gp.quicksum(
        sc2j[i][ll][t - int(gbv.dtUL[i][gbv.in_place[j][ll]])] for ll in range(len(gbv.in_place[j]))
        if t - gbv.dtUL[i][gbv.in_place[j][ll]] >= 0)
                     - xj[i][t - int(gbv.dtUP[i][j])] == gbv.Q[i][j][t] for i in range(gbv.K) for t in range(gbv.N) if
                     t - gbv.dtUP[i][j] >= 0), name='10c1')
    prob.addConstrs((yUIj[i][t] - gp.quicksum(
        sc2j[i][ll][t - int(gbv.dtUL[i][gbv.in_place[j][ll]])] for ll in range(len(gbv.in_place[j]))
        if t - gbv.dtUL[i][gbv.in_place[j][ll]] >= 0) == gbv.Q[i][j][t] for i in range(gbv.K) for t in range(gbv.N) if
                    t - gbv.dtUP[i][j] < 0), name='10c2')

    # (10d)
    prob.addConstrs((yUOj[i][t] - gp.quicksum(sc1j[i][ll][t] for ll in range(len(gbv.out_place[j])))
                     - gp.quicksum(gbv.q[e] * xj[gbv.cE[e][1]-1][t] for e in range(gbv.E) if gbv.cE[e][0] == i+1)
                     - zcj[i][t] - gp.quicksum(rj[a][t] for a in range(gbv.A) if gbv.cA[a][0] == i+1)
                     + gp.quicksum(rj[a][t] for a in range(gbv.A) if gbv.cA[a][1] == i+1) == 0
                     for i in range(gbv.K) for t in range(gbv.N)), name='10d')

    # (10e)
    prob.addConstrs((gp.quicksum(gbv.nu[i][j] * xj[i][t] for i in range(gbv.K)) <= gbv.C[j][t] for t in range(gbv.N)),
                    name='10e')

    # (10f)
    prob.addConstrs((rj[a][t] >= 0 for a in range(gbv.A) for t in range(gbv.N)), name='10f1')
    prob.addConstrs((rj[a][t] - vj[i][t] <= 0 for t in range(gbv.N) for i in range(gbv.K)
                     for a in range(gbv.A) if gbv.cA[a][0] == i+1), name='10f2')

    # (10g)
    prob.addConstrs((yUOj[i][t] >= 0 for t in range(gbv.N) for i in range(gbv.K)), name='10g1')
    prob.addConstrs((vj[i][t] - yUOj[i][t] >= 0 for t in range(gbv.N) for i in range(gbv.K)), name='10g2')

    # (10h)
    prob.addConstrs((xj[i][t] >= 0 for t in range(gbv.N) for i in range(gbv.K)), name='10h1')
    prob.addConstrs((yUIj[i][t] >= 0 for t in range(gbv.N) for i in range(gbv.K)), name='10h2')
    prob.addConstrs((vj[i][t] >= 0 for t in range(gbv.N) for i in range(gbv.K)), name='10h3')
    prob.addConstrs((zcj[i][t] >= 0 for t in range(gbv.N) for i in range(gbv.K)), name='10h4')

    # (10i)
    prob.addConstrs(
        (sc1j[i][ll][t] >= 0 for t in range(gbv.N) for ll in range(len(gbv.out_place[j])) for i in range(gbv.K)),
        name='10i1')
    prob.addConstrs(
        (sc2j[i][ll][t] >= 0 for t in range(gbv.N) for ll in range(len(gbv.in_place[j])) for i in range(gbv.K)),
        name='10i1')
    '''
    # (10j)
    prob.addConstrs((xj[i][t] - gbv.m[i] * wj[i][t] == 0 for i in range(gbv.K)
                     for t in range(gbv.N)), name='10j')

    # (10k)
    prob.addConstrs((xUbj[i][t] * gbv.wLB[i][j] - wj[i][t] <= 0 for i in range(gbv.K) for t in
                     range(gbv.N)),
                    name='10k1')
    prob.addConstrs((xUbj[i][t] * gbv.wUB[i][j] - wj[i][t] >= 0 for i in range(gbv.K) for t in
                     range(gbv.N)),
                    name='10k2')
    '''
    if gbv.ite == 0:
        prob.setObjective(gp.quicksum(gbv.H[i][j] * vj[i][t] for i in range(gbv.K) for t in range(gbv.N))
                          - gp.quicksum(gbv.Lam[i][j][t] * zcj[i][t] for t in range(gbv.N)
                                        for i in range(gbv.K))
                          - gp.quicksum(gbv.Eta[i][gbv.out_place[j][ll]][t] * sc1j[i][ll][t] for t in range(gbv.N)
                                        for ll in range(len(gbv.out_place[j])) for i in range(gbv.K))
                          - gp.quicksum(gbv.Xi[i][gbv.in_place[j][ll]][t] * sc2j[i][ll][t] for t in range(gbv.N)
                                        for ll in range(len(gbv.in_place[j])) for i in range(gbv.K))
                          )
    else:
        prob.setObjective(gp.quicksum(gbv.H[i][j] * vj[i][t] for i in range(gbv.K) for t in range(gbv.N))
                          - gp.quicksum(gbv.Lam[i][j][t] * zcj[i][t] for t in range(gbv.N)
                                        for i in range(gbv.K))
                          - gp.quicksum(gbv.Eta[i][gbv.out_place[j][ll]][t] * sc1j[i][ll][t] for t in range(gbv.N)
                                        for ll in range(len(gbv.out_place[j])) for i in range(gbv.K))
                          - gp.quicksum(gbv.Xi[i][gbv.in_place[j][ll]][t] * sc2j[i][ll][t] for t in range(gbv.N)
                                        for ll in range(len(gbv.in_place[j])) for i in range(gbv.K))
                          + gbv.rho / 2 * gp.quicksum((gbv.Z[i][j][t] - zcj[i][t]) ** 2 for t in range(gbv.N)
                                                      for i in range(gbv.K))
                          + gbv.rho / 2 * gp.quicksum(
            (gbv.S[i][gbv.out_place[j][ll]][t] - sc1j[i][ll][t]) ** 2 for t in range(gbv.N)
            for ll in range(len(gbv.out_place[j])) for i in range(gbv.K))
                          + gbv.rho / 2 * gp.quicksum(
            (gbv.S[i][gbv.in_place[j][ll]][t] - sc2j[i][ll][t]) ** 2 for t in range(gbv.N)
            for ll in range(len(gbv.in_place[j])) for i in range(gbv.K))
                          )

    prob.optimize()
    gbv.V[:, j, :] = vj.X
    gbv.X[:, j, :] = xj.X
    gbv.R[:, j, :] = rj.X
    gbv.YUI[:, j, :] = yUIj.X
    gbv.YUO[:, j, :] = yUOj.X
    gbv.SC1[:, gbv.out_place[j], :] = sc1j.X
    gbv.SC2[:, gbv.in_place[j], :] = sc2j.X
    gbv.ZC[:, j, :] = zcj.X
    #gbv.W[:, j, :] = wj.X
    #gbv.XUb[:, j, :] = xUbj.X
