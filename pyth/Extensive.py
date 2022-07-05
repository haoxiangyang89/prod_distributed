def solve_extensive(dat):
    #(1)

    prob = gp.Model("extensive")

    # variable
    v = prob.addMVar((K, M, N))  # (i,j,t)
    u = prob.addMVar((K, N))  # (i,t)
    z = prob.addMVar((K, M, N))  # (i,j,t)
    yUI = prob.addMVar((K, M, N))  # (i,j,t)
    yUO = prob.addMVar((K, M, N))  # (i,j,t)
    x = prob.addMVar((K, M, N))  # (i,j,t)
    r = prob.addMVar((A, M, N))  # (a,j,t)
    s = prob.addMVar((K, L, N))  # (i,l,t)
    w = prob.addMVar((K, M, N), vtype=GRB.INTEGER)  # (i,j,t)
    xUb = prob.addMVar((K, M, N), vtype=GRB.BINARY)  # (i,j,t)

    # constraint()
    # (1b)-(1c)
    prob.addConstrs((u[i][0] + gp.quicksum(z[i][j][0] for j in range(M)) == D[i][0] for i in range(K)), name='1b-1c1')
    prob.addConstrs(
        (u[i][t] - u[i][t - 1] + gp.quicksum(z[i][j][t] for j in range(M)) == D[i][t] for i in range(K) for t in
         range(1, N)), name='1c2')

    # (1d)
    #prob.addConstrs(
    #    (gp.quicksum(u[i][tt] / D[i][tt] for i in range(K) for tt in range(t + 1) if D[i][tt] != 0) <= epsilonUD[t] for
    #     t in range(N)), name='1d')

    # (1e)
    prob.addConstrs((v[i][j][0] - yUI[i][j][0] + yUO[i][j][0] == v0[i][j] for i in range(K) for j in range(M)),
                    name='1e1')
    prob.addConstrs(
        (v[i][j][t] - v[i][j][t - 1] - yUI[i][j][t] + yUO[i][j][t] == 0 for i in range(K) for j in range(M) for t in
         range(1, N)), name='1e2')

    # (1f)
    prob.addConstrs((gp.quicksum(
        360 / (12 * N) * v[i][j][tt] - epsilonUI * yUO[i][j][tt] for j in range(M) for tt in range(t + 1)) <= 0 for i in
                     range(K) for t in range(N)), name='1f')

    # (1g)
    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - dtUL[i][l]] for l in range(L) if cL[l][1] == j + 1 and t - dtUL[i][l] >= 0) - x[i][j][
             t - int(dtUP[i][j])] == Q[i][j][t] for i in range(K) for j in range(M) for t in range(N) if
         t >= dtUP[i][j]),
        name='1g1')

    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - dtUL[i][l]] for l in range(L) if cL[l][1] == j + 1 and t - dtUL[i][l] >= 0) == Q[i][j][t] for i
         in range(K) for j in range(M) for t in range(N) if t < dtUP[i][j]),
        name='1g2')

    # (1h)
    prob.addConstrs((yUO[i][j][t] - gp.quicksum(s[i][l][t] for l in range(L) if cL[l][0] == j + 1) - gp.quicksum(
        q[e] * x[cE[e][1] - 1][j][t] for e in range(E) if cE[e][0] == i + 1) - z[i][j][t] - gp.quicksum(
        r[a][j][t] for a in range(A) if cA[a][0] == i + 1) + gp.quicksum(
        r[a][j][t] for a in range(A) if cA[a][1] == i + 1) == 0 for i in range(K) for j in range(M) for t in range(N)),
                    name='1h')

    # (1i)
    prob.addConstrs(
        (gp.quicksum(nu[i][j] * x[i][j][t] for i in range(K)) <= C[j][t] for j in range(M) for t in range(N)),
        name='1i')

    # (1j)
    prob.addConstrs((r[a][j][t] >= 0 for j in range(M) for t in range(N) for a in range(A)), name='1j1')
    prob.addConstrs(
        (v[i][j][t] - r[a][j][t] >= 0 for j in range(M) for t in range(N) for i in range(K) for a in range(A) if
         cA[a][0] == i + 1),
        name='1j2')

    # (1k)
    prob.addConstrs((yUO[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)),
                    name='1k1')
    prob.addConstrs(
        (v[i][j][t] - yUO[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)),
        name='1k2')

    # (1l)
    prob.addConstrs((x[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)), name='1l1')
    prob.addConstrs((yUI[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)), name='1l2')
    prob.addConstrs((v[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)), name='1l3')
    prob.addConstrs((z[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)), name='1l4')

    # (1m)
    prob.addConstrs((u[i][t] >= 0 for i in range(K) for t in range(N)), name='1m')

    # (1n)
    prob.addConstrs((s[i][l][t] >= 0 for i in range(K) for l in range(L) for t in range(N)), name='1n')

    # (1o)
    prob.addConstrs((x[i][j][t] - m[i] * w[i][j][t] == 0 for i in range(K)
                     for j in range(M)
                     for t in range(N)), name='1o')

    # (1p)
    prob.addConstrs((xUb[i][j][t] * wLB[i][j] - w[i][j][t] <= 0 for i in range(K) for j in range(M) for t in range(N)),
                    name='1p1')
    prob.addConstrs((xUb[i][j][t] * wUB[i][j] - w[i][j][t] >= 0 for i in range(K) for j in range(M) for t in range(N)),
                    name='1p2')

    # objective
    prob.setObjective(gp.quicksum(H[i][j]*v[i][j][t] for i in range(K) for j in range(M) for t in range(N)))

    return prob