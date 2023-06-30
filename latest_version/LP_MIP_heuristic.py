'''item-based decomposition algorithm'''
import functools

'''Server process managers are more flexible than using shared memory objects because they can be made to support 
arbitrary object types like lists, dictionaries, Queue, Value, Array, etc. They are, however, slower than using 
shared memory.'''

'''
import os
os.environ("VECLIB_MAXIMUM_THREADS") = "1"  #export VECLIB_MAXIMUM_THREADS=1
os.environ("MKL_NUM_THREADS") = "1"
os.environ("NUMEXPR_NUM_THREADS") = "1"
os.environ("OMP_NUM_THREADS") = "1"
'''

from gurobipy import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import time
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Process
from functools import partial
# from ray.util.multiprocessing import Pool
import random
from itertools import repeat
from multiprocessing import set_start_method


# import functools
# import pathos.pools as pp
# from pathos.pp import ParallelPool
# from pathos.multiprocessing import ProcessingPool as Pool
# from pathos.helpers import cpu_count

# item_model_var = dict()

def generate_data(seed):
    random.seed(seed)
    np.random.seed(seed)

    data = dict()
    var = dict()


    # 随机生成数据
    data["N"] = 30  # T:time 16; 30
    data["M"] = 3  # J:plant 6/90; 3
    data["K"] = 5  # I:item 78/600; 5

    # set of transportation arc
    cL = []
    cLL = np.random.choice(a=[False, True], size=(data["M"], data["M"]), p=[0.9, 0.1])
    for j1 in range(data["M"]):
        for j2 in range(data["M"]):
            if j2 != j1 and cLL[j1][j2]:
                cL.append((j1 + 1, j2 + 1))
    data["cL"] = cL
    data["L"] = len(cL)

    # 生成bom tree
    #tr_depth = data["K"] // 4
    tr_depth = 3
    tr_index = np.random.choice(a=list(range(tr_depth)), size=(1, data["K"]))
    tr_index_dic = {}
    for d in range(tr_depth):
        tr_index_dic[d] = []
    for i in range(data["K"]):
        tr_index_dic[tr_index[0][i]].append(i + 1)
    cE = []
    for d in range(tr_depth):
        for i in tr_index_dic[d]:
            for dd in range(d + 1, tr_depth):
                for ii in tr_index_dic[dd]:
                    tem = random.uniform(0, 1)
                    if tem > 0.9:
                        cE.append((i, ii))
    data["cE"] = cE
    data["E"] = len(cE)

    # set of replacement arc
    cA = []
    cAA = np.random.choice(a=[False, True], size=(data["K"], data["K"]), p=[0.9, 0.1])
    for i1 in range(1, data["K"]):
        for i2 in range(0, i1):
            if cAA[i1][i2]:
                cA.append((i1 + 1, i2 + 1))
                cA.append((i2 + 1, i1 + 1))
    data["cA"] = cA
    data["A"] = len(cA)

    data["H"] = np.random.randint(1, 5, size=(data["K"], data["M"]))   # holding cost

    data["P"] = np.random.randint(200, 500, size=(data["K"], data["N"]))   # penalty on unmet demand

    data["D"] = np.random.randint(10, 50, size=(data["K"], data["N"]))     # demand 10,50

    # epsilonUD=np.ones(N)*0.05
    # epsilonUI=0.8

    data["v0"] = np.random.randint(5, size=(data["K"], data["M"]))    # initial inventory

    data["nu"] = np.random.randint(1, 3, size=(data["K"], data["M"]))    # unit capacity used for production

    data["C"] = np.random.randint(data["K"] * 2 * 10, data["K"] * 2 * 30, size=(data["M"], data["N"]))    # production capacity

    data["dtUL"] = np.random.randint(1, 3, size=(data["K"], data["L"]))  # delta t^L _il, transportation time

    data["dtUP"] = np.random.randint(1, 3, size=(data["K"], data["M"]))  # delta t^P_ij, production time

    data["Q"] = np.zeros((data["K"], data["M"], data["N"]))     # purchase delivery

    data["q"] = np.random.randint(1, 5, size=data["E"])    # production consumption relationship on bom tree

    # gbv.m = 5 * np.ones(gbv.K)
    # gbv.wLB = 1 * np.ones((gbv.K, gbv.M))
    # gbv.wUB = 20 * np.ones((gbv.K, gbv.M))
    '''

    # 小规模数据
    data["N"] = 2  # T:time
    data["M"] = 3  # J:plant
    data["K"] = 3  # I:item
    data["L"] = 2  # number of transportation arcs
    data["E"] = 1  # number of production arcs in the bom tree
    data["A"] = 1  # number of replacement arcs

    H = np.array([[2, 1, 3]])
    data["H"] = np.tile(H, (data["K"], 1))  # holding cost

    data["P"] = 200 * np.ones((data["K"], data["N"]))  # penalty on unmet demand

    data["D"] = np.array([[5, 7], [10, 8], [6, 4]])  # demand

    # gbv.epsilonUD = np.ones(gbv.N) * 0.05
    # gbv.epsilonUI = 0.8

    data["v0"] = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])  # initial inventory

    data["nu"] = np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]])  # unit capacity used for production

    C = np.array([[50], [50], [50]])  # production capacity
    data["C"] = np.tile(C, (1, data["N"]))

    data["cL"] = [(1, 2), (2, 3)]  # set of transportation arcs
    data["cE"] = [(2, 1)]  # set of production relationship arcs
    data["cA"] = [(3, 2)]  # set of replacement arcs

    data["dtUL"] = np.ones((data["K"], data["L"]))  # delta t^L _il, transportation time

    data["dtUP"] = np.zeros((data["K"], data["M"]))  # delta t^P_ij, production time

    data["Q"] = np.zeros((data["K"], data["M"], data["N"]))  # purchase delivery

    data["q"] = 2 * np.ones(data["E"])  # production consumption relationship on bom tree
    '''

    # 每个item作为集合A中边的出端(i,i')或者入端(i',i)时，相应边在集合A中对应的序号
    out_place = []
    for i in range(data["K"]):
        tem = []
        for a in range(data["A"]):
            if data["cA"][a][0] == i + 1:
                tem.append(a)
        out_place.append(tem)
    data["out_place"] = out_place

    in_place = []
    for i in range(data["K"]):
        tem = []
        for a in range(data["A"]):
            if data["cA"][a][1] == i + 1:
                tem.append(a)
        in_place.append(tem)
    data["in_place"] = in_place

    '''parameters'''
    # parameters["pri_re"] = []
    # parameters["d_re"] = []
    data["MaxIte"] = 1000  # 最大迭代次数
    # parameters for computing primal/dual tolerance
    data["ep_abs"] = 1e-2
    data["ep_rel"] = 1e-4

    # magnitude of penalty on z-,y- relative to original variables' costs
    data["mag"] = 1e4

    '''variables'''
    var["X"] = np.zeros((data["K"], data["M"], data["N"]))  # production
    var["S"] = np.zeros((data["K"], data["L"], data["N"]))  # transportation
    var["Z_P"] = np.zeros((data["K"], data["M"], data["N"]))  # supply
    var["Z_M"] = np.zeros((data["K"], data["M"], data["N"]))  # supply
    var["R"] = np.zeros((data["A"], data["M"], data["N"]))  # replacement
    var["V"] = np.zeros((data["K"], data["M"], data["N"]))  # inventory
    var["U"] = np.zeros((data["K"], data["N"]))  # unmet demand
    var["YUI"] = np.zeros((data["K"], data["M"], data["N"]))  # inbound quantity
    var["YUO_P"] = np.zeros((data["K"], data["M"], data["N"]))  # outbound quantity
    var["YUO_M"] = np.zeros((data["K"], data["M"], data["N"]))  # outbound quantity

    # variable fixing
    var["X_FIX"] = np.zeros((data["K"], data["M"], data["N"]))  # production
    var["R_FIX"] = np.zeros((data["A"], data["M"], data["N"]))  # replacement
    var["Bool_X_FIX"] = np.zeros((data["K"], data["M"], data["N"]))  # production
    var["Bool_R_FIX"] = np.zeros((data["A"], data["M"], data["N"]))  # replacement
    # 矩阵x的扩张，x_{i(i')jt}=x_{ijt} for i'
    #var["xx"] = np.zeros((data["K"], data["K"], data["M"], data["N"]))

    # Copying variables
    var["XC"] = np.zeros((data["K"], data["K"], data["M"], data["N"]))  # x_{i(i')jt}
    var["RC1"] = np.zeros((data["A"], data["M"], data["N"]))  # r_{a(i)jt}
    var["RC2"] = np.zeros((data["A"], data["M"], data["N"]))  # r_{a(i')jt}

    # Dual variables
    var["Mu"] = np.zeros((data["K"], data["K"], data["M"], data["N"]))
    var["Ksi"] = np.zeros((data["A"], data["M"], data["N"]))
    var["Eta"] = np.zeros((data["A"], data["M"], data["N"]))

    # parameters
    var["rho_X"] = 2 * np.ones((data["K"], data["M"], data["N"]))  # penalty on X
    var["rho_R"] = 2 * np.ones((data["A"], data["M"], data["N"]))  # replacement
    # 初始化变量
    var["ite"] = 0  # ite为迭代次数

    # fixing variable hyper-parameter
    var["lag_X"] = 1   # 4
    var["lag_R"] = 1   # 10

    # LP warm start iterations
    var["LP_warm_start_ite"] = 200 #200

    return data, var

def solver(data):
    prob = gp.Model("extensive")

    # variable
    v = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    u = prob.addMVar((data["K"], data["N"]))  # (i,t)
    z_p = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    z_m = prob.addMVar((data["K"], data["M"], data["N"]))
    yUI = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    yUO_p = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    yUO_m = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    x = prob.addMVar((data["K"], data["M"], data["N"]), vtype=GRB.INTEGER)  # (i,j,t)
    r = prob.addMVar((data["A"], data["M"], data["N"]))  # (a,j,t)
    s = prob.addMVar((data["K"], data["L"], data["N"]))  # (i,l,t)
    #w = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.INTEGER)  # (i,j,t)
    #xUb = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.BINARY)  # (i,j,t)

    # constraint
    # (1b)-(1c)
    prob.addConstrs((u[i][0] + gp.quicksum(z_p[i][j][0] - z_m[i][j][0] for j in range(data["M"])) == data["D"][i][0] for i in range(data["K"])),
                    name='1b-1c1')
    prob.addConstrs(
        (u[i][t] - u[i][t - 1] + gp.quicksum(z_p[i][j][t] - z_m[i][j][t] for j in range(data["M"])) == data["D"][i][t] for i in range(data["K"])
         for t in range(1, data["N"])), name='1c2')

    # (1d)
    # prob.addConstrs(
    #    (gp.quicksum(u[i][tt] / D[i][tt] for i in range(K) for tt in range(t + 1) if D[i][tt] != 0) <= epsilonUD[t] for
    #     t in range(N)), name='1d')

    # (1e)
    prob.addConstrs((v[i][j][0] - yUI[i][j][0] + yUO_p[i][j][0] - yUO_m[i][j][0] == data["v0"][i][j] for i in range(data["K"])
                     for j in range(data["M"])), name='1e1')
    prob.addConstrs(
        (v[i][j][t] - v[i][j][t - 1] - yUI[i][j][t] + yUO_p[i][j][t] - yUO_m[i][j][t] == 0 for i in range(data["K"]) for j in range(data["M"])
         for t in range(1, data["N"])), name='1e2')

    # (1f)
    # prob.addConstrs((gp.quicksum(
    #     360 / (12 * N) * v[i][j][tt] - epsilonUI * yUO[i][j][tt] for j in range(M) for tt in range(t + 1)) <= 0
    #     for i in range(K) for t in range(N)), name='1f')

    # (1g)
    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - int(data["dtUL"][i][l])] for l in range(data["L"]) if data["cL"][l][1] == j + 1
            and t - data["dtUL"][i][l] >= 0) - x[i][j][t - int(data["dtUP"][i][j])] == data["Q"][i][j][t] for i in range(data["K"])
         for j in range(data["M"]) for t in range(data["N"]) if t >= data["dtUP"][i][j]),
        name='1g1')
    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - data["dtUL"][i][l]] for l in range(data["L"]) if data["cL"][l][1] == j + 1 and t - data["dtUL"][i][l] >= 0) ==
         data["Q"][i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"]) if t < data["dtUP"][i][j]),
        name='1g2')

    # (1h)
    prob.addConstrs((yUO_p[i][j][t] - yUO_m[i][j][t] - gp.quicksum(s[i][l][t] for l in range(data["L"]) if data["cL"][l][0] == j + 1) -
                     gp.quicksum(data["q"][e] * x[data["cE"][e][1] - 1][j][t] for e in range(data["E"]) if data["cE"][e][0] == i + 1)
                     - (z_p[i][j][t] - z_m[i][j][t]) - gp.quicksum(
        r[a][j][t] for a in range(data["A"]) if data["cA"][a][0] == i + 1) + gp.quicksum(
        r[a][j][t] for a in range(data["A"]) if data["cA"][a][1] == i + 1) == 0 for i in range(data["K"])
                     for j in range(data["M"]) for t in range(data["N"])),
                    name='1h')

    # (1i)
    prob.addConstrs(
        (gp.quicksum(data["nu"][i][j] * x[i][j][t] for i in range(data["K"])) <= data["C"][j][t] for j in range(data["M"])
         for t in range(data["N"])), name='1i')

    # (1j)
    #prob.addConstrs((r[a][j][t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in range(data["A"])), name='1j1')
    prob.addConstrs(
        (data["v0"][i][j] - r[a][j][0] >= 0 for j in range(data["M"]) for i in
         range(data["K"])
         for a in range(data["A"]) if data["cA"][a][0] == i + 1),
        name='1j2')
    prob.addConstrs(
        (v[i][j][t - 1] - r[a][j][t] >= 0 for j in range(data["M"]) for t in range(1, data["N"]) for i in range(data["K"])
         for a in range(data["A"]) if data["cA"][a][0] == i + 1),
        name='1j3')

    # (1k)
    #prob.addConstrs((yUO_p[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
    #                name='1k1')
    #prob.addConstrs((yUO_m[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
    #                name='1k1')
    prob.addConstrs(
        (data["v0"][i][j] - (yUO_p[i][j][0] - yUO_m[i][j][0]) >= 0 for i in range(data["K"]) for j in range(data["M"])),
        name='3k2')
    prob.addConstrs(
        (v[i][j][t - 1] - (yUO_p[i][j][t] - yUO_m[i][j][t]) >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(1, data["N"])),
        name='1k2')

    # (1l)
    #prob.addConstrs((x[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l1')
    #prob.addConstrs((yUI[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l2')
    #prob.addConstrs((v[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l3')
    #prob.addConstrs((z_p[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l4')
    #prob.addConstrs((z_m[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
    #                name='1l4')

    # (1m)
    #prob.addConstrs((u[i][t] >= 0 for i in range(data["K"]) for t in range(data["N"])), name='1m')

    # (1n)
    #prob.addConstrs((s[i][l][t] >= 0 for i in range(data["K"]) for l in range(data["L"]) for t in range(data["N"])), name='1n')
    '''
    # (1o)
    prob.addConstrs((x[i][j][t] - gbv.m[i] * w[i][j][t] == 0 for i in range(gbv.K)
                     for j in range(gbv.M)
                     for t in range(gbv.N)), name='1o')

    # (1p)
    prob.addConstrs((xUb[i][j][t] * gbv.wLB[i][j] - w[i][j][t] <= 0 for i in range(gbv.K) for j in range(gbv.M) for t in
                     range(gbv.N)),
                    name='1p1')
    prob.addConstrs((xUb[i][j][t] * gbv.wUB[i][j] - w[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in
                     range(gbv.N)),
                    name='1p2')
    '''
    # objective
    prob.setObjective(gp.quicksum(
        data["H"][i][j] * v[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"]))
                      + gp.quicksum(data["P"][i][t] * u[i][t] for i in range(data["K"]) for t in range(data["N"]))
                      + gp.quicksum(
        data["mag"] * data["H"][i][j] * yUO_m[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in
        range(data["N"]))
                      + gp.quicksum(
        data["mag"] * data["P"][i][t] * z_m[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in
        range(data["N"])))
    #time_end1 = time.time()
    #time_e_model = time_end1 - time_start1

    #time_start2 = time.time()
    prob.optimize()
    #time_end2 = time.time()
    #time_e_op = time_end2 - time_start2

    #gbv.W_E = w.X
    #gbv.XUb_E = xUb.X

    X_sol = np.reshape(x.X, (-1, 1))
    S_sol = np.reshape(s.X, (-1, 1))
    Z_P_sol = np.reshape(z_p.X, (-1, 1))
    Z_M_sol = np.reshape(z_m.X, (-1, 1))
    R_sol = np.reshape(r.X, (-1, 1))
    V_sol = np.reshape(v.X, (-1, 1))
    U_sol = np.reshape(u.X, (-1, 1))
    YUI_sol = np.reshape(yUI.X, (-1, 1))
    YUO_P_sol = np.reshape(yUO_p.X, (-1, 1))
    YUO_M_sol = np.reshape(yUO_m.X, (-1, 1))

    Sol = np.concatenate((X_sol, S_sol, Z_P_sol, Z_M_sol, R_sol, V_sol, U_sol, YUI_sol, YUO_P_sol, YUO_M_sol), axis=0)

    return prob.objVal, Sol
    #x.X, s.X, z.X, r.X, v.X, u.X, yUI.X, yUO.X

def model_var_setup(data, i):
    # 建立item i的model和variable
    prob = gp.Model("item " + str(i))

    #prob.setParam("Threads", 1)
    #prob.Params.Threads = 1

    # variable
    # variable
    ui = prob.addMVar(data["N"])  # u_{it} for t
    si = prob.addMVar((data["L"], data["N"]))  # s_{ilt} for l,t
    zi_p = prob.addMVar((data["M"], data["N"]))  # z_{ijt} for j,t
    zi_m = prob.addMVar((data["M"], data["N"]))  # z_{ijt} for j,t
    vi = prob.addMVar((data["M"], data["N"]))  # v_{ijt} for j,t
    yUIi = prob.addMVar((data["M"], data["N"]))  # y^{I}_{ijt} for j,t
    yUOi_p = prob.addMVar((data["M"], data["N"]))  # y^{O}_{ijt} for j,t
    yUOi_m = prob.addMVar((data["M"], data["N"]))  # y^{O}_{ijt} for j,t
    xCi = prob.addMVar((data["K"], data["M"], data["N"]))  # x_{i'j(i)t} for i',j,t
    if len(data["out_place"][i]) > 0:
        rC1i = prob.addMVar((len(data["out_place"][i]), data["M"], data["N"]))  # r_{a(i)jt} for a=(i,i')
    if len(data["in_place"][i]) > 0:
        rC2i = prob.addMVar((len(data["in_place"][i]), data["M"], data["N"]))  # r_{a(i)jt} for a=(i',i)
    # xUbi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.BINARY)   # x^{b}_{ijt} for j,t
    # wi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.INTEGER)   # w_{ijt} for j,t

    # Constraint
    # unmet demand transition
    prob.addConstr(ui[0] + gp.quicksum(zi_p[j, 0] - zi_m[j, 0] for j in range(data["M"])) == data["D"][i][0], name='3b+3c1')
    prob.addConstrs(
        (ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in range(data["M"])) == data["D"][i][t] for t in
         range(1, data["N"])), name='3c2')

    # inventory transition
    prob.addConstrs((vi[j, 0] - yUIi[j, 0] + yUOi_p[j, 0] - yUOi_m[j, 0] == data["v0"][i][j] for j in range(data["M"])),
                    name='3e1')
    prob.addConstrs(
        (vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in range(data["M"]) for t in
         range(1, data["N"])), name='3e2')

    # inbound total
    prob.addConstrs(
        (yUIi[j, t] - gp.quicksum(
            si[ll, t - int(data["dtUL"][i][ll])] for ll in range(data["L"]) if
            data["cL"][ll][1] == j + 1 and t - data["dtUL"][i][ll] >= 0) -
         xCi[i, j,
             t - int(data["dtUP"][i][j])] == data["Q"][i][j][t] for j in range(data["M"]) for t in range(data["N"])
         if
         t >= data["dtUP"][i][j]),
        name='3g1')
    prob.addConstrs(
        (yUIi[j, t] - gp.quicksum(
            si[ll, t - data["dtUL"][i][ll]] for ll in range(data["L"]) if
            data["cL"][ll][1] == j + 1 and t - data["dtUL"][i][ll] >= 0) ==
         data["Q"][i][j][t] for j in
         range(data["M"]) for t in range(data["N"]) if t < data["dtUP"][i][j]),
        name='3g2')

    # outbound total
    if len(data["out_place"][i]) > 0:
        if len(data["in_place"][i]) > 0:
            prob.addConstrs(
                (yUOi_p[j, t] - yUOi_m[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 (zi_p[j,
                    t] - zi_m[j,
                    t]) - gp.quicksum(
                    rC1i[a, j, t] for a in range(len(data["out_place"][i]))) + gp.quicksum(
                    rC2i[a, j, t] for a in range(len(data["in_place"][i]))) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')
        elif len(data["in_place"][i]) == 0:
            prob.addConstrs(
                (yUOi_p[j, t] - yUOi_m[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 (zi_p[j,
                    t] - zi_m[j,
                    t]) - gp.quicksum(
                    rC1i[a, j, t] for a in range(len(data["out_place"][i]))) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')
    elif len(data["out_place"][i]) == 0:
        if len(data["in_place"][i]) > 0:
            prob.addConstrs(
                (yUOi_p[j, t] - yUOi_m[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 (zi_p[j,
                    t] - zi_m[j,
                    t]) + gp.quicksum(
                    rC2i[a, j, t] for a in range(len(data["in_place"][i]))) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')
        elif len(data["in_place"][i]) == 0:
            prob.addConstrs(
                (yUOi_p[j, t] - yUOi_m[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 (zi_p[j,
                    t] - zi_m[j,
                    t]) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')

    # production capacity
    prob.addConstrs(
        (gp.quicksum(data["nu"][ii][j] * xCi[ii, j, t] for ii in range(data["K"])) <= data["C"][j][t] for j in
         range(data["M"]) for
         t in
         range(data["N"])),
        name='3i')

    # replacement bounds
    if len(data["out_place"][i]) > 0:
        '''
        prob.addConstrs(
            (rC1i[a, j, t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in
             range(len(data["out_place"][i]))),
            name='3j1')
        '''
        prob.addConstrs(
            (data["v0"][i][j] - rC1i[a, j, 0] >= 0 for j in range(data["M"]) for a in
             range(len(data["out_place"][i]))),
            name='3j2')
        prob.addConstrs(
            (vi[j, t - 1] - rC1i[a, j, t] >= 0 for j in range(data["M"]) for t in range(1, data["N"]) for a in
             range(len(data["out_place"][i]))),
            name='3j3')

    if len(data["in_place"][i]) > 0:
        prob.addConstrs(
            (data["v0"][data["cA"][data["in_place"][i][a]][0] - 1][j] - rC2i[a, j, 0] >= 0 for j in range(data["M"]) for
             a in
             range(len(data["in_place"][i]))), name='3j4')


    # outbound quantity bounds
    #prob.addConstrs((yUOi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])),
    #                name='3k1')
    prob.addConstrs(
        (data["v0"][i][j] - (yUOi_p[j, 0] - yUOi_m[j, 0]) >= 0 for j in range(data["M"])),
        name='3k2')
    prob.addConstrs(
        (vi[j, t - 1] - (yUOi_p[j, t] - yUOi_m[j, t]) >= 0 for j in range(data["M"]) for t in range(1, data["N"])),
        name='3k3')

    # non-negativity
    '''
    prob.addConstrs((yUIi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])), name='3l')
    prob.addConstrs((vi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])), name='3l2')
    prob.addConstrs((zi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])), name='3l3')
    prob.addConstrs(
        (xCi[ii, j, t] >= 0 for ii in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
        name='3m')
    prob.addConstrs((ui[t] >= 0 for t in range(data["N"])), name='3n')
    prob.addConstrs((si[ll, t] >= 0 for ll in range(data["L"]) for t in range(data["N"])), name='3o')
    '''
    vars = [ui, si, zi_p, zi_m, vi, yUIi, yUOi_p, yUOi_m, xCi]
    if len(data["out_place"][i]) > 0:
        vars.append(rC1i)
    if len(data["in_place"][i]) > 0:
        vars.append(rC2i)

    return prob, vars

def model_item_sub(data, var, i):
    global item_model, item_var

    start = time.time()

    prob = item_model[i]

    ui = item_var[i][0]
    si = item_var[i][1]
    zi_p = item_var[i][2]
    zi_m = item_var[i][3]
    vi = item_var[i][4]
    yUIi = item_var[i][5]
    yUOi_p = item_var[i][6]
    yUOi_m = item_var[i][7]
    xCi = item_var[i][8]
    if len(data["out_place"][i]) > 0:
        rC1i = item_var[i][9]
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][10]
    elif len(data["out_place"][i]) == 0:
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][9]

    prob.addConstrs(
        xCi[ii][j][t] == var["X_FIX"][ii][j][t] for ii in range(data["K"]) for j in range(data["M"]) for t in
        range(data["N"]) if var["Bool_X_FIX"][ii][j][t])
    if len(data["out_place"][i]) > 0:
        prob.addConstrs(rC1i[a][j][t] == var["R_FIX"][data["out_place"][i][a]][j][t] for a in range(len(data["out_place"][i])) for j in range(data["M"]) for t in
        range(data["N"]) if var["Bool_R_FIX"][data["out_place"][i][a]][j][t])
    if len(data["in_place"][i]) > 0:
        prob.addConstrs(rC2i[a][j][t] == var["R_FIX"][data["in_place"][i][a]][j][t] for a in range(len(data["in_place"][i])) for j in range(data["M"]) for t in
        range(data["N"]) if var["Bool_R_FIX"][data["in_place"][i][a]][j][t])

    time_model_var = time.time()

    ob = gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in range(data["N"])) + gp.quicksum(
        data["P"][i][t] * ui[t] for t in range(data["N"])) + data["mag"] * gp.quicksum(
        data["H"][i][j] * yUOi_m[j][t] for j in range(data["M"]) for t in
        range(data["N"])) + data["mag"] * gp.quicksum(
        data["P"][i][t] * zi_m[j][t] for j in range(data["M"]) for t in
        range(data["N"])) \
         - gp.quicksum(
        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
        range(data["M"]) for t in
        range(data["N"]))

    if len(data["out_place"][i]) > 0:
        ob -= gp.quicksum(
            var["Ksi"][data["out_place"][i][a]][j][t] * rC1i[a, j, t] for a in
            range(len(data["out_place"][i])) for
            j in
            range(data["M"]) for t in range(data["N"]))
    if len(data["in_place"][i]) > 0:
        ob -= gp.quicksum(
            var["Eta"][data["in_place"][i][a]][j][t] * rC2i[a, j, t] for a in
            range(len(data["in_place"][i])) for j
            in
            range(data["M"]) for t in range(data["N"]))

    # objective
    if var["ite"] == 0:
        pass
    else:
        ob += 1 / 2 * gp.quicksum(
                        var["rho_X"][ii][j][t] * (var["X"][ii][j][t] - xCi[ii, j, t]) ** 2 for ii in range(data["K"]) for j
                        in
                        range(data["M"]) for t in
                        range(data["N"]))

        if len(data["out_place"][i]) > 0:
            ob += 1 / 2 * gp.quicksum(
                        var["rho_R"][data["out_place"][i][a]][j][t] * (var["R"][data["out_place"][i][a]][j][t] - rC1i[a, j, t]) ** 2 for a in
                        range(len(data["out_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))

        if len(data["in_place"][i]) > 0:
            ob += 1 / 2 * gp.quicksum(
                        var["rho_R"][data["in_place"][i][a]][j][t] * (var["R"][data["in_place"][i][a]][j][t] - rC2i[a, j, t]) ** 2 for a in
                        range(len(data["in_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))

    prob.setObjective(ob)

    time_obj = time.time()

    prob.optimize()

    end = time.time()

    result = [ui.X, si.X, zi_p.X, zi_m.X, vi.X, yUIi.X, yUOi_p.X, yUOi_m.X, xCi.X]
    if len(data["out_place"][i]) > 0:
        result.append(rC1i.X)
    if len(data["in_place"][i]) > 0:
        result.append(rC2i.X)

    time_x = end - start
    time_x_model_var_setup = time_model_var - start
    time_x_set_obj = time_obj - time_model_var
    time_x_opt = end - time_obj

    result.extend([time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt])

    return result

def model_item_global(var):  # solve 2nd block of ADMM, which is to update the global variables
    '''
    K, M, N = var["X"].shape
    X_new = np.sum(var["XC"], axis=1) / K
    #X_new = np.floor(X_new)
    X_new = np.round(X_new)

    R_new = (var["RC1"] + var["RC2"]) / 2

    return [X_new, R_new]
    '''
    if var["ite"] <= var["LP_warm_start_ite"]:
        X_new = np.sum(var["XC"], axis=1) / data["K"]

        time_z_solver = 0
        time_z_heur = 0
    else:
        time_z_solver_start = time.time()
        #X_heur = np.zeros((data["K"], data["M"], data["N"]))
        X_solver = np.zeros((data["K"], data["M"], data["N"]))
        #X_round_down = np.floor(np.sum(var["XC"], axis=1) / data["K"])
        for plant_index in range(data["M"]):
            for time_index in range(data["N"]):
                # option: PSO/Genetic
                #X_heur[:, plant_index, time_index] = global_X_heur(var, plant_index, time_index, "Greedy")
                X_solver[:, plant_index, time_index] = global_X_heur(var, plant_index, time_index, "solver")
        time_z_solver_end = time.time()

        X_heur = np.zeros((data["K"], data["M"], data["N"]))
        for plant_index in range(data["M"]):
            for time_index in range(data["N"]):
                X_heur[:, plant_index, time_index] = global_X_heur(var, plant_index, time_index, "Greedy")
        time_z_heur_end = time.time()

        X_new = X_heur.copy()     # X_new = X_solver.copy()

        time_z_solver = time_z_solver_end - time_z_solver_start
        time_z_heur = time_z_heur_end - time_z_solver_start

        '''
        wr = "Iteration : " + str(var["ite"]) + "\nX_solver - X_round_down: " + str(
            X_solver - X_round_down) + "\nX_solver - X_heur : " + str(X_solver - X_heur) + "\n"
        wr_s = open('Z_Sub_Heuristic_Solver_Diff.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
        '''

    R_new = (var["RC1"] + var["RC2"]) / 2


    return [X_new, R_new, time_z_solver, time_z_heur]

def PSO_fitness(particle_swarm_x, penal_cap_vio, var, plant_index, time_index):
    particle_no = particle_swarm_x.shape[0]

    particle_obj = np.zeros(particle_no)
    for p in range(particle_no):
        sol_x = particle_swarm_x[p]

        obj_dual = np.sum(np.multiply(np.sum(var["Mu"][:, :, plant_index, time_index], axis=1), sol_x))
        obj_quadratic = 1 / 2 * np.sum(np.multiply(np.diag(var["rho_X"][:, plant_index, time_index]),
            np.power((var["XC"][:, :, plant_index, time_index] - sol_x.reshape(-1, 1)), 2)))
        obj_penalty_capacity_violation = penal_cap_vio * max(0, np.sum(np.multiply(data["nu"][:, plant_index], sol_x)) -
                                                             data["C"][plant_index][time_index])

        obj_plant_time = obj_dual + obj_quadratic + obj_penalty_capacity_violation

        particle_obj[p] = obj_plant_time

    return particle_obj

def Genetic_fitness(pop, var, plant_index, time_index):
    capacity_satisfy_flag = (np.matmul(pop, data["nu"][:, plant_index].reshape(-1, 1)) - data["C"][plant_index][
        time_index]).flatten()
    feasible_solution_idx = capacity_satisfy_flag <= 0

    fitness = np.zeros(pop.shape[0])

    def feasible_sol_fitness(fea_sol_set, var):
        sol_no = fea_sol_set.shape[0]
        sol_f = np.zeros(sol_no)
        for p in range(sol_no):
            sol_x = fea_sol_set[p]
            obj_dual = np.sum(np.multiply(np.sum(var["Mu"][:, :, plant_index, time_index], axis=1), sol_x))
            obj_quadratic = 1 / 2 * np.sum(np.multiply(np.diag(var["rho_X"][:, plant_index, time_index]),
                                                       np.power(
                                                           (var["XC"][:, :, plant_index, time_index] - sol_x.reshape(
                                                               -1, 1)), 2)))
            obj = obj_dual + obj_quadratic
            sol_f[p] = obj
        return sol_f

    if np.count_nonzero(feasible_solution_idx) > 0:
        fitness[feasible_solution_idx] = feasible_sol_fitness(pop[feasible_solution_idx], var)
        f_worst = min(fitness[feasible_solution_idx])
    else:
        f_worst = 0

    if np.count_nonzero(feasible_solution_idx) < pop.shape[0]:
        fitness[np.invert(feasible_solution_idx)] = f_worst + np.absolute(
            capacity_satisfy_flag[np.invert(feasible_solution_idx)])

    return fitness

def Genetic_tournament_select(pop, pop_f, tour_size):
    tour_pop_index = np.random.choice(pop.shape[0], tour_size, replace=False)
    idx = np.argmin(pop_f[tour_pop_index])

    return tour_pop_index[idx]     # return index of chosen individual

def global_X_heur(var, plant_index, time_index, option):
    var_ub = np.floor(np.divide(data["C"][plant_index][time_index], data["nu"][:, plant_index]))
    var_lb = np.zeros(data["K"])

    if option == "solver":
        prob = gp.Model("global_plant_time " + str((plant_index, time_index)))

        x = prob.addMVar(data["K"], vtype=GRB.INTEGER)

        prob.addConstr(
            gp.quicksum(data["nu"][i][plant_index] * x[i] for i in range(data["K"])) <= data["C"][plant_index][time_index])

        prob.setObjective(gp.quicksum(
            var["Mu"][i, ii, plant_index, time_index] * x[i] for i in range(data["K"]) for ii
            in range(data["K"])) + 1 / 2 * gp.quicksum(
            var["rho_X"][i, plant_index, time_index] * (
                        x[i] - var["XC"][i, ii, plant_index, time_index]) ** 2 for i in
            range(data["K"]) for ii in
            range(data["K"])))

        prob.optimize()

        return x.X

    if option == "PSO":
        # parameters
        par_no = 50  # particle number
        PSO_max_ite = 100   #100
        cog_learn_rate = 2.05  # cognitive learning rate
        social_learn_rate = 2.05   # social learning rates
        v_max = (var_ub - var_lb) * 0.2
        good_particle_tol = 0.2
        penal_cap_vio = 1   #var["rho_X"][0, 0, 0] * 1000
        Henon_map_a = 1.4
        Henon_map_b = 0.3

        t = cog_learn_rate + social_learn_rate
        constr_coef = 2 / abs(2 - t - math.sqrt(t * t - 4 * t))   # constriction coefficient

        # random initialization of particles
        position = np.round(np.random.uniform(low=var_lb, high=var_ub, size=(par_no, data["K"])))
        position[0] = np.floor(np.sum(var["XC"][:, :, plant_index, time_index], axis=1) / data["K"])   # use round down solution(feasible) as an initial point

        velocity = np.random.uniform(low=-v_max, high=v_max, size=(par_no, data["K"]))

        # current_fitness
        current_fitness = PSO_fitness(position, penal_cap_vio, var, plant_index, time_index)

        # initialization of historical best position for each particle and the whole swarm
        p_best = position.copy()
        p_best_fitness = current_fitness.copy()

        g_best_fitness = min(current_fitness)
        g_best_index = np.argmin(current_fitness)
        g_best = position[g_best_index]

        # Henon_mapping
        Henon_map_h = np.random.uniform(0, 1, par_no)
        Henon_map_H = np.random.uniform(0, 1, par_no)

        # PSO
        for ite in range(1, PSO_max_ite + 1):
            # Henon's map
            new_h = 1 - Henon_map_a * np.power(Henon_map_h, 2) + Henon_map_H
            Henon_map_H = Henon_map_b * Henon_map_h
            Henon_map_h = new_h
            # normalization to [0,1]
            Henon_map_h = (Henon_map_h - np.min(Henon_map_h)) / (np.max(Henon_map_h) - np.min(Henon_map_h))
            Henon_map_H = (Henon_map_H - np.min(Henon_map_H)) / (np.max(Henon_map_H) - np.min(Henon_map_H))

            # uniform and Gaussian random number
            Uniform_rand_num = random.uniform(0, 1)
            Gaussian_rand_num = np.random.normal(0, 1, par_no)
            # normalization to [0,1]
            Gaussian_rand_num = (Gaussian_rand_num - np.min(Gaussian_rand_num)) / (
                        np.max(Gaussian_rand_num) - np.min(Gaussian_rand_num))

            # update velocity
            if ite == 1:
                velocity += cog_learn_rate * np.matmul(np.diag(Gaussian_rand_num),
                                                       p_best - position) + social_learn_rate * Uniform_rand_num * (
                                        g_best - position)

                velocity *= constr_coef
                velocity = np.maximum(-v_max, np.minimum(velocity, v_max))   # restrict jump step
            elif ite > 1:
                good_particle_metric = 1 - (max(current_fitness) - current_fitness) / (max(current_fitness) - min(current_fitness))
                good_particle_index = good_particle_metric < good_particle_tol

                if np.count_nonzero(good_particle_index) > 0:
                    # good particle : chaotic sequence based on Henon map in velocity updating
                    velocity[good_particle_index] += cog_learn_rate * np.matmul(
                        np.diag(Henon_map_h[good_particle_index]),
                        p_best[good_particle_index] - position[
                            good_particle_index]) + social_learn_rate * np.matmul(
                        np.diag(Henon_map_H[good_particle_index]), g_best - position[good_particle_index])
                if np.count_nonzero(good_particle_index) < par_no:
                    velocity[np.invert(good_particle_index)] += cog_learn_rate * np.matmul(np.diag(Gaussian_rand_num[np.invert(
                                                                                               good_particle_index)]),
                                                                                           p_best[np.invert(
                                                                                               good_particle_index)] -
                                                                                           position[np.invert(
                                                                                               good_particle_index)]) + social_learn_rate * Uniform_rand_num * (
                                                                        g_best - position[
                                                                    np.invert(good_particle_index)])

                # restrict jump step
                velocity *= constr_coef
                velocity = np.maximum(-v_max, np.minimum(velocity, v_max))

            # update position
            position += velocity
            position = np.round(position)   # round to the nearest integer
            position = np.maximum(var_lb, np.minimum(position, var_ub))    # bound variables

            # update p_best and g_best
            current_fitness = PSO_fitness(position, penal_cap_vio, var, plant_index, time_index)
            p_best_update_index = current_fitness < p_best_fitness
            p_best[p_best_update_index] = position[p_best_update_index]
            p_best_fitness[p_best_update_index] = current_fitness[p_best_update_index]

            if min(p_best_fitness) < g_best_fitness:
                g_best_fitness = min(p_best_fitness)
                g_best_index = np.argmin(p_best_fitness)
                g_best = p_best[g_best_index]

        return g_best

    if option == "Genetic":
        # parameters
        pop_no = 50  # population
        Genetic_max_ite = 100
        tour_size = 3    # tournament size
        location_a = 0    # (crossover) location parameter
        scaling_b = 0.35      # (crossover) scaling parameter
        power_index = 4      # (mutation) index of power mutation
        prob_c = 0.8      # Crossover probability
        prob_m = 0.005     # mutation probability

        # random initialization of particles
        population = np.round(np.random.uniform(low=var_lb, high=var_ub, size=(pop_no, data["K"])))
        population[0] = np.floor(np.sum(var["XC"][:, :, plant_index, time_index], axis=1) / data[
            "K"])  # use round down solution(feasible) as an initial point
        # population fitness
        population_fitness = Genetic_fitness(population, var, plant_index, time_index)

        new_population = np.zeros((pop_no, data["K"]))

        # laplace random number
        Laplace_rand_beta = np.zeros(data["K"])

        for ite in range(Genetic_max_ite):
            for ind in range(pop_no // 2):
                # tournament selection on initial (old) population to make mating pool
                parent_1_idx = Genetic_tournament_select(population, population_fitness, tour_size)
                while True:
                    parent_2_idx = Genetic_tournament_select(population, population_fitness, tour_size)
                    if abs(parent_1_idx - parent_2_idx) > 1e-2:    # if parent 1 is not equal to parent 2
                        break
                parent_1 = population[parent_1_idx]
                parent_2 = population[parent_2_idx]

                if random.random() <= prob_c:
                    # laplace crossover
                    uniform_rand_u = np.random.uniform(0, 1, data["K"])
                    uniform_rand_r = np.random.uniform(0, 1, data["K"])

                    Laplace_rand_beta[uniform_rand_r <= 0.5] = location_a - scaling_b * np.log(
                        uniform_rand_u[uniform_rand_r <= 0.5])
                    Laplace_rand_beta[uniform_rand_r > 0.5] = location_a + scaling_b * np.log(
                        uniform_rand_u[uniform_rand_r > 0.5])

                    new_population[ind * 2] = parent_1 + np.multiply(Laplace_rand_beta,
                                                                     np.absolute(parent_1 - parent_2))
                    new_population[ind * 2 + 1] = parent_2 + np.multiply(Laplace_rand_beta,
                                                                         np.absolute(parent_1 - parent_2))
                else:
                    new_population[ind * 2] = parent_1
                    new_population[ind * 2 + 1] = parent_2

            # power mutation and integer restriction
            for ind in range(pop_no):
                individual = new_population[ind]
                if random.random() <= prob_m:
                    individual_after_mutation = np.zeros(data["K"])

                    s = random.random() ** power_index
                    t = np.divide(individual - var_lb, var_ub - individual + 1e-5)
                    r = random.random()

                    individual_after_mutation[t < r] = individual[t < r] - s * (individual[t < r] - var_lb[t < r])
                    individual_after_mutation[t >= r] = individual[t >= r] + s * (var_ub[t >= r] - individual[t >= r])

                    individual = individual_after_mutation

                # check bound
                indv_smaller_lb = individual < var_lb
                indv_greater_ub = individual > var_ub
                if np.count_nonzero(indv_smaller_lb) > 0:
                    individual[indv_smaller_lb] = var_lb[indv_smaller_lb]
                if np.count_nonzero(indv_greater_ub) > 0:
                    individual[indv_greater_ub] = var_ub[indv_greater_ub]

                # integer restrictions
                int_restr_rand = np.random.uniform(0, 1, data["K"])
                individual[int_restr_rand <= 0.5] = np.floor(individual[int_restr_rand <= 0.5])
                individual[int_restr_rand > 0.5] = np.ceil(individual[int_restr_rand > 0.5])

                new_population[ind] = individual

            # merge parent and offspring together to select new population
            parent_offspring = np.zeros((pop_no * 2, data["K"]))
            parent_offspring[range(0, pop_no)] = population
            parent_offspring[range(pop_no, pop_no * 2)] = new_population

            parent_offspring_fitness = Genetic_fitness(parent_offspring, var, plant_index, time_index)
            best_indiv_index = np.argpartition(parent_offspring_fitness, pop_no)
            best_indiv_index = best_indiv_index[:pop_no]

            population = parent_offspring[best_indiv_index]
            population_fitness = parent_offspring_fitness[best_indiv_index]

        idx = np.argmin(population_fitness)
        best_individual = population[idx]

        return best_individual

    if option == "Greedy":
        x_round_down = np.floor(np.sum(var["XC"][:, :, plant_index, time_index], axis=1) / data["K"])

        delta_obj = x_round_down * var["rho_X"][0, 0, 0] * data["K"] + np.sum(var["Mu"][:, :, plant_index, time_index],
                                                                              axis=1) - \
                    var["rho_X"][0, 0, 0] * np.sum(var["XC"][:, :, plant_index, time_index], axis=1) + var["rho_X"][
                        0, 0, 0] * data["K"] / 2
        delta_capacity = data["nu"][:, plant_index]

        marginal_benefit = np.divide(delta_obj, delta_capacity)

        present_cap = np.sum(np.multiply(data["nu"][:, plant_index], x_round_down))
        cap_one_more_unit = present_cap + data["nu"][:, plant_index]

        index1 = x_round_down < var_ub
        index2 = marginal_benefit < 0
        index3 = cap_one_more_unit <= data["C"][plant_index, time_index]
        cand_idx = np.logical_and(np.logical_and(index1, index2), index3)

        x_H = x_round_down.copy()
        while True:
            if np.count_nonzero(cand_idx) == 0:
                break
            elif np.count_nonzero(cand_idx) > 0:
                min_marg_benefit_idx = np.nonzero(cand_idx)[0][np.argmin(marginal_benefit[cand_idx])]
                x_H[min_marg_benefit_idx] += 1

                delta_obj[min_marg_benefit_idx] = x_H[min_marg_benefit_idx] * var["rho_X"][0, 0, 0] * data[
                    "K"] + np.sum(
                    var["Mu"][min_marg_benefit_idx, :, plant_index, time_index]) - var["rho_X"][0, 0, 0] * np.sum(
                    var["XC"][min_marg_benefit_idx, :, plant_index, time_index]) + var["rho_X"][0, 0, 0] * data[
                                                      "K"] / 2
                marginal_benefit[min_marg_benefit_idx] = delta_obj[min_marg_benefit_idx] / data["nu"][
                    min_marg_benefit_idx, plant_index]

                present_cap += data["nu"][min_marg_benefit_idx, plant_index]
                cap_one_more_unit += data["nu"][min_marg_benefit_idx, plant_index]

                if (x_H[min_marg_benefit_idx] == var_ub[min_marg_benefit_idx] or marginal_benefit[
                    min_marg_benefit_idx] >= 0 or cap_one_more_unit[min_marg_benefit_idx] > data["C"][
                    plant_index, time_index]):
                    cand_idx[min_marg_benefit_idx] = False

        return x_H

def LB_X(data, var):
    prob = gp.Model("LB_X")

    x = prob.addMVar((data["K"], data["M"], data["N"]), vtype=GRB.INTEGER)

    prob.addConstrs(gp.quicksum(data["nu"][i][j] * x[i][j][t] for i in range(data["K"])) <= data["C"][j][t] for j in range(data["M"]) for t in range(data["N"]))

    prob.setObjective(gp.quicksum(
        var["Mu"][i][ii][j][t] * x[i][j][t] for i in range(data["K"]) for ii in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])))

    prob.optimize()

    return prob.ObjVal

def LB_item(data, var, i):
    global item_model, item_var

    prob = item_model[i]

    ui = item_var[i][0]
    zi_m = item_var[i][3]
    vi = item_var[i][4]
    yUOi_m = item_var[i][7]
    xCi = item_var[i][8]
    if len(data["out_place"][i]) > 0:
        rC1i = item_var[i][9]
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][10]
    elif len(data["out_place"][i]) == 0:
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][9]

    prob.addConstrs(vi[j, t] <= var["V"][i][j][t] for j in range(data["M"]) for t in range(data["N"]))
    if len(data["in_place"][i]) > 0:
        prob.addConstrs(
            var["V"][data["cA"][data["in_place"][i][a]][0] - 1][j][t - 1] - rC2i[a, j, t] >= 0 for j in range(data["M"]) for
            a in
            range(len(data["in_place"][i])) for t in range(1, data["N"]))

    ob = gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in range(data["N"])) + gp.quicksum(
        data["P"][i][t] * ui[t] for t in range(data["N"])) + data["mag"] * gp.quicksum(
        data["H"][i][j] * yUOi_m[j][t] for j in range(data["M"]) for t in
        range(data["N"])) + data["mag"] * gp.quicksum(
        data["P"][i][t] * zi_m[j][t] for j in range(data["M"]) for t in
        range(data["N"])) \
         - gp.quicksum(
        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
        range(data["M"]) for t in
        range(data["N"]))

    if len(data["out_place"][i]) > 0:
        ob -= gp.quicksum(
            var["Ksi"][data["out_place"][i][a]][j][t] * rC1i[a, j, t] for a in
            range(len(data["out_place"][i])) for
            j in
            range(data["M"]) for t in range(data["N"]))
    if len(data["in_place"][i]) > 0:
        ob -= gp.quicksum(
            var["Eta"][data["in_place"][i][a]][j][t] * rC2i[a, j, t] for a in
            range(len(data["in_place"][i])) for j
            in
            range(data["M"]) for t in range(data["N"]))

    prob.setObjective(ob)

    prob.optimize()

    if prob.Status == GRB.OPTIMAL:
        wr = "Subproblem item " + str(i) + ': Optimal solution found!\n'
        wr_s = open('../new_version(1.25)/LB_problem_detective.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
        '''
        result = [prob.ObjVal, vi.X, xCi.X]
        if len(out_place[i]) > 0:
            result.append(rC1i.X)
        if len(in_place[i]) > 0:
            result.append(rC2i.X)
        '''
        return prob.ObjVal
    elif prob.Status == GRB.INFEASIBLE:
        #print('Model is infeasible')
        wr = "Subproblem item " + str(i) + ': Infeasible!\n'
        wr_s = open('../new_version(1.25)/LB_problem_detective.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
        #sys.exit(0)
    elif prob.Status == GRB.UNBOUNDED:
        #print('Model is unbounded')
        wr = "Subproblem item " + str(i) + ': Unbounded!\n'
        wr_s = open('../new_version(1.25)/LB_problem_detective.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
        #sys.exit(0)

def UB_item(data, ref_glob, i):
    global item_model, item_var

    incumb_X = ref_glob[0]
    incumb_R = ref_glob[1]

    prob = item_model[i]

    ui = item_var[i][0]
    si = item_var[i][1]
    zi_p = item_var[i][2]
    zi_m = item_var[i][3]
    vi = item_var[i][4]
    yUIi = item_var[i][5]
    yUOi_p = item_var[i][6]
    yUOi_m = item_var[i][7]
    xCi = item_var[i][8]
    if len(data["out_place"][i]) > 0:
        rC1i = item_var[i][9]
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][10]
    elif len(data["out_place"][i]) == 0:
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][9]

    #yUOi.lb = -GRB.INFINITY
    #zi.lb = -GRB.INFINITY
    #prob.setAttr("lb", yUOi, -GRB.INFINITY)

    prob.addConstrs(xCi[ii][j][t] == incumb_X[ii][j][t]
                    for ii in range(data["K"]) for j in range(data["M"]) for t in range(data["N"]))

    if len(data["out_place"][i]) > 0:
        prob.addConstrs(rC1i[a][j][t] == incumb_R[data["out_place"][i][a]][j][t] for a in
                        range(len(data["out_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))

    if len(data["in_place"][i]) > 0:
        prob.addConstrs(rC2i[a][j][t] == incumb_R[data["in_place"][i][a]][j][t] for a in
                        range(len(data["in_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))

    prob.setObjective(
        gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in range(data["N"])) + gp.quicksum(
            data["P"][i][t] * ui[t] for t in range(data["N"])) + data["mag"] * gp.quicksum(
            data["H"][i][j] * yUOi_m[j][t] for j in range(data["M"]) for t in
            range(data["N"])) + data["mag"] * gp.quicksum(
            data["P"][i][t] * zi_m[j][t] for j in range(data["M"]) for t in
            range(data["N"]))
    )

    prob.optimize()

    if prob.Status == GRB.OPTIMAL:
        wr = "Subproblem item " + str(i) + ': Optimal solution found!\n'
        wr_s = open('UB_problem_detective.txt', 'a')
        wr_s.write(wr)
        wr_s.close()

        return prob.ObjVal

        #result = [ui.X, si.X, zi_p.X, zi_m.X, vi.X, yUIi.X, yUOi_p.X, yUOi_m.X]
        #return [prob.ObjVal, result]
    elif prob.Status == GRB.INFEASIBLE:
        #print('Model is infeasible')
        wr = "Subproblem item " + str(i) + ': Infeasible!\n'
        wr_s = open('../new_version(1.25)/UB_problem_detective.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
        #sys.exit(0)
    elif prob.Status == GRB.UNBOUNDED:
        #print('Model is unbounded')
        wr = "Subproblem item " + str(i) + ': Unbounded!\n'
        wr_s = open('../new_version(1.25)/UB_problem_detective.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
        #sys.exit(0)

def comp_obj(data, var):  # Compute the objective corresponding to a solution
    V_temp = np.sum(var["V"], axis=2)
    YUO_M_temp = np.sum(var["YUO_M"], axis=2)
    Z_M_temp = np.sum(var["Z_M"], axis=1)

    ob = np.sum(np.multiply(data["P"], var["U"])) + np.sum(np.multiply(data["H"], V_temp)) + data["mag"] * np.sum(
        np.multiply(data["P"], Z_M_temp)) + data["mag"] * np.sum(np.multiply(data["H"], YUO_M_temp))

    return ob

def go(data, var):  # solve items' problems in parallel
    # p = mp.Pool(processes=min(mp.cpu_count(), data["K"]))
    # p.map(self, range(data["K"]))
    '''
    procs = []
    # instantiating process with arguments
    for i in range(data["K"]):
        # print(name)
        if key == "setup":
            proc = Process(target=model_var_setup, args=(data, i))
        elif key == "setObj":
            proc = Process(target=model_item_setObj, args=(i, data, variables, parameters,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    '''
    #with Pool(processes=20) as pool:
    with Pool(processes=min(mp.cpu_count(), data["K"])) as pool:
        time_x = []
        time_x_model_var_setup = []
        time_x_set_obj = []
        time_x_opt = []

        items = [(data, var, i) for i in list(range(data['K']))]
        start = time.time()
        results = pool.starmap(model_item_sub, items)
        end = time.time()
        '''
        results = []
        time_x = []
        for i in range(data['K']):
            start = time.time()
            result = model_item_sub(data, i)
            end = time.time()

            time_x.append(end - start)
            results.append(result)
        '''

        # variables["U"][i] = ui.X
        U_new = var["U"].copy()
        S_new = var["S"].copy()
        Z_P_new = var["Z_P"].copy()
        Z_M_new = var["Z_M"].copy()
        V_new = var["V"].copy()
        YUI_new = var["YUI"].copy()
        YUO_P_new = var["YUO_P"].copy()
        YUO_M_new = var["YUO_M"].copy()
        XC_new = var["XC"].copy()
        RC1_new = var["RC1"].copy()
        RC2_new = var["RC2"].copy()

        for i in range(data['K']):
            U_new[i] = results[i][0]
            S_new[i] = results[i][1]
            Z_P_new[i] = results[i][2]
            Z_M_new[i] = results[i][3]
            V_new[i] = results[i][4]
            YUI_new[i] = results[i][5]
            YUO_P_new[i] = results[i][6]
            YUO_M_new[i] = results[i][7]
            XC_new[:, i, :, :] = results[i][8]
            if len(data["out_place"][i]) > 0:
                RC1_new[data["out_place"][i]] = results[i][9]
                if len(data["in_place"][i]) > 0:
                    RC2_new[data["in_place"][i]] = results[i][10]
            elif len(data["out_place"][i]) == 0:
                if len(data["in_place"][i]) > 0:
                    RC2_new[data["in_place"][i]] = results[i][9]

            time_x.append(results[i][-4])
            time_x_model_var_setup.append(results[i][-3])
            time_x_set_obj.append(results[i][-2])
            time_x_opt.append(results[i][-1])

        parallel_time = end - start

        #return time_x, sum(time_x)
        return [[U_new, S_new, Z_P_new, Z_M_new, V_new, YUI_new, YUO_P_new, YUO_M_new, XC_new, RC1_new, RC2_new], [time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt, parallel_time]]

        #return time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt, parallel_time

        # async_results = [pool.apply_async(model_item_setObj, args=(data, i)) for i in range(data['K'])]

def UpperBounding(data, var, choice):
    with Pool(processes=min(mp.cpu_count(), data["K"])) as pool:
        if choice == "average":  # x_hat取copy的平均
            incumb_X = var["X"].copy()
            incumb_R = var["R"].copy()

            incumb_pair = [(incumb_X, incumb_R)]

        elif choice == "slam_max":  # slam heuristic x_hat取copy的最大或最小
            incumb_X = np.amax(var["XC"], axis=1)
            incumb_R = np.maximum(var["RC1"], var["RC2"])

            incumb_pair = [(incumb_X, incumb_R)]
        elif choice == "slam_min":
            incumb_X = np.amin(var["XC"], axis=1)
            incumb_R = np.minimum(var["RC1"], var["RC2"])

            incumb_pair = [(incumb_X, incumb_R)]

        else:  # try over other scenario's solution, choice = "runover_sec" or "part_sce_inorder" or "part_sce_shuffle"
            incumb_pair = []

            for ref_s in range(data["K"]):
                incumb_X = var["XC"][:, ref_s, :, :]
                # incumb_X = np.round(var["XC"][:, ref_sce, :, :])
                # R除了ref_sce作为出端或者入端的边按其算出来的边赋值，其他取平均
                incumb_R = var["R"].copy()
                if len(data["out_place"][ref_s]) > 0:
                    incumb_R[data["out_place"][ref_s]] = var["RC1"][data["out_place"][ref_s]]
                if len(data["in_place"][ref_s]) > 0:
                    incumb_R[data["in_place"][ref_s]] = var["RC2"][data["in_place"][ref_s]]

                incumb_pair.append((incumb_X, incumb_R))

            if choice == "part_sce_shuffle":
                random.shuffle(incumb_pair)

        ref_UB = []
        for candidate in incumb_pair:
            items = [(data, candidate, i) for i in list(range(data["K"]))]

            results = pool.starmap(UB_item, items)
            probs_state = any(item in results for item in results if item == None)
            if bool(probs_state) == False:  # 子问题均有finite optimal solution则可生成一个有效的UB
                bound = sum(results)
                ref_UB.append(bound)

                if choice == "part_sce_inorder" or "part_sce_shuffle":
                    break

        # incumbent run over all scenario's solution and take best ub
        # or loop in order or reshuffle until find a feasible bound then break
        if len(ref_UB) > 0:
            return min(ref_UB)
            # 否则，若对每个candidate都有子问题infeasible或unbounded，无返回值

def LowerBounding(data, var):
    with Pool(processes=min(mp.cpu_count(), data["K"])) as pool:
        items = [(data, var, i) for i in list(range(data["K"]))]

        # bound = LB(var, data)
        result_X = LB_X(data, var)

        results = pool.starmap(LB_item, items)
        probs_state = any(item in results for item in results if item == None)
        if bool(probs_state) == False:  # 子问题均有finite optimal solution
            '''
            bound = 0
            SubG_V = np.zeros((K, M, N))
            SubG_R = np.zeros((A, M, N))
            SubG_XC = np.zeros((K, K, M, N))
            SubG_RC1 = np.zeros((A, M, N))
            SubG_RC2 = np.zeros((A, M, N))
            for i in range(K):
                bound += results[i][0]
                SubG_V[i] = results[i][1]
                SubG_XC[:, i, :, :] = results[i][2]
                if len(out_place[i]) > 0:
                    SubG_RC1[out_place[i]] = results[i][3]
                    if len(in_place[i]) > 0:
                        SubG_RC2[in_place[i]] = results[i][4]
                elif len(out_place[i]) == 0:
                    if len(in_place[i]) > 0:
                        SubG_RC2[in_place[i]] = results[i][3]
            for a in range(A):
                r_value = (Ksi[a] + Eta[a] >= 0)
                for j in range(M):
                    for t in range(N):
                        if r_value[j][t] == False:
                            SubG_R[a][j][t] = SubG_V[data_cA[a][0] - 1][j][t]
            '''

            bound = sum(results) + result_X
            return bound
        # 否则，若有子问题infeasible或unbounded，无返回值

def remaining_solver(data, var):
    prob = gp.Model("remaining_extensive")

    # variable
    v = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    u = prob.addMVar((data["K"], data["N"]))  # (i,t)
    z_p = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    z_m = prob.addMVar((data["K"], data["M"], data["N"]))
    yUI = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    yUO_p = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    yUO_m = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    x = prob.addMVar((data["K"], data["M"], data["N"]), vtype=GRB.INTEGER)  # (i,j,t)
    r = prob.addMVar((data["A"], data["M"], data["N"]))  # (a,j,t)
    s = prob.addMVar((data["K"], data["L"], data["N"]))  # (i,l,t)
    #w = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.INTEGER)  # (i,j,t)
    #xUb = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.BINARY)  # (i,j,t)

    # constraint
    prob.addConstrs(
        x[i][j][t] == var["X_FIX"][i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in
        range(data["N"]) if var["Bool_X_FIX"][i][j][t])
    prob.addConstrs(
            r[a][j][t] == var["R_FIX"][a][j][t] for a in range(data["A"]) for j
            in range(data["M"]) for t in
            range(data["N"]) if var["Bool_R_FIX"][a][j][t])

    # (1b)-(1c)
    prob.addConstrs((u[i][0] + gp.quicksum(z_p[i][j][0] - z_m[i][j][0] for j in range(data["M"])) == data["D"][i][0] for i in range(data["K"])),
                    name='1b-1c1')
    prob.addConstrs(
        (u[i][t] - u[i][t - 1] + gp.quicksum(z_p[i][j][t] - z_m[i][j][t] for j in range(data["M"])) == data["D"][i][t] for i in range(data["K"])
         for t in range(1, data["N"])), name='1c2')

    # (1d)
    # prob.addConstrs(
    #    (gp.quicksum(u[i][tt] / D[i][tt] for i in range(K) for tt in range(t + 1) if D[i][tt] != 0) <= epsilonUD[t] for
    #     t in range(N)), name='1d')

    # (1e)
    prob.addConstrs((v[i][j][0] - yUI[i][j][0] + yUO_p[i][j][0] - yUO_m[i][j][0] == data["v0"][i][j] for i in range(data["K"])
                     for j in range(data["M"])), name='1e1')
    prob.addConstrs(
        (v[i][j][t] - v[i][j][t - 1] - yUI[i][j][t] + yUO_p[i][j][t] - yUO_m[i][j][t] == 0 for i in range(data["K"]) for j in range(data["M"])
         for t in range(1, data["N"])), name='1e2')

    # (1f)
    # prob.addConstrs((gp.quicksum(
    #     360 / (12 * N) * v[i][j][tt] - epsilonUI * yUO[i][j][tt] for j in range(M) for tt in range(t + 1)) <= 0
    #     for i in range(K) for t in range(N)), name='1f')

    # (1g)
    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - int(data["dtUL"][i][l])] for l in range(data["L"]) if data["cL"][l][1] == j + 1
            and t - data["dtUL"][i][l] >= 0) - x[i][j][t - int(data["dtUP"][i][j])] == data["Q"][i][j][t] for i in range(data["K"])
         for j in range(data["M"]) for t in range(data["N"]) if t >= data["dtUP"][i][j]),
        name='1g1')
    prob.addConstrs(
        (yUI[i][j][t] - gp.quicksum(
            s[i][l][t - data["dtUL"][i][l]] for l in range(data["L"]) if data["cL"][l][1] == j + 1 and t - data["dtUL"][i][l] >= 0) ==
         data["Q"][i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"]) if t < data["dtUP"][i][j]),
        name='1g2')

    # (1h)
    prob.addConstrs((yUO_p[i][j][t] - yUO_m[i][j][t] - gp.quicksum(s[i][l][t] for l in range(data["L"]) if data["cL"][l][0] == j + 1) -
                     gp.quicksum(data["q"][e] * x[data["cE"][e][1] - 1][j][t] for e in range(data["E"]) if data["cE"][e][0] == i + 1)
                     - (z_p[i][j][t] - z_m[i][j][t]) - gp.quicksum(
        r[a][j][t] for a in range(data["A"]) if data["cA"][a][0] == i + 1) + gp.quicksum(
        r[a][j][t] for a in range(data["A"]) if data["cA"][a][1] == i + 1) == 0 for i in range(data["K"])
                     for j in range(data["M"]) for t in range(data["N"])),
                    name='1h')

    # (1i)
    prob.addConstrs(
        (gp.quicksum(data["nu"][i][j] * x[i][j][t] for i in range(data["K"])) <= data["C"][j][t] for j in range(data["M"])
         for t in range(data["N"])), name='1i')

    # (1j)
    #prob.addConstrs((r[a][j][t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in range(data["A"])), name='1j1')
    prob.addConstrs(
        (data["v0"][i][j] - r[a][j][0] >= 0 for j in range(data["M"]) for i in
         range(data["K"])
         for a in range(data["A"]) if data["cA"][a][0] == i + 1),
        name='1j2')
    prob.addConstrs(
        (v[i][j][t - 1] - r[a][j][t] >= 0 for j in range(data["M"]) for t in range(1, data["N"]) for i in range(data["K"])
         for a in range(data["A"]) if data["cA"][a][0] == i + 1),
        name='1j3')

    # (1k)
    #prob.addConstrs((yUO_p[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
    #                name='1k1')
    #prob.addConstrs((yUO_m[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
    #                name='1k1')
    prob.addConstrs(
        (data["v0"][i][j] - (yUO_p[i][j][0] - yUO_m[i][j][0]) >= 0 for i in range(data["K"]) for j in range(data["M"])),
        name='3k2')
    prob.addConstrs(
        (v[i][j][t - 1] - (yUO_p[i][j][t] - yUO_m[i][j][t]) >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(1, data["N"])),
        name='1k2')

    # (1l)
    #prob.addConstrs((x[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l1')
    #prob.addConstrs((yUI[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l2')
    #prob.addConstrs((v[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l3')
    #prob.addConstrs((z_p[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l4')
    #prob.addConstrs((z_m[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
    #                name='1l4')

    # (1m)
    #prob.addConstrs((u[i][t] >= 0 for i in range(data["K"]) for t in range(data["N"])), name='1m')

    # (1n)
    #prob.addConstrs((s[i][l][t] >= 0 for i in range(data["K"]) for l in range(data["L"]) for t in range(data["N"])), name='1n')
    '''
    # (1o)
    prob.addConstrs((x[i][j][t] - gbv.m[i] * w[i][j][t] == 0 for i in range(gbv.K)
                     for j in range(gbv.M)
                     for t in range(gbv.N)), name='1o')

    # (1p)
    prob.addConstrs((xUb[i][j][t] * gbv.wLB[i][j] - w[i][j][t] <= 0 for i in range(gbv.K) for j in range(gbv.M) for t in
                     range(gbv.N)),
                    name='1p1')
    prob.addConstrs((xUb[i][j][t] * gbv.wUB[i][j] - w[i][j][t] >= 0 for i in range(gbv.K) for j in range(gbv.M) for t in
                     range(gbv.N)),
                    name='1p2')
    '''
    # objective
    prob.setObjective(gp.quicksum(
        data["H"][i][j] * v[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"]))
                      + gp.quicksum(data["P"][i][t] * u[i][t] for i in range(data["K"]) for t in range(data["N"]))
                      + gp.quicksum(
        data["mag"] * data["H"][i][j] * yUO_m[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in
        range(data["N"]))
                      + gp.quicksum(
        data["mag"] * data["P"][i][t] * z_m[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in
        range(data["N"])))
    #time_end1 = time.time()
    #time_e_model = time_end1 - time_start1

    #time_start2 = time.time()
    prob.optimize()
    #time_end2 = time.time()
    #time_e_op = time_end2 - time_start2

    #gbv.W_E = w.X
    #gbv.XUb_E = xUb.X

    X_sol = np.reshape(x.X, (-1, 1))
    S_sol = np.reshape(s.X, (-1, 1))
    Z_P_sol = np.reshape(z_p.X, (-1, 1))
    Z_M_sol = np.reshape(z_m.X, (-1, 1))
    R_sol = np.reshape(r.X, (-1, 1))
    V_sol = np.reshape(v.X, (-1, 1))
    U_sol = np.reshape(u.X, (-1, 1))
    YUI_sol = np.reshape(yUI.X, (-1, 1))
    YUO_P_sol = np.reshape(yUO_p.X, (-1, 1))
    YUO_M_sol = np.reshape(yUO_m.X, (-1, 1))

    Sol = np.concatenate((X_sol, S_sol, Z_P_sol, Z_M_sol, R_sol, V_sol, U_sol, YUI_sol, YUO_P_sol, YUO_M_sol), axis=0)

    return prob.objVal, Sol
    #x.X, s.X, z.X, r.X, v.X, u.X, yUI.X, yUO.X

def ADMM(data, var, solver_ob, solver_sol):
    # generate instance of methods of item-based decomposition algorithm
    # dim = item_dim()

    # 为每个item首先建立一个model存起来
    # lists = list(range(data["K"]))
    # tem = pool.map(dim.model_var_setup, lists)
    # tem = pool.map(dim.model_var_setup, lists)
    '''
    dim.key = "setup"
    tem = dim.go()
    for i in lists:
        item_model[i] = tem[i][0]  # tem[i][0]为item i(i=0,...,K-1)的model
        item_var[i] = tem[i][1:]  # tem[i][1:]为item i(i=0,...,K-1)的变量的列表
    '''

    objs = []
    rel_obj_err = []
    rel_sol_err = []
    nz = []   # iteration index

    # UB and LB
    bound_index = []
    UB = math.inf
    UB_OBJ = []
    LB = -math.inf
    LB_OBJ = []

    X_stable_ite = np.zeros((data["K"], data["M"], data["N"]))  # production
    R_stable_ite = np.zeros((data["A"], data["M"], data["N"]))  # replacement
    X_stable_value = np.zeros((data["K"], data["M"], data["N"]))  # production
    R_stable_value = np.zeros((data["A"], data["M"], data["N"]))  # replacement
    # set up model and variables
    #go("setup", data, lock)
    #p = mp.Pool(processes=min(mp.cpu_count(), data["K"]))
    #p.starmap(model_var_setup, zip(repeat(data), list(range(data['K']))))

    # 初始化变量
    # time_start2 = time.time()

    # tem = pool.map(dim.model_item_setObj, lists)
    # x-subproblem
    tem = go(data, var)
    var["U"] = tem[0][0]
    var["S"] = tem[0][1]
    var["Z_P"] = tem[0][2]
    var["Z_M"] = tem[0][3]
    var["V"] = tem[0][4]
    var["YUI"] = tem[0][5]
    var["YUO_P"] = tem[0][6]
    var["YUO_M"] = tem[0][7]
    var["XC"] = tem[0][8]
    var["RC1"] = tem[0][9]
    var["RC2"] = tem[0][10]
    #for i in list(range(data['K'])):
    #    model_item_setObj(data, i)

    # z-subproblem
    tem = model_item_global(var)
    var["X"] = tem[0]
    var["R"] = tem[1]

    # initialize rho
    #rho_update = 10

    xx = np.zeros((data["K"], data["K"], data["M"], data["N"]))
    for ii in range(data["K"]):
        xx[:, ii, :, :] = var["X"]

    # normalized average deviation
    norm_ave_dev_sol = 1
    #cost_dev = 1

    # determine whether copies of a variable are homogeneous
    # after first detecting homogeneous copies of a variable, set flag = True
    flag_X = np.zeros((data["K"], data["M"], data["N"]))
    flag_R = np.zeros((data["A"], data["M"], data["N"]))

    # force fixing if converging sufficiently
    #flag_X_force = False       # if detect sufficient convergence, then let be True
    #ite_X_force = 0
    #flag_R_force = False
    #ite_R_force = 0

    # imaginary cost
    '''
    HH = np.zeros((data["K"], data["M"], data["N"]))  # holding cost
    for t in range(data["N"]):
        HH[:, :, t] = data["H"]

    HH_R = np.zeros((data["A"], data["M"], data["N"]))
    for a in range(data["A"]):
        HH_R[a] = HH[data["cA"][a][1] - 1]

    var["rho_X"] = np.divide(HH, np.amax(var["XC"], axis=1) - np.amin(var["XC"], axis=1) + 1)

    var["rho_R"] = np.divide(HH_R, np.maximum(0.5 * np.absolute(var["R"] - var["RC1"]) + 0.5 * np.absolute(var["R"] - var["RC2"]), 1))
    '''
    var_fixing_rate = 0
    flag_remaining_solver = False

    # primal residual
    pr = 1

    # begin ADMM loop
    while pr > 1e-2:  # 当primal/dual residual < tolerance时终止循环
        if var_fixing_rate > 90:
            break

        if UB < math.inf and LB > -math.inf:
            rel_err = (UB - LB) / (UB + 1e-10)
            if rel_err < 1e-2:
                pass
                #break
        time_ite_start = time.time()

        var["ite"] += 1
        #if var["ite"] > 10:
        if var["ite"] > data["MaxIte"]:  # 当达到最大迭代次数时终止循环
            break

        # Update dual variables
        var["Mu"] += 1.6 * np.multiply(var["rho_X"], xx - var["XC"])
        var["Ksi"] += 1.6 * np.multiply(var["rho_R"], var["R"] - var["RC1"])
        var["Eta"] += 1.6 * np.multiply(var["rho_R"], var["R"] - var["RC2"])

        # update rho
        '''
        var["rho_X"] *= np.sum(np.divide(np.sum(np.power(var["XC"] - xx, 2), axis=1),
                                  np.amax(var["XC"], axis=1) - np.amin(var["XC"], axis=1) + 1)) / (data["K"] * data["K"] * data["M"] * data["N"]) * rho_update + 1

        var["rho_R"] *= np.sum(np.divide(np.power(var["RC1"] - var["R"], 2) + np.power(var["RC2"] - var["R"], 2),
                                  np.maximum(var["RC1"], var["RC2"]) - np.minimum(var["RC1"],
                                                                                  var["RC2"]) + 1)) / (2 * data["A"] * data["M"] * data["N"]) * rho_update + 1
        '''
        # x-subproblem
        # tem = pool.map(dim.model_item_setObj, lists)
        #time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt, parallel_time = go(data)
        tem = go(data, var)
        var["U"] = tem[0][0]
        var["S"] = tem[0][1]
        var["Z_P"] = tem[0][2]
        var["Z_M"] = tem[0][3]
        var["V"] = tem[0][4]
        var["YUI"] = tem[0][5]
        var["YUO_P"] = tem[0][6]
        var["YUO_M"] = tem[0][7]
        var["XC"] = tem[0][8]
        var["RC1"] = tem[0][9]
        var["RC2"] = tem[0][10]

        time_x = tem[1][0]
        time_x_model_var_setup = tem[1][1]
        time_x_set_obj = tem[1][2]
        time_x_opt = tem[1][3]
        parallel_time = tem[1][4]
        #p = mp.Pool(processes=min(mp.cpu_count(), data["K"]))
        #p.starmap(model_item_setObj, zip(repeat(data), list(range(data['K']))))
        # async_results = [pool.apply_async(model_item_setObj, args=(data, i)) for i in range(data['K'])]

        # pool.map(dim.model_item_setObj, lists)

        # z-subproblem
        tem = model_item_global(var)
        var["X"] = tem[0]
        var["R"] = tem[1]
        time_z_solver = tem[2]
        time_z_heur = tem[3]

        for ii in range(data["K"]):
            xx[:, ii, :, :] = var["X"]

        # primal residual
        tem1 = (var["R"] - var["RC1"]).reshape((-1, 1))
        tem2 = (var["R"] - var["RC2"]).reshape((-1, 1))
        tem3 = (xx - var["XC"]).reshape((-1, 1))
        tem = np.concatenate((tem1, tem2, tem3), axis=0)
        pr = LA.norm(tem)

        # determine whether copies of a variable are homogeneous
        if var["ite"] > var["LP_warm_start_ite"]:
            temp = (np.amax(var["XC"], axis=1) - np.amin(var["XC"], axis=1) < 1e-5)
            for i in range(data["K"]):
                for j in range(data["M"]):
                    for t in range(data["N"]):
                        if not var["Bool_X_FIX"][i][j][t]:  # if this variable has not yet been fixed
                            if temp[i][j][t]:
                                if not flag_X[i][j][t]:
                                    X_stable_ite[i][j][t] = 1
                                    X_stable_value[i][j][t] = var["X"][i, j, t]
                                    flag_X[i][j][t] = True
                                else:
                                    if abs(var["X"][i, j, t] - X_stable_value[i][j][t]) < 1e-5:
                                        X_stable_ite[i][j][t] += 1
                                    else:
                                        X_stable_ite[i][j][t] = 1
                                        X_stable_value[i][j][t] = var["X"][i, j, t]
                            else:
                                X_stable_ite[i][j][t] = 0

                            if X_stable_ite[i][j][t] == var["lag_X"] * data["K"] + 1:
                                var["X_FIX"][i][j][t] = X_stable_value[i][j][t]
                                var["Bool_X_FIX"][i][j][t] = True

            temp = (np.abs(var["RC1"] - var["RC2"]) < 1e-5)
            for a in range(data["A"]):
                for j in range(data["M"]):
                    for t in range(data["N"]):
                        if not var["Bool_R_FIX"][a][j][t]:  # if this variable has not yet been fixed
                            if temp[a][j][t]:
                                if not flag_R[a][j][t]:
                                    R_stable_ite[a][j][t] = 1
                                    R_stable_value[a][j][t] = var["R"][a, j, t]
                                    flag_R[a][j][t] = True
                                else:
                                    if abs(var["R"][a, j, t] - R_stable_value[a][j][t]) < 1e-5:
                                        R_stable_ite[a][j][t] += 1
                                    else:
                                        R_stable_ite[a][j][t] = 1
                                        R_stable_value[a][j][t] = var["R"][a, j, t]
                            else:
                                R_stable_ite[a][j][t] = 0

                            if R_stable_ite[a][j][t] == var["lag_R"] * 2 + 1:
                                var["R_FIX"][a][j][t] = R_stable_value[a][j][t]
                                var["Bool_R_FIX"][a][j][t] = True


        # force fixing if converging sufficiently (even not exactly homogeneous)
        '''
        norm_ave_dev_X = 0
        for i in range(data["K"]):
            for j in range(data["M"]):
                for t in range(data["N"]):
                    if var["X"][i, j, t] != 0:
                        norm_ave_dev_X += np.sum(np.abs(var["XC"][i, :, j, t] - var["X"][i, j, t]) / var["X"][i, j, t])
        norm_ave_dev_X /= data["K"]

        R_nonzero = (var["R"] != 0)
        norm_ave_dev_R = np.sum(np.divide(
            np.abs(var["RC1"][R_nonzero] - var["R"][R_nonzero]) + np.abs(var["RC2"][R_nonzero] - var["R"][R_nonzero]),
            var["R"][R_nonzero])) / 2

        norm_ave_dev_sol = (norm_ave_dev_X + norm_ave_dev_R) / 2
        '''

        '''
        if flag_X_force == False:
            if norm_ave_dev_X <= 1e-1:
                var["lag_X"] = 0
                flag_X_force = True
                ite_X_force = 0
        else:
            ite_X_force += 1
            if ite_X_force == 2:
                cost = np.multiply(HH, var["X"])
                cost[var["Bool_X_FIX"] == True] = math.inf
                free_x_index = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
                free_x_min_cost = cost[free_x_index]

                if free_x_min_cost < math.inf:
                    var["X_FIX"][free_x_index] = var["X"][free_x_index]
                    var["Bool_X_FIX"][free_x_index] = True

                ite_X_force = 0

        if flag_R_force == False:
            if norm_ave_dev_R <= 1e-1:
                var["lag_R"] = 0
                flag_R_force = True
                ite_R_force = 0
        else:
            ite_R_force += 1
            if ite_R_force == 2:
                cost = np.multiply(HH_R, var["R"])
                cost[var["Bool_R_FIX"] == True] = math.inf
                free_r_index = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
                free_r_min_cost = cost[free_r_index]

                if free_r_min_cost < math.inf:
                    var["R_FIX"][free_r_index] = var["R"][free_r_index]
                    var["Bool_R_FIX"][free_r_index] = True

                ite_R_force = 0
        '''

        #cost_dev_X = np.sum(np.multiply(HH, np.amax(var["XC"], axis=1) - np.amin(var["XC"], axis=1))) / np.sum(np.multiply(HH, np.amax(var["XC"], axis=1)))
        #cost_dev_R = np.sum(np.multiply(HH_R, np.maximum(var["RC1"], var["RC2"]) - np.minimum(var["RC1"], var["RC2"]))) / np.sum(np.multiply(HH_R, np.maximum(var["RC1"], var["RC2"])))
        #cost_dev = (cost_dev_X + cost_dev_R) / 2
        '''判断z问题的解析解x是否满足capacity constraint'''
        '''
        nu_mul_x = np.zeros((data["M"], data["N"]))
        for j in range(data["M"]):
            for t in range(data["N"]):
                nu_mul_x[j][t] = np.matmul(data["nu"][:, j], var["X"][:, j, t])
        flag = np.all(nu_mul_x <= data["C"])
        '''

        if var["ite"] > var["LP_warm_start_ite"] and var["ite"] % 1 == 0:
            bound_index.append(var["ite"])

            wr = "Iteration " + str(var["ite"]) + ':\n'
            wr_s = open('UB_problem_detective.txt', 'a')
            wr_s.write(wr)
            wr_s.close()

            #pass
            choice = "average"     # choice = "average"/"slam_min"/"runover_sec"/"part_sce_shuffle"/"part_sce_inorder"
            UB_new = UpperBounding(data, var, choice)
            if UB_new != None:
                UB_OBJ.append(UB_new)
                if UB_new < UB:
                    UB = UB_new
            else:
                UB_OBJ.append(math.inf)

            wr_s = open('LB_problem_detective.txt', 'a')
            wr_s.write(wr)
            wr_s.close()

            # LB using dual variables of ADMM
            LB_new = LowerBounding(data, var)
            # update dual variables individually
            # result = LowerBounding(nec_data, data_nu_C, [SubG_Mu, SubG_Ksi, SubG_Eta], var["V"], data_cA)
            if LB_new != None:
                LB_OBJ.append(LB_new)
                if LB_new > LB:
                    LB = LB_new
            else:
                LB_OBJ.append(-math.inf)


        # if beta_step < 1e-6:
        #    pass
        # break

        #UB_OBJ.append(UB)
        #LB_OBJ.append(LB)

        # adaptively update penalty
        '''
        if pr > 10 * dr and pr > p_tol:
            var["rho"] *= 2
        elif dr > 10 * pr and dr > d_tol:
            var["rho"] /= 2
        '''

        time_ite_end = time.time()

        var_fixing_rate = (np.count_nonzero(var["Bool_X_FIX"]) + np.count_nonzero(var["Bool_R_FIX"])) / (
                    (data["K"] + data["A"]) * data["M"] * data["N"]) * 100

        # 每十次迭代输出一次结果
        if var["ite"] % 10 == 0:
            ite_index = var["ite"] / 10
            nz.append(ite_index)

            #OB = comp_obj(data, var)
            #objs.append(OB)

            obj_error = abs(UB - solver_ob) / solver_ob * 100  # 单位是%
            rel_obj_err.append(obj_error)

            '''
            X_sol = np.reshape(var["X"], (-1, 1))
            S_sol = np.reshape(var["S"], (-1, 1))
            Z_P_sol = np.reshape(var["Z_P"], (-1, 1))
            Z_M_sol = np.reshape(var["Z_M"], (-1, 1))
            R_sol = np.reshape(var["R"], (-1, 1))
            V_sol = np.reshape(var["V"], (-1, 1))
            U_sol = np.reshape(var["U"], (-1, 1))
            YUI_sol = np.reshape(var["YUI"], (-1, 1))
            YUO_P_sol = np.reshape(var["YUO_P"], (-1, 1))
            YUO_M_sol = np.reshape(var["YUO_M"], (-1, 1))
            Sol = np.concatenate((X_sol, S_sol, Z_P_sol, Z_M_sol, R_sol, V_sol, U_sol, YUI_sol, YUO_P_sol, YUO_M_sol), axis=0)
            sol_error = LA.norm(Sol - solver_sol) / LA.norm(solver_sol) * 100  # 单位是%
            rel_sol_err.append(sol_error)
            '''
            # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
            # gbv.o_err.append(temo_error)

            wr = "Iteration : " + str(var["ite"]) + "\nPrimal residual : " + str(pr) \
                 + "\nVariable fixing rate(%) : " + str(var_fixing_rate) + "\nRelative objective error of UB wrt solver(%) : " + str(obj_error) + \
                 '\n' + "iteration time : " + str(
                time_ite_end - time_ite_start) + \
                 "\n x_sub time : " + str(time_x) + "\n x_model_var_setup time : " + str(
                time_x_model_var_setup) + "\n x_model_set_objective time : " + str(
                time_x_set_obj) + "\n x_solve time : " + str(time_x_opt) + "\n parallel time : " + str(
                parallel_time) + "\nz_sub solver time : " + str(time_z_solver) + "\nz_sub heuristic time : " + str(time_z_heur) + "\n"


            '''
            wr = "Iteration : " + str(data["ite"]) + '\n' + '\n' + "Primal residual : " + str(
                pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "iteration time : " + str(
                time_ite_end - time_ite_start) + \
                 "\n x_sub time : " + str(time_x) + "\n for loop time : " + str(for_time) + "\n\n"
            '''
            '''
            if flag == True:
                wr += "Solution of z-subproblem satisfy capacity constraints!\n\n"
            else:
                wr += "Solution of z-subproblem doesn't satisfy capacity constraints!\n\n"
            '''
            wr += "Upper bound: " + str(UB) + "\nLower bound: " + str(LB) + "\n\n"
            wr_s = open('LP_MIP_heuristic.txt', 'a')
            wr_s.write(wr)
            wr_s.close()

    #time_end2 = time.time()
    #time_A = time_end2 - time_start2

    if var["ite"] % 10 != 0:
        ite_index = var["ite"] / 10
        nz.append(ite_index)
        # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
        # gbv.o_err.append(temo_error)
        #OB = comp_obj(data, var)
        #objs.append(OB)

        obj_error = abs(UB - solver_ob) / solver_ob * 100           # 单位是%
        rel_obj_err.append(obj_error)
        '''
        X_sol = np.reshape(var["X"], (-1, 1))
        S_sol = np.reshape(var["S"], (-1, 1))
        Z_P_sol = np.reshape(var["Z_P"], (-1, 1))
        Z_M_sol = np.reshape(var["Z_M"], (-1, 1))
        R_sol = np.reshape(var["R"], (-1, 1))
        V_sol = np.reshape(var["V"], (-1, 1))
        U_sol = np.reshape(var["U"], (-1, 1))
        YUI_sol = np.reshape(var["YUI"], (-1, 1))
        YUO_P_sol = np.reshape(var["YUO_P"], (-1, 1))
        YUO_M_sol = np.reshape(var["YUO_M"], (-1, 1))
        Sol = np.concatenate((X_sol, S_sol, Z_P_sol, Z_M_sol, R_sol, V_sol, U_sol, YUI_sol, YUO_P_sol, YUO_M_sol),
                             axis=0)
        sol_error = LA.norm(Sol - solver_sol) / LA.norm(solver_sol) * 100    # 单位是%
        rel_sol_err.append(sol_error)
        '''

        wr = "Iteration : " + str(var["ite"]) + "\nPrimal residual : " + str(pr) \
                 + "\nVariable fixing rate(%) : " + str(var_fixing_rate) + "\nRelative objective error of UB wrt solver(%) : " + str(obj_error) + \
             '\n' + "iteration time : " + str(
            time_ite_end - time_ite_start) + \
             "\n x_sub time : " + str(time_x) + "\n x_model_var_setup time : " + str(
            time_x_model_var_setup) + "\n x_model_set_objective time : " + str(
            time_x_set_obj) + "\n x_solve time : " + str(time_x_opt) + "\n parallel time : " + str(
            parallel_time) + "\nz_sub solver time : " + str(time_z_solver) + "\nz_sub heuristic time : " + str(time_z_heur) + "\n\n"
        wr += "Upper bound: " + str(UB) + "\nLower bound: " + str(LB) + "\n\n" + "Finished!"
        wr_s = open('LP_MIP_heuristic.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
    else:
        wr = "Finished!"
        wr_s = open('LP_MIP_heuristic.txt', 'a')
        wr_s.write(wr)
        wr_s.close()

    flag_remaining_solver = True
    obj_remaining_solver, sol_remaining_solver = remaining_solver(data, var)

    if flag_remaining_solver:
        wr = "\nObjective of remaining extensive problem by solver : " + str(obj_remaining_solver) + "\n"
        wr_s = open('LP_MIP_heuristic.txt', 'a')
        wr_s.write(wr)
        wr_s.close()

    '''
    plt.figure(1)
    plt.plot(nz, pri_re, c='red', marker='*', linestyle='-', label='Primal residual')
    plt.plot(nz, d_re, c='green', marker='+', linestyle='--', label='Dual residual')
    plt.legend()
    plt.title("primal/dual residual")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("primal/dual residual")  # 设置y轴标注
    plt.savefig("LS1_medium.png")

    plt.figure(2)
    plt.plot(nz, objs, c='blue', linestyle='-', label='Objective')
    plt.axhline(solver_ob, c='red', linestyle='-', label='Gurobi Objective')
    plt.legend()
    plt.title("Objective value")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("Objective")  # 设置y轴标注
    plt.savefig("LS2_medium.png")

    plt.figure(3)
    plt.plot(nz, rel_obj_err, c='m', linestyle='-', label='Objective')
    plt.title("Relative error of objective value")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("Relative error (%)")  # 设置y轴标注
    plt.savefig("LS3_medium.png")

    plt.figure(4)
    plt.plot(nz, rel_sol_err, c='k', linestyle='-', label='Objective')
    plt.title("Relative error of solutions (L2-norm)")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("Relative error (%)")  # 设置y轴标注
    plt.savefig("LS4_medium.png")
    '''
    return bound_index, UB_OBJ, LB_OBJ, UB, LB

if __name__ == '__main__':
    set_start_method('fork')
    data, var = generate_data(5)

    item_model = []
    item_var = []

    # set up model for each item
    for i in range(data["K"]):
        prob, vars = model_var_setup(data, i)
        item_model.append(prob)
        item_var.append(vars)

    # Gurobi
    start_solver = time.time()
    solver_ob, solver_sol = solver(data)
    end_solver = time.time()

    # ADMM
    start_ADMM = time.time()
    bound_index, UB_OBJ, LB_OBJ, UB, LB = ADMM(data, var, solver_ob, solver_sol)
    end_ADMM = time.time()

    # ADMM收敛时计算所得解相应的目标函数
    #admm_ob = comp_obj(data, var)

    wr = "\nFinal upper bound : " + str(UB) + "\nFinal lower bound : " + str(
        LB) + "\nGurobi Objective : " + str(
        solver_ob) + '\nADMM time : ' + str(
        end_ADMM - start_ADMM) + "\nGurobi time : " + str(end_solver - start_solver) + "\niteration : " + str(
        bound_index) + "\nUB : " + str(UB_OBJ) + "\nLB : " + str(LB_OBJ) + "\nvariable fixing : " + str(var["Bool_X_FIX"]) + "\n"
    wr_s = open('LP_MIP_heuristic.txt', 'a')
    wr_s.write(wr)
    wr_s.close()

    '''
    data, var = generate_data(5)
    print("cL : " + str(data["cL"]) + "\ncA : " + str(data["cA"]) + "\ncE : " + str(data["cE"]))
    '''

