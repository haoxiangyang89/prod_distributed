'''item-based decomposition algorithm'''
import functools

'''Server process managers are more flexible than using shared memory objects because they can be made to support 
arbitrary object types like lists, dictionaries, Queue, Value, Array, etc. They are, however, slower than using 
shared memory.'''
from gurobipy import gurobipy as gp
# from gurobipy import GRB
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
    data["N"] = 5  # T:time 16; 30
    data["M"] = 10  # J:plant 6/90; 6
    data["K"] = 20  # I:item 78/600; 78

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
    tr_depth = data["K"] // 4
    #tr_depth = 3
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

    data["D"] = np.random.randint(10, 50, size=(data["K"], data["N"]))     # demand

    # epsilonUD=np.ones(N)*0.05
    # epsilonUI=0.8

    data["v0"] = np.random.randint(5, size=(data["K"], data["M"]))    # initial inventory

    data["nu"] = np.random.randint(1, 3, size=(data["K"], data["M"]))    # unit capacity used for production

    data["C"] = np.random.randint(data["K"] * 2 * 10, data["K"] * 2 * 30, size=(data["K"], data["N"]))    # production capacity

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
    data["MaxIte"] = 2000  # 最大迭代次数
    # parameters for computing primal/dual tolerance
    data["ep_abs"] = 1e-2
    data["ep_rel"] = 1e-4

    '''variables'''
    var["X"] = np.zeros((data["K"], data["M"], data["N"]))  # production
    var["S"] = np.zeros((data["K"], data["L"], data["N"]))  # transportation
    var["Z"] = np.zeros((data["K"], data["M"], data["N"]))  # supply
    var["R"] = np.zeros((data["A"], data["M"], data["N"]))  # replacement
    var["V"] = np.zeros((data["K"], data["M"], data["N"]))  # inventory
    var["U"] = np.zeros((data["K"], data["N"]))  # unmet demand
    var["YUI"] = np.zeros((data["K"], data["M"], data["N"]))  # inbound quantity
    var["YUO"] = np.zeros((data["K"], data["M"], data["N"]))  # outbound quantity

    # 矩阵x的扩张，x_{i(i')jt}=x_{ijt} for i'
    var["xx"] = np.zeros((data["K"], data["K"], data["M"], data["N"]))

    # Copying variables
    var["XC"] = np.zeros((data["K"], data["K"], data["M"], data["N"]))  # x_{i(i')jt}
    var["RC1"] = np.zeros((data["A"], data["M"], data["N"]))  # r_{a(i)jt}
    var["RC2"] = np.zeros((data["A"], data["M"], data["N"]))  # r_{a(i')jt}

    # Dual variables
    var["Mu"] = np.zeros((data["K"], data["K"], data["M"], data["N"]))
    var["Ksi"] = np.zeros((data["A"], data["M"], data["N"]))
    var["Eta"] = np.zeros((data["A"], data["M"], data["N"]))

    # parameters
    var["rho"] = 10  # initial penalty
    # 初始化变量
    var["ite"] = 0  # ite为迭代次数

    return data, var

def solver(data):
    prob = gp.Model("extensive")

    # variable
    v = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    u = prob.addMVar((data["K"], data["N"]))  # (i,t)
    z = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    yUI = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    yUO = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    x = prob.addMVar((data["K"], data["M"], data["N"]))  # (i,j,t)
    r = prob.addMVar((data["A"], data["M"], data["N"]))  # (a,j,t)
    s = prob.addMVar((data["K"], data["L"], data["N"]))  # (i,l,t)
    #w = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.INTEGER)  # (i,j,t)
    #xUb = prob.addMVar((gbv.K, gbv.M, gbv.N), vtype=GRB.BINARY)  # (i,j,t)

    # constraint
    # (1b)-(1c)
    prob.addConstrs((u[i][0] + gp.quicksum(z[i][j][0] for j in range(data["M"])) == data["D"][i][0] for i in range(data["K"])),
                    name='1b-1c1')
    prob.addConstrs(
        (u[i][t] - u[i][t - 1] + gp.quicksum(z[i][j][t] for j in range(data["M"])) == data["D"][i][t] for i in range(data["K"])
         for t in range(1, data["N"])), name='1c2')

    # (1d)
    # prob.addConstrs(
    #    (gp.quicksum(u[i][tt] / D[i][tt] for i in range(K) for tt in range(t + 1) if D[i][tt] != 0) <= epsilonUD[t] for
    #     t in range(N)), name='1d')

    # (1e)
    prob.addConstrs((v[i][j][0] - yUI[i][j][0] + yUO[i][j][0] == data["v0"][i][j] for i in range(data["K"])
                     for j in range(data["M"])), name='1e1')
    prob.addConstrs(
        (v[i][j][t] - v[i][j][t - 1] - yUI[i][j][t] + yUO[i][j][t] == 0 for i in range(data["K"]) for j in range(data["M"])
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
    prob.addConstrs((yUO[i][j][t] - gp.quicksum(s[i][l][t] for l in range(data["L"]) if data["cL"][l][0] == j + 1) -
                     gp.quicksum(data["q"][e] * x[data["cE"][e][1] - 1][j][t] for e in range(data["E"]) if data["cE"][e][0] == i + 1)
                     - z[i][j][t] - gp.quicksum(
        r[a][j][t] for a in range(data["A"]) if data["cA"][a][0] == i + 1) + gp.quicksum(
        r[a][j][t] for a in range(data["A"]) if data["cA"][a][1] == i + 1) == 0 for i in range(data["K"])
                     for j in range(data["M"]) for t in range(data["N"])),
                    name='1h')

    # (1i)
    prob.addConstrs(
        (gp.quicksum(data["nu"][i][j] * x[i][j][t] for i in range(data["K"])) <= data["C"][j][t] for j in range(data["M"])
         for t in range(data["N"])), name='1i')

    # (1j)
    prob.addConstrs((r[a][j][t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in range(data["A"])), name='1j1')
    prob.addConstrs(
        (v[i][j][t] - r[a][j][t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for i in range(data["K"])
         for a in range(data["A"]) if data["cA"][a][0] == i + 1),
        name='1j2')

    # (1k)
    prob.addConstrs((yUO[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
                    name='1k1')
    prob.addConstrs(
        (v[i][j][t] - yUO[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
        name='1k2')

    # (1l)
    prob.addConstrs((x[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l1')
    prob.addConstrs((yUI[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l2')
    prob.addConstrs((v[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l3')
    prob.addConstrs((z[i][j][t] >= 0 for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])), name='1l4')

    # (1m)
    prob.addConstrs((u[i][t] >= 0 for i in range(data["K"]) for t in range(data["N"])), name='1m')

    # (1n)
    prob.addConstrs((s[i][l][t] >= 0 for i in range(data["K"]) for l in range(data["L"]) for t in range(data["N"])), name='1n')
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
    prob.setObjective(gp.quicksum(data["H"][i][j] * v[i][j][t] for i in range(data["K"]) for j in range(data["M"]) for t in range(data["N"]))
                      + gp.quicksum(data["P"][i][t] * u[i][t] for i in range(data["K"]) for t in range(data["N"])))
    #time_end1 = time.time()
    #time_e_model = time_end1 - time_start1

    #time_start2 = time.time()
    prob.optimize()
    #time_end2 = time.time()
    #time_e_op = time_end2 - time_start2

    #gbv.W_E = w.X
    #gbv.XUb_E = xUb.X

    return prob.objVal
    #x.X, s.X, z.X, r.X, v.X, u.X, yUI.X, yUO.X

def model_var_setup(data, i):
    # 建立item i的model和variable
    prob = gp.Model("item " + str(i))

    # variable
    # variable
    ui = prob.addMVar(data["N"])  # u_{it} for t
    si = prob.addMVar((data["L"], data["N"]))  # s_{ilt} for l,t
    zi = prob.addMVar((data["M"], data["N"]))  # z_{ijt} for j,t
    vi = prob.addMVar((data["M"], data["N"]))  # v_{ijt} for j,t
    yUIi = prob.addMVar((data["M"], data["N"]))  # y^{I}_{ijt} for j,t
    yUOi = prob.addMVar((data["M"], data["N"]))  # y^{O}_{ijt} for j,t
    xCi = prob.addMVar((data["K"], data["M"], data["N"]))  # x_{i'j(i)t} for i',j,t
    if len(data["out_place"][i]) > 0:
        rC1i = prob.addMVar((len(data["out_place"][i]), data["M"], data["N"]))  # r_{a(i)jt} for a=(i,i')
    if len(data["in_place"][i]) > 0:
        rC2i = prob.addMVar((len(data["in_place"][i]), data["M"], data["N"]))  # r_{a(i)jt} for a=(i',i)
    # xUbi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.BINARY)   # x^{b}_{ijt} for j,t
    # wi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.INTEGER)   # w_{ijt} for j,t

    # Constraint
    # unmet demand transition
    prob.addConstr(ui[0] + gp.quicksum(zi[j, 0] for j in range(data["M"])) == data["D"][i][0], name='3b+3c1')
    prob.addConstrs(
        (ui[t] - ui[t - 1] + gp.quicksum(zi[j, t] for j in range(data["M"])) == data["D"][i][t] for t in
         range(1, data["N"])), name='3c2')

    # inventory transition
    prob.addConstrs((vi[j, 0] - yUIi[j, 0] + yUOi[j, 0] == data["v0"][i][j] for j in range(data["M"])),
                    name='3e1')
    prob.addConstrs(
        (vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi[j, t] == 0 for j in range(data["M"]) for t in
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
                (yUOi[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 zi[j,
                    t] - gp.quicksum(
                    rC1i[a, j, t] for a in range(len(data["out_place"][i]))) + gp.quicksum(
                    rC2i[a, j, t] for a in range(len(data["in_place"][i]))) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')
        elif len(data["in_place"][i]) == 0:
            prob.addConstrs(
                (yUOi[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 zi[j,
                    t] - gp.quicksum(
                    rC1i[a, j, t] for a in range(len(data["out_place"][i]))) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')
    elif len(data["out_place"][i]) == 0:
        if len(data["in_place"][i]) > 0:
            prob.addConstrs(
                (yUOi[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 zi[j,
                    t] + gp.quicksum(
                    rC2i[a, j, t] for a in range(len(data["in_place"][i]))) == 0 for j in range(data["M"]) for t in
                 range(data["N"])),
                name='3h')
        elif len(data["in_place"][i]) == 0:
            prob.addConstrs(
                (yUOi[j, t] - gp.quicksum(
                    si[ll, t] for ll in range(data["L"]) if data["cL"][ll][0] == j + 1) - gp.quicksum(
                    data["q"][e] * xCi[data["cE"][e][1] - 1, j, t] for e in range(data["E"]) if
                    data["cE"][e][0] == i + 1) -
                 zi[j,
                    t] == 0 for j in range(data["M"]) for t in
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
        prob.addConstrs(
            (rC1i[a, j, t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in
             range(len(data["out_place"][i]))),
            name='3j1')
        prob.addConstrs(
            (vi[j, t] - rC1i[a, j, t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in
             range(len(data["out_place"][i]))),
            name='3j3')
    if len(data["in_place"][i]) > 0:
        prob.addConstrs(
            (rC2i[a, j, t] >= 0 for j in range(data["M"]) for t in range(data["N"]) for a in
             range(len(data["in_place"][i]))),
            name='3j2')

    # outbound quantity bounds
    prob.addConstrs((yUOi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])),
                    name='3k1')
    prob.addConstrs(
        (vi[j, t] - yUOi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])),
        name='3k2')

    # non-negativity
    prob.addConstrs((yUIi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])), name='3l')
    prob.addConstrs((vi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])), name='3l2')
    prob.addConstrs((zi[j, t] >= 0 for j in range(data["M"]) for t in range(data["N"])), name='3l3')
    prob.addConstrs(
        (xCi[ii, j, t] >= 0 for ii in range(data["K"]) for j in range(data["M"]) for t in range(data["N"])),
        name='3m')
    prob.addConstrs((ui[t] >= 0 for t in range(data["N"])), name='3n')
    prob.addConstrs((si[ll, t] >= 0 for ll in range(data["L"]) for t in range(data["N"])), name='3o')

    vars = [ui, si, zi, vi, yUIi, yUOi, xCi]
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
    zi = item_var[i][2]
    vi = item_var[i][3]
    yUIi = item_var[i][4]
    yUOi = item_var[i][5]
    xCi = item_var[i][6]
    if len(data["out_place"][i]) > 0:
        rC1i = item_var[i][7]
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][8]
    elif len(data["out_place"][i]) == 0:
        if len(data["in_place"][i]) > 0:
            rC2i = item_var[i][7]

    time_model_var = time.time()

    # objective
    if var["ite"] == 0:
        if len(data["out_place"][i]) > 0:
            if len(data["in_place"][i]) > 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                    - gp.quicksum(
                        var["Ksi"][data["out_place"][i][a]][j][t] * rC1i[a, j, t] for a in
                        range(len(data["out_place"][i])) for
                        j in
                        range(data["M"]) for t in range(data["N"]))
                    - gp.quicksum(
                        var["Eta"][data["in_place"][i][a]][j][t] * rC2i[a, j, t] for a in
                        range(len(data["in_place"][i])) for j
                        in
                        range(data["M"]) for t in range(data["N"]))
                )
            elif len(data["in_place"][i]) == 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                    - gp.quicksum(
                        var["Ksi"][data["out_place"][i][a]][j][t] * rC1i[a, j, t] for a in
                        range(len(data["out_place"][i])) for
                        j in
                        range(data["M"]) for t in range(data["N"]))
                )
        elif len(data["out_place"][i]) == 0:
            if len(data["in_place"][i]) > 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                    - gp.quicksum(
                        var["Eta"][data["in_place"][i][a]][j][t] * rC2i[a, j, t] for a in
                        range(len(data["in_place"][i])) for j
                        in
                        range(data["M"]) for t in range(data["N"]))
                )
            elif len(data["in_place"][i]) == 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                )
    else:
        if len(data["out_place"][i]) > 0:
            if len(data["in_place"][i]) > 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                    - gp.quicksum(
                        var["Ksi"][data["out_place"][i][a]][j][t] * rC1i[a, j, t] for a in
                        range(len(data["out_place"][i])) for
                        j in
                        range(data["M"]) for t in range(data["N"]))
                    - gp.quicksum(
                        var["Eta"][data["in_place"][i][a]][j][t] * rC2i[a, j, t] for a in
                        range(len(data["in_place"][i])) for j
                        in
                        range(data["M"]) for t in range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["X"][ii][j][t] - xCi[ii, j, t]) ** 2 for ii in range(data["K"]) for j
                        in
                        range(data["M"]) for t in
                        range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["R"][data["out_place"][i][a]][j][t] - rC1i[a, j, t]) ** 2 for a in
                        range(len(data["out_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["R"][data["in_place"][i][a]][j][t] - rC2i[a, j, t]) ** 2 for a in
                        range(len(data["in_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))
                )
            elif len(data["in_place"][i]) == 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                    - gp.quicksum(
                        var["Ksi"][data["out_place"][i][a]][j][t] * rC1i[a, j, t] for a in
                        range(len(data["out_place"][i])) for
                        j in
                        range(data["M"]) for t in range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["X"][ii][j][t] - xCi[ii, j, t]) ** 2 for ii in range(data["K"]) for j
                        in
                        range(data["M"]) for t in
                        range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["R"][data["out_place"][i][a]][j][t] - rC1i[a, j, t]) ** 2 for a in
                        range(len(data["out_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))
                )
        elif len(data["out_place"][i]) == 0:
            if len(data["in_place"][i]) > 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in
                        range(data["N"]))
                    - gp.quicksum(
                        var["Eta"][data["in_place"][i][a]][j][t] * rC2i[a, j, t] for a in
                        range(len(data["in_place"][i])) for j
                        in
                        range(data["M"]) for t in range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["X"][ii][j][t] - xCi[ii, j, t]) ** 2 for ii in range(data["K"]) for j
                        in
                        range(data["M"]) for t in
                        range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["R"][data["in_place"][i][a]][j][t] - rC2i[a, j, t]) ** 2 for a in
                        range(len(data["in_place"][i]))
                        for j in
                        range(data["M"]) for t in range(data["N"]))
                )
            elif len(data["in_place"][i]) == 0:
                prob.setObjective(
                    gp.quicksum(data["H"][i][j] * vi[j, t] for j in range(data["M"]) for t in
                                range(data["N"]))
                    + gp.quicksum(data["P"][i][t] * ui[t] for t in range(data["N"]))
                    - gp.quicksum(
                        var["Mu"][ii][i][j][t] * xCi[ii, j, t] for ii in range(data["K"]) for j in
                        range(data["M"]) for t in range(data["N"]))
                    + var["rho"] / 2 * gp.quicksum(
                        (var["X"][ii][j][t] - xCi[ii, j, t]) ** 2 for ii in range(data["K"]) for j
                        in
                        range(data["M"]) for t
                        in
                        range(data["N"])))

    time_obj = time.time()

    prob.optimize()

    end = time.time()

    result = [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X]
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
    K, M, N = var["X"].shape
    X_new = np.maximum(0, np.sum(var["rho"] * var["XC"] - var["Mu"], axis=1)) / (
            var["rho"] * K)

    xx_new = np.zeros((K, K, M, N))
    for i in range(K):
        for j in range(M):
            for t in range(N):
                xx_new[i, :, j, t] = X_new[i][j][t]

    R_new = np.maximum(0, var["RC1"] + var["RC2"] - (var["Ksi"] + var["Eta"]) /
                           var["rho"]) / 2

    return [X_new, xx_new, R_new]

def comp_obj(data, var):  # Compute the objective corresponding to a solution
    VV = np.sum(var["V"], axis=2)
    ob = np.sum(np.multiply(data["P"], var["U"])) + np.sum(np.multiply(data["H"], VV))

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
        U_new = var["U"]
        S_new = var["S"]
        Z_new = var["Z"]
        V_new = var["V"]
        YUI_new = var["YUI"]
        YUO_new = var["YUO"]
        XC_new = var["XC"]
        RC1_new = var["RC1"]
        RC2_new = var["RC2"]

        for i in range(data['K']):
            U_new[i] = results[i][0]
            S_new[i] = results[i][1]
            Z_new[i] = results[i][2]
            V_new[i] = results[i][3]
            YUI_new[i] = results[i][4]
            YUO_new[i] = results[i][5]
            XC_new[:, i, :, :] = results[i][6]
            if len(data["out_place"][i]) > 0:
                RC1_new[data["out_place"][i]] = results[i][7]
                if len(data["in_place"][i]) > 0:
                    RC2_new[data["in_place"][i]] = results[i][8]
            elif len(data["out_place"][i]) == 0:
                if len(data["in_place"][i]) > 0:
                    RC2_new[data["in_place"][i]] = results[i][7]

            time_x.append(results[i][-4])
            time_x_model_var_setup.append(results[i][-3])
            time_x_set_obj.append(results[i][-2])
            time_x_opt.append(results[i][-1])

        parallel_time = end - start

        #return time_x, sum(time_x)
        return [[U_new, S_new, Z_new, V_new, YUI_new, YUO_new, XC_new, RC1_new, RC2_new], [time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt, parallel_time]]
        #return time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt, parallel_time

        # async_results = [pool.apply_async(model_item_setObj, args=(data, i)) for i in range(data['K'])]

def ADMM(data, var):
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
    pr = 10  # primal residual
    dr = 10  # dual residual
    p_tol = 1e-5  # primal tolerance
    d_tol = 1e-5  # dual tolerance

    # p,n for the computation of primal/dual tolerance
    sqrt_dim_p = math.sqrt(data["M"] * data["N"] * (data["K"] ** 2 + data["A"] * 2))
    sqrt_dim_n = sqrt_dim_p

    # set up model and variables
    #go("setup", data, lock)
    #p = mp.Pool(processes=min(mp.cpu_count(), data["K"]))
    #p.starmap(model_var_setup, zip(repeat(data), list(range(data['K']))))

    # 初始化变量
    time_start2 = time.time()

    # tem = pool.map(dim.model_item_setObj, lists)
    # x-subproblem
    tem = go(data, var)
    var["U"] = tem[0][0]
    var["S"] = tem[0][1]
    var["Z"] = tem[0][2]
    var["V"] = tem[0][3]
    var["YUI"] = tem[0][4]
    var["YUO"] = tem[0][5]
    var["XC"] = tem[0][6]
    var["RC1"] = tem[0][7]
    var["RC2"] = tem[0][8]
    #for i in list(range(data['K'])):
    #    model_item_setObj(data, i)

    # z-subproblem
    tem = model_item_global(var)
    var["X"] = tem[0]
    var["xx"] = tem[1]
    var["R"] = tem[2]

    # begin ADMM loop
    while pr > p_tol or dr > d_tol:  # 当primal/dual residual < tolerance时终止循环
        time_ite_start = time.time()

        var["ite"] += 1
        if var["ite"] > data["MaxIte"]:  # 当达到最大迭代次数时终止循环
            break

        # x-subproblem
        # tem = pool.map(dim.model_item_setObj, lists)
        #time_x, time_x_model_var_setup, time_x_set_obj, time_x_opt, parallel_time = go(data)
        tem = go(data, var)
        var["U"] = tem[0][0]
        var["S"] = tem[0][1]
        var["Z"] = tem[0][2]
        var["V"] = tem[0][3]
        var["YUI"] = tem[0][4]
        var["YUO"] = tem[0][5]
        var["XC"] = tem[0][6]
        var["RC1"] = tem[0][7]
        var["RC2"] = tem[0][8]

        time_x = tem[1][0]
        time_x_model_var_setup = tem[1][1]
        time_x_set_obj = tem[1][2]
        time_x_opt = tem[1][3]
        parallel_time = tem[1][4]
        #p = mp.Pool(processes=min(mp.cpu_count(), data["K"]))
        #p.starmap(model_item_setObj, zip(repeat(data), list(range(data['K']))))
        # async_results = [pool.apply_async(model_item_setObj, args=(data, i)) for i in range(data['K'])]

        # pool.map(dim.model_item_setObj, lists)

        # 存储z^k
        xp = var["X"]
        rp = var["R"]

        # z-subproblem
        tem = model_item_global(var)
        var["X"] = tem[0]
        var["xx"] = tem[1]
        var["R"] = tem[2]

        # Update dual variables
        var["Mu"] += 1.6 * var["rho"] * (var["xx"] - var["XC"])
        var["Ksi"] += 1.6 * var["rho"] * (var["R"] - var["RC1"])
        var["Eta"] += 1.6 * var["rho"] * (var["R"] - var["RC2"])

        xx_p = np.zeros((data["K"], data["K"], data["M"], data["N"]))
        for i in range(data["K"]):
            for j in range(data["M"]):
                for t in range(data["N"]):
                    xx_p[i, :, j, t] = xp[i][j][t]
        # primal residual
        tem1 = (var["R"] - var["RC1"]).reshape((-1, 1))
        tem2 = (var["R"] - var["RC2"]).reshape((-1, 1))
        tem3 = (var["xx"] - var["XC"]).reshape((-1, 1))
        tem = np.concatenate((tem1, tem2, tem3), axis=0)
        pr = LA.norm(tem)

        # dual residual
        tem1 = (var["R"] - rp).reshape((-1, 1))
        tem2 = (var["xx"] - xx_p).reshape((-1, 1))
        tem = np.concatenate((tem1, tem1, tem2), axis=0)
        dr = var["rho"] * LA.norm(tem)

        # primal tolerance
        A_x = np.concatenate((var["XC"].reshape(-1, 1), var["RC1"].reshape(-1, 1), var["RC2"].reshape(-1, 1)),
                             axis=0)
        B_z = np.concatenate(
            (var["xx"].reshape((-1, 1)), var["R"].reshape((-1, 1)), var["R"].reshape((-1, 1))), axis=0)
        p_tol = sqrt_dim_p * data["ep_abs"] + data["ep_rel"] * max(LA.norm(A_x), LA.norm(B_z))

        # dual tolerance
        A_y = np.concatenate(
            (var["Mu"].reshape((-1, 1)), var["Ksi"].reshape((-1, 1)), var["Eta"].reshape((-1, 1))), axis=0)
        d_tol = sqrt_dim_n * data["ep_abs"] + data["ep_rel"] * LA.norm(A_y)

        # adaptively update penalty
        if pr > 10 * dr:
            var["rho"] *= 2
        elif dr > 10 * pr:
            var["rho"] /= 2

        time_ite_end = time.time()

        # 每十次迭代输出一次结果
        if var["ite"] % 1 == 0:
            # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
            # gbv.o_err.append(temo_error)

            wr = "Iteration : " + str(var["ite"]) + '\n' + '\n' + "Primal residual : " + str(
                pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "iteration time : " + str(
                time_ite_end - time_ite_start) + \
                 "\n x_sub time : " + str(time_x) + "\n x_model_var_setup time : " + str(
                time_x_model_var_setup) + "\n x_model_set_objective time : " + str(
                time_x_set_obj) + "\n x_solve time : " + str(time_x_opt) + "\n parallel time : " + str(parallel_time) + "\n\n"
            '''
            wr = "Iteration : " + str(data["ite"]) + '\n' + '\n' + "Primal residual : " + str(
                pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "iteration time : " + str(
                time_ite_end - time_ite_start) + \
                 "\n x_sub time : " + str(time_x) + "\n for loop time : " + str(for_time) + "\n\n"
            '''
            wr_s = open('LS.txt', 'a')
            wr_s.write(wr)
            wr_s.close()

    time_end2 = time.time()
    time_A = time_end2 - time_start2

    '''
    if data["ite"] % 10 != 0:
        # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
        # gbv.o_err.append(temo_error)

        wr = "Iteration : " + str(data["ite"]) + '\n' + "Primal residual : " + str(
            pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "ADMM time: " + str(
            time_A) + '\n' + '\n' + "Finished!"

        wr_s = open('LS.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
    else:
        wr = "ADMM time: " + str(
            time_A) + '\n' + '\n' + "Finished!"
        wr_s = open('LS.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
    '''
    '''
    nz = len(pri_re)
    pri_re = np.array(pri_re)
    d_re = np.array(d_re)

    plt.figure(1)
    plt.plot(range(nz), pri_re, c='red', marker='*', linestyle='-', label='Primal residual')
    plt.plot(range(nz), d_re, c='green', marker='+', linestyle='--', label='Dual residual')
    plt.legend()
    plt.title("primal/dual residual")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("primal/dual residual")  # 设置y轴标注
    plt.savefig("LS1.png")

    plt.figure(2)
    plt.plot(range(nz), .Ob, c='blue', linestyle='-', label='Objective')
    plt.title("Objective value")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("Objective")  # 设置y轴标注
    plt.savefig("LS2.png")

    plt.figure(3)
    plt.plot(range(nz), gbv.o_err, c='m', linestyle='-', label='Objective')
    plt.title("Relative error of objective value")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("Relative error (%)")  # 设置y轴标注
    plt.savefig("LS3.png")

    plt.figure(4)
    plt.plot(range(nz), gbv.r_err, c='k', linestyle='-', label='Objective')
    plt.title("Relative error of solutions (L2-norm)")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("Relative error (%)")  # 设置y轴标注
    plt.savefig("LS4.png")
    '''

if __name__ == '__main__':
    set_start_method('fork')
    data, var = generate_data(1)

    item_model = []
    item_var = []

    # set up model for each item
    for i in range(data["K"]):
        prob, vars = model_var_setup(data, i)
        item_model.append(prob)
        item_var.append(vars)

    # ADMM
    start_ADMM = time.time()
    ADMM(data, var)
    end_ADMM = time.time()

    # ADMM收敛时计算所得解相应的目标函数
    admm_ob = comp_obj(data, var)
    # Gurobi
    start_solver = time.time()
    solver_ob = solver(data)
    end_solver = time.time()

    wr = "\nADMM Objective : " + str(admm_ob) + "\nGurobi Objective : " + str(solver_ob) + '\nADMM time : ' + str(
        end_ADMM - start_ADMM) + "\nGurobi time : " + str(end_solver - start_solver) + "\n"
    wr_s = open('LS.txt', 'a')
    wr_s.write(wr)
    wr_s.close()
    '''
    wr = "X: " + str(data["X"]) + '\n S: ' + str(data["S"]) + "\n Z: " + str(data["Z"]) + "\n R: " + str(
        data["R"]) + "\n V: " + str(data["V"]) + "\n U: " + str(data["U"]) + "\n YUI: " + str(
        data["YUI"]) + "\n YUO: " + str(data["YUO"])
    wr_s = open('LS.txt', 'a')
    wr_s.write(wr)
    wr_s.close()
    '''

    '''
    for i in range(data["K"]):
        model_item_setObj(data, i)

    with Pool() as pool:
        # submit all tasks
        l = [(data, 0), (data, 1), (data, 2)]
        pool.starmap(model_item_setObj, l)
    '''
    '''
    procs = []
    # instantiating process with arguments
    for i in range(data["K"]):
        proc = Process(target=model_item_setObj, args=(data, i))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    '''










