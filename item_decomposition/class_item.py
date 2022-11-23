from gurobipy import gurobipy as gp
# from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import time
import math
import matplotlib.pyplot as plt
# import multiprocessing as mp
# from multiprocessing.dummy import Pool
from multiprocessing import Process
# from ray.util.multiprocessing import Pool
import random
# import functools
import pathos.pools as pp
# from pathos.pp import ParallelPool
# from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import cpu_count
class item_dim():
    rho = 50
    # 初始化变量, hard-wired parameter setups
    ite = 0

    def __init__(self, s=None):
        if s != None:
            random.seed(s)
            np.random.seed(s)

        self.N = 16  # T:time 16; 30
        self.M = 40  # J:plant 6/90; 6
        self.K = 10  # I:item 78/600; 78

        self.cL = []
        cLL = np.random.choice(a=[False, True], size=(self.M, self.M), p=[0.9, 0.1])
        for j1 in range(self.M):
            for j2 in range(self.M):
                if j2 != j1 and cLL[j1][j2]:
                    self.cL.append((j1 + 1, j2 + 1))
        self.L = len(self.cL)

        tr_depth = self.K // 10
        tr_index = np.random.choice(a=list(range(tr_depth)), size=(1, self.K))
        tr_index_dic = {}
        for d in range(tr_depth):
            tr_index_dic[d] = []
        for i in range(self.K):
            tr_index_dic[tr_index[0][i]].append(i + 1)
        self.cE = []
        for d in range(tr_depth):
            for i in tr_index_dic[d]:
                for dd in range(d + 1, tr_depth):
                    for ii in tr_index_dic[dd]:
                        tem = random.uniform(0, 1)
                        if tem > 0.9:
                            self.cE.append((i, ii))
        self.E = len(self.cE)

        self.cA = []
        cAA = np.random.choice(a=[False, True], size=(self.K, self.K), p=[0.9, 0.1])
        for i1 in range(1, self.K):
            for i2 in range(0, i1):
                if cAA[i1][i2]:
                    self.cA.append((i1 + 1, i2 + 1))
                    self.cA.append((i2 + 1, i1 + 1))
        self.A = len(self.cA)

        self.H = np.random.randint(1, 5, size=(self.K, self.M))

        self.P = np.random.randint(200, 500, size=(self.K, self.N))

        self.D = np.random.randint(10, 50, size=(self.K, self.N))

        # epsilonUD=np.ones(N)*0.05
        # epsilonUI=0.8

        self.v0 = np.random.randint(5, size=(self.K, self.M))

        self.nu = np.random.randint(1, 3, size=(self.K, self.M))

        self.C = np.random.randint(self.K * 2 * 10, self.K * 2 * 30, size=(self.K, self.N))

        self.dtUL = np.random.randint(1, 3, size=(self.K, self.L))  # delta t^L _il

        self.dtUP = np.random.randint(1, 3, size=(self.K, self.M))  # delta t^P_ij

        self.Q = np.zeros((self.K, self.M, self.N))

        self.q = np.random.randint(1, 5, size=self.E)

        #gbv.m = 5 * np.ones(gbv.K)
        #gbv.wLB = 1 * np.ones((gbv.K, gbv.M))
        #gbv.wUB = 20 * np.ones((gbv.K, gbv.M))

        '''
        self.N = 2  # T:time
        self.M = 3  # J:plant
        self.K = 3  # I:item
        self.L = 2  # l
        self.E = 1
        self.A = 1  # a

        self.H = np.array([[2, 1, 3]])
        self.H = np.tile(self.H, (self.K, 1))
        self.P = 200 * np.ones((self.K, self.N))

        self.D = np.array([[5, 7], [10, 8], [6, 4]])

        # gbv.epsilonUD = np.ones(gbv.N) * 0.05
        # gbv.epsilonUI = 0.8

        self.v0 = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])

        self.nu = np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]])

        self.C = np.array([[50], [50], [50]])
        self.C = np.tile(self.C, (1, self.N))

        self.cL = [(1, 2), (2, 3)]
        self.cE = [(2, 1)]
        self.cA = [(3, 2)]

        self.dtUL = np.ones((self.K, self.L))  # delta t^L _il

        self.dtUP = np.zeros((self.K, self.M))  # delta t^P_ij

        self.Q = np.zeros((self.K, self.M, self.N))

        self.q = 2 * np.ones(self.E)
        '''

        # 每个item作为集合A中边的出端(i,i')或者入端(i',i)时，相应边在集合A中对应的序号
        self.out_place = []
        for i in range(self.K):
            tem = []
            for a in range(self.A):
                if self.cA[a][0] == i + 1:
                    tem.append(a)
            self.out_place.append(tem)

        self.in_place = []
        for i in range(self.K):
            tem = []
            for a in range(self.A):
                if self.cA[a][1] == i + 1:
                    tem.append(a)
            self.in_place.append(tem)

        # 为每个item首先建立一个model存起来
        self.item_model = [0] * self.K  # 存储item的model
        self.item_var = [0] * self.K  # 存储item的变量

        # ADMM variables
        self.X = np.zeros((self.K, self.M, self.N))
        self.S = np.zeros((self.K, self.L, self.N))
        self.Z = np.zeros((self.K, self.M, self.N))
        self.R = np.zeros((self.A, self.M, self.N))
        self.V = np.zeros((self.K, self.M, self.N))
        self.U = np.zeros((self.K, self.N))
        self.YUI = np.zeros((self.K, self.M, self.N))
        self.YUO = np.zeros((self.K, self.M, self.N))
        # gbv.XUb = np.zeros((gbv.K, gbv.M, gbv.N))
        # gbv.W = np.zeros((gbv.K, gbv.M, gbv.N))

        # 对x的复制，x_{i(i')jt}, where x_{i(i')jt}=x_{ijt} for i'
        self.xx = np.zeros((self.K, self.K, self.M, self.N))

        # Copying variables
        self.XC = np.zeros((self.K, self.K, self.M, self.N))  # x_{i(i')jt}
        self.RC1 = np.zeros((self.A, self.M, self.N))  # r_{a(i)jt}
        self.RC2 = np.zeros((self.A, self.M, self.N))  # r_{a(i')jt}

        # Dual variables
        self.Mu = np.zeros((self.K, self.K, self.M, self.N))
        self.Ksi = np.zeros((self.A, self.M, self.N))
        self.Eta = np.zeros((self.A, self.M, self.N))

    def model_var_setup(self, i):     # 建立item i的model和variable, local problem
        prob = gp.Model("item " + str(i))

        '''
        # variable
        ui = prob.addMVar(self.N)  # u_{it} for t
        si = prob.addMVar((self.L, self.N))  # s_{ilt} for l,t
        zi = prob.addMVar((self.M, self.N))  # z_{ijt} for j,t
        vi = prob.addMVar((self.M, self.N))  # v_{ijt} for j,t
        yUIi = prob.addMVar((self.M, self.N))  # y^{I}_{ijt} for j,t
        yUOi = prob.addMVar((self.M, self.N))  # y^{O}_{ijt} for j,t
        xCi = prob.addMVar((self.K, self.M, self.N))  # x_{i'j(i)t} for i',j,t
        if len(self.out_place[i]) > 0:
            rC1i = prob.addMVar((len(self.out_place[i]), self.M, self.N))  # r_{a(i)jt} for a=(i,i')
        if len(self.in_place[i]) > 0:
            rC2i = prob.addMVar((len(self.in_place[i]), self.M, self.N))  # r_{a(i)jt} for a=(i',i)
        # xUbi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.BINARY)   # x^{b}_{ijt} for j,t
        # wi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.INTEGER)   # w_{ijt} for j,t

        # Constraint
        # Unmet demand balance
        prob.addConstr(ui[0] + gp.quicksum(zi[j, 0] for j in range(self.M)) == self.D[i][0],
                       name='Unmet demand balance (1): ' + str(i))
        prob.addConstrs(
            (ui[t] - ui[t - 1] + gp.quicksum(zi[j, t] for j in range(self.M)) == self.D[i][t] for t in
             range(1, self.N)), name='Unmet demand balance (2): ' + str(i))

        # Inventory balance
        prob.addConstrs((vi[j, 0] - yUIi[j, 0] + yUOi[j, 0] == self.v0[i][j] for j in range(self.M)),
                        name='Inventory balance (1): ' + str(i))
        prob.addConstrs(
            (vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi[j, t] == 0 for j in range(self.M) for t in
             range(1, self.N)), name='Inventory balance (2): ' + str(i))

        # Inbound balance
        prob.addConstrs(
            (yUIi[j, t] - gp.quicksum(
                si[ll, t - int(self.dtUL[i][ll])] for ll in range(self.L) if self.cL[ll][1] == j + 1 and t - self.dtUL[i][ll] >= 0) -
             xCi[i, j, t - int(self.dtUP[i][j])] == self.Q[i][j][t] for j in range(self.M) for t in range(self.N) if
             t >= self.dtUP[i][j]),
            name='Inbound balance (1): ' + str(i))
        prob.addConstrs(
            (yUIi[j, t] - gp.quicksum(
                si[ll, t - self.dtUL[i][ll]] for ll in range(self.L) if
                self.cL[ll][1] == j + 1 and t - self.dtUL[i][ll] >= 0) ==
             self.Q[i][j][t] for j in
             range(self.M) for t in range(self.N) if t < self.dtUP[i][j]),
            name='Inbound balance (2): ' + str(i))

        # Outbound balance
        prob.addConstrs((yUOi[j, t] - gp.quicksum(si[ll, t] for ll in range(self.L) if self.cL[ll][0] == j + 1) - gp.quicksum(
            self.q[e] * xCi[self.cE[e][1] - 1, j, t] for e in range(self.E) if self.cE[e][0] == i + 1) - zi[j, t] - gp.quicksum(
            rC1i[a, j, t] for a in range(len(self.out_place[i]))) + gp.quicksum(
            rC2i[a, j, t] for a in range(len(self.in_place[i]))) == 0 for j in range(self.M) for t in range(self.N)),
                        name='Outbound balance: ' + str(i))

        # Capacity constraint
        prob.addConstrs(
            (gp.quicksum(self.nu[ii][j] * xCi[ii, j, t] for ii in range(self.K)) <= self.C[j][t] for j in range(self.M) for t in
             range(self.N)),
            name='Capacity constraint: ' + str(i))

        # Nonnegative variables
        prob.addConstrs(
            (rC1i[a, j, t] >= 0 for j in range(self.M) for t in range(self.N) for a in range(len(self.out_place[i]))),
            name='Nonnegative replacement copies (1): ' + str(i))
        prob.addConstrs(
            (rC2i[a, j, t] >= 0 for j in range(self.M) for t in range(self.N) for a in range(len(self.in_place[i]))),
            name='Nonnegative replacement copies (2): ' + str(i))
        prob.addConstrs((yUOi[j, t] >= 0 for j in range(self.M) for t in range(self.N)),
                        name='Nonnegative outbound: ' + str(i))
        prob.addConstrs((yUIi[j, t] >= 0 for j in range(self.M) for t in range(self.N)),
                        name='Nonnegative inbound: ' + str(i))
        prob.addConstrs((vi[j, t] >= 0 for j in range(self.M) for t in range(self.N)),
                        name='Nonnegative inventory: ' + str(i))
        prob.addConstrs((zi[j, t] >= 0 for j in range(self.M) for t in range(self.N)), name='Nonnegative supply: ' + str(i))
        prob.addConstrs((xCi[ii, j, t] >= 0 for ii in range(self.K) for j in range(self.M) for t in range(self.N)),
                        name='Nonnegative production copies: ' + str(i))
        prob.addConstrs((ui[t] >= 0 for t in range(self.N)), name='Nonnegative unmet demand: ' + str(i))
        prob.addConstrs((si[ll, t] >= 0 for ll in range(self.L) for t in range(self.N)),
                        name='Nonnegative transportation: ' + str(i))

        # Replacement,Outbound no greater than inventory
        prob.addConstrs(
            (vi[j, t] - rC1i[a, j, t] >= 0 for j in range(self.M) for t in range(self.N) for a in
             range(len(self.out_place[i]))),
            name='Replacement no greater than inventory: ' + str(i))
        prob.addConstrs(
            (vi[j, t] - yUOi[j, t] >= 0 for j in range(self.M) for t in range(self.N)),
            name='Outbound no greater than inventory: ' + str(i))
        '''
        # variable
        ui = prob.addMVar(self.N)  # u_{it} for t
        si = prob.addMVar((self.L, self.N))  # s_{ilt} for l,t
        zi = prob.addMVar((self.M, self.N))  # z_{ijt} for j,t
        vi = prob.addMVar((self.M, self.N))  # v_{ijt} for j,t
        yUIi = prob.addMVar((self.M, self.N))  # y^{I}_{ijt} for j,t
        yUOi = prob.addMVar((self.M, self.N))  # y^{O}_{ijt} for j,t
        xCi = prob.addMVar((self.K, self.M, self.N))  # x_{i'j(i)t} for i',j,t
        if len(self.out_place[i]) > 0:
            rC1i = prob.addMVar((len(self.out_place[i]), self.M, self.N))  # r_{a(i)jt} for a=(i,i')
        if len(self.in_place[i]) > 0:
            rC2i = prob.addMVar((len(self.in_place[i]), self.M, self.N))  # r_{a(i)jt} for a=(i',i)
        # xUbi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.BINARY)   # x^{b}_{ijt} for j,t
        # wi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.INTEGER)   # w_{ijt} for j,t

        # Constraint
        # (3b)-(3c)
        prob.addConstr(ui[0] + gp.quicksum(zi[j][0] for j in range(self.M)) == self.D[i][0], name='3b+3c1')
        prob.addConstrs(
            (ui[t] - ui[t - 1] + gp.quicksum(zi[j][t] for j in range(self.M)) == self.D[i][t] for t in
             range(1, self.N)), name='3c2')

        # (3e)
        prob.addConstrs((vi[j][0] - yUIi[j][0] + yUOi[j][0] == self.v0[i][j] for j in range(self.M)),
                        name='3e1')
        prob.addConstrs(
            (vi[j][t] - vi[j][t - 1] - yUIi[j][t] + yUOi[j][t] == 0 for j in range(self.M) for t in
             range(1, self.N)), name='3e2')

        # (3g)
        prob.addConstrs(
            (yUIi[j][t] - gp.quicksum(
                si[ll][t - int(self.dtUL[i][ll])] for ll in range(self.L) if
                self.cL[ll][1] == j + 1 and t - self.dtUL[i][ll] >= 0) -
             xCi[i][j][
                 t - int(self.dtUP[i][j])] == self.Q[i][j][t] for j in range(self.M) for t in range(self.N) if
             t >= self.dtUP[i][j]),
            name='3g1')
        prob.addConstrs(
            (yUIi[j][t] - gp.quicksum(
                si[ll][t - self.dtUL[i][ll]] for ll in range(self.L) if
                self.cL[ll][1] == j + 1 and t - self.dtUL[i][ll] >= 0) ==
             self.Q[i][j][t] for j in
             range(self.M) for t in range(self.N) if t < self.dtUP[i][j]),
            name='3g2')

        # (3h)
        if len(self.out_place[i]) > 0:
            if len(self.in_place[i]) > 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(self.L) if self.cL[ll][0] == j + 1) - gp.quicksum(
                        self.q[e] * xCi[self.cE[e][1] - 1][j][t] for e in range(self.E) if self.cE[e][0] == i + 1) -
                     zi[j][
                         t] - gp.quicksum(
                        rC1i[a][j][t] for a in range(len(self.out_place[i]))) + gp.quicksum(
                        rC2i[a][j][t] for a in range(len(self.in_place[i]))) == 0 for j in range(self.M) for t in
                     range(self.N)),
                    name='3h')
            elif len(self.in_place[i]) == 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(self.L) if self.cL[ll][0] == j + 1) - gp.quicksum(
                        self.q[e] * xCi[self.cE[e][1] - 1][j][t] for e in range(self.E) if self.cE[e][0] == i + 1) -
                     zi[j][
                         t] - gp.quicksum(
                        rC1i[a][j][t] for a in range(len(self.out_place[i]))) == 0 for j in range(self.M) for t in
                     range(self.N)),
                    name='3h')
        elif len(self.out_place[i]) == 0:
            if len(self.in_place[i]) > 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(self.L) if self.cL[ll][0] == j + 1) - gp.quicksum(
                        self.q[e] * xCi[self.cE[e][1] - 1][j][t] for e in range(self.E) if self.cE[e][0] == i + 1) -
                     zi[j][
                         t] + gp.quicksum(
                        rC2i[a][j][t] for a in range(len(self.in_place[i]))) == 0 for j in range(self.M) for t in
                     range(self.N)),
                    name='3h')
            elif len(self.in_place[i]) == 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(self.L) if self.cL[ll][0] == j + 1) - gp.quicksum(
                        self.q[e] * xCi[self.cE[e][1] - 1][j][t] for e in range(self.E) if self.cE[e][0] == i + 1) -
                     zi[j][
                         t] == 0 for j in range(self.M) for t in
                     range(self.N)),
                    name='3h')

        # (3i)
        prob.addConstrs(
            (gp.quicksum(self.nu[ii][j] * xCi[ii][j][t] for ii in range(self.K)) <= self.C[j][t] for j in range(self.M) for
             t in
             range(self.N)),
            name='3i')

        # (3j)
        if len(self.out_place[i]) > 0:
            prob.addConstrs(
                (rC1i[a][j][t] >= 0 for j in range(self.M) for t in range(self.N) for a in
                 range(len(self.out_place[i]))),
                name='3j1')
            prob.addConstrs(
                (vi[j][t] - rC1i[a][j][t] >= 0 for j in range(self.M) for t in range(self.N) for a in
                 range(len(self.out_place[i]))),
                name='3j3')
        if len(self.in_place[i]) > 0:
            prob.addConstrs(
                (rC2i[a][j][t] >= 0 for j in range(self.M) for t in range(self.N) for a in
                 range(len(self.in_place[i]))),
                name='3j2')

        # (3k)
        prob.addConstrs((yUOi[j][t] >= 0 for j in range(self.M) for t in range(self.N)),
                        name='3k1')
        prob.addConstrs(
            (vi[j][t] - yUOi[j][t] >= 0 for j in range(self.M) for t in range(self.N)),
            name='3k2')

        # (3l)
        prob.addConstrs((yUIi[j][t] >= 0 for j in range(self.M) for t in range(self.N)), name='3l')
        prob.addConstrs((vi[j][t] >= 0 for j in range(self.M) for t in range(self.N)), name='3l2')
        prob.addConstrs((zi[j][t] >= 0 for j in range(self.M) for t in range(self.N)), name='3l3')

        # (3m)
        prob.addConstrs((xCi[ii][j][t] >= 0 for ii in range(self.K) for j in range(self.M) for t in range(self.N)),
                        name='3m')

        # (3n)
        prob.addConstrs((ui[t] >= 0 for t in range(self.N)), name='3n')

        # (3o)
        prob.addConstrs((si[ll][t] >= 0 for ll in range(self.L) for t in range(self.N)), name='3o')

        if len(self.out_place[i]) > 0:
            if len(self.in_place[i]) > 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi, rC1i, rC2i]
            elif len(self.in_place[i]) == 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi, rC1i]
        elif len(self.out_place[i]) == 0:
            if len(self.in_place[i]) > 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi, rC2i]
            elif len(self.in_place[i]) == 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi]

    def model_item_setObj(self, i):

        prob = self.item_model[i]
        # [ui, si, zi, vi, yUIi, yUOi, xCi, rC1i, rC2i] = self.item_var[i]
        # out_place: forward star, in_place: backward star
        # load the data structure
        if len(self.out_place[i]) > 0:
            if len(self.in_place[i]) > 0:
                ui = self.item_var[i][0]
                si = self.item_var[i][1]
                zi = self.item_var[i][2]
                vi = self.item_var[i][3]
                yUIi = self.item_var[i][4]
                yUOi = self.item_var[i][5]
                xCi = self.item_var[i][6]
                rC1i = self.item_var[i][7]
                rC2i = self.item_var[i][8]
            elif len(self.in_place[i]) == 0:
                ui = self.item_var[i][0]
                si = self.item_var[i][1]
                zi = self.item_var[i][2]
                vi = self.item_var[i][3]
                yUIi = self.item_var[i][4]
                yUOi = self.item_var[i][5]
                xCi = self.item_var[i][6]
                rC1i = self.item_var[i][7]
        elif len(self.out_place[i]) == 0:
            if len(self.in_place[i]) > 0:
                ui = self.item_var[i][0]
                si = self.item_var[i][1]
                zi = self.item_var[i][2]
                vi = self.item_var[i][3]
                yUIi = self.item_var[i][4]
                yUOi = self.item_var[i][5]
                xCi = self.item_var[i][6]
                rC2i = self.item_var[i][7]
            elif len(self.in_place[i]) == 0:
                ui = self.item_var[i][0]
                si = self.item_var[i][1]
                zi = self.item_var[i][2]
                vi = self.item_var[i][3]
                yUIi = self.item_var[i][4]
                yUOi = self.item_var[i][5]
                xCi = self.item_var[i][6]

        '''
        if self.ite == 0:
            prob.setObjective(gp.quicksum(self.H.tolist()[i][j] * vi.tolist()[j][t] for j in range(self.M) for t in range(self.N))
                              + gp.quicksum(self.P.tolist()[i][t] * ui.tolist()[t] for t in range(self.N))
                              - gp.quicksum(
                self.Mu.tolist()[ii][i][j][t] * xCi.tolist()[ii][j][t] for ii in range(self.K) for j in
                range(self.M) for t in
                range(self.N))
                              - gp.quicksum(
                self.Ksi.tolist()[self.out_place[i][a]][j][t] * rC1i.tolist()[a][j][t] for a in range(len(self.out_place[i])) for j in
                range(self.M) for t in range(self.N))
                              - gp.quicksum(
                self.Eta.tolist()[self.in_place[i][a]][j][t] * rC2i.tolist()[a][j][t] for a in range(len(self.in_place[i])) for j in
                range(self.M) for t in range(self.N))
                              )
        else:
            prob.setObjective(
                gp.quicksum(self.H.tolist()[i][j] * vi.tolist()[j][t] for j in range(self.M) for t in range(self.N))
                + gp.quicksum(self.P.tolist()[i][t] * ui.tolist()[t] for t in range(self.N))
                - gp.quicksum(
                    self.Mu.tolist()[ii][i][j][t] * xCi.tolist()[ii][j][t] for ii in range(self.K) for j in
                    range(self.M) for t in
                    range(self.N))
                - gp.quicksum(
                    self.Ksi.tolist()[self.out_place[i][a]][j][t] * rC1i.tolist()[a][j][t] for a in range(len(self.out_place[i]))
                    for j in
                    range(self.M) for t in range(self.N))
                - gp.quicksum(
                    self.Eta.tolist()[self.in_place[i][a]][j][t] * rC2i.tolist()[a][j][t] for a in range(len(self.in_place[i]))
                    for j in
                    range(self.M) for t in range(self.N))
                              + self.rho / 2 * gp.quicksum(
                (self.X.tolist()[ii][j][t] - xCi.tolist()[ii][j][t]) ** 2 for ii in range(self.K) for j in range(self.M) for t in range(self.N))
                              + self.rho / 2 * gp.quicksum(
                (self.R.tolist()[self.out_place[i][a]][j][t] - rC1i.tolist()[a][j][t]) ** 2 for a in range(len(self.out_place[i])) for j in
                range(self.M) for t in range(self.N))
                              + self.rho / 2 * gp.quicksum(
                (self.R.tolist()[self.in_place[i][a]][j][t] - rC2i.tolist()[a][j][t]) ** 2 for a in range(len(self.in_place[i])) for j in
                range(self.M) for t in range(self.N))
                              )
        '''
        # set up the objective function
        if self.ite == 0:
            if len(self.out_place[i]) > 0:
                if len(self.in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      - gp.quicksum(
                        self.Ksi[self.out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(self.out_place[i])) for
                        j in
                        range(self.M) for t in range(self.N))
                                      - gp.quicksum(
                        self.Eta[self.in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(self.in_place[i])) for j
                        in
                        range(self.M) for t in range(self.N))
                                      )
                elif len(self.in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      - gp.quicksum(
                        self.Ksi[self.out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(self.out_place[i])) for
                        j in
                        range(self.M) for t in range(self.N))
                                      )
            elif len(self.out_place[i]) == 0:
                if len(self.in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      - gp.quicksum(
                        self.Eta[self.in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(self.in_place[i])) for j
                        in
                        range(self.M) for t in range(self.N))
                                      )
                elif len(self.in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      )
        else:
            if len(self.out_place[i]) > 0:
                if len(self.in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      - gp.quicksum(
                        self.Ksi[self.out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(self.out_place[i])) for
                        j in
                        range(self.M) for t in range(self.N))
                                      - gp.quicksum(
                        self.Eta[self.in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(self.in_place[i])) for j
                        in
                        range(self.M) for t in range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(self.K) for j in range(self.M) for t in
                        range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.R[self.out_place[i][a]][j][t] - rC1i[a][j][t]) ** 2 for a in range(len(self.out_place[i]))
                        for j in
                        range(self.M) for t in range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.R[self.in_place[i][a]][j][t] - rC2i[a][j][t]) ** 2 for a in range(len(self.in_place[i]))
                        for j in
                        range(self.M) for t in range(self.N))
                                      )
                elif len(self.in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      - gp.quicksum(
                        self.Ksi[self.out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(self.out_place[i])) for
                        j in
                        range(self.M) for t in range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(self.K) for j in range(self.M) for t in
                        range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.R[self.out_place[i][a]][j][t] - rC1i[a][j][t]) ** 2 for a in range(len(self.out_place[i]))
                        for j in
                        range(self.M) for t in range(self.N))
                                      )
            elif len(self.out_place[i]) == 0:
                if len(self.in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in
                        range(self.N))
                                      - gp.quicksum(
                        self.Eta[self.in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(self.in_place[i])) for j
                        in
                        range(self.M) for t in range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(self.K) for j in range(self.M) for t in
                        range(self.N))
                                      + self.rho / 2 * gp.quicksum(
                        (self.R[self.in_place[i][a]][j][t] - rC2i[a][j][t]) ** 2 for a in range(len(self.in_place[i]))
                        for j in
                        range(self.M) for t in range(self.N))
                                      )
                elif len(self.in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(self.H[i][j] * vi[j][t] for j in range(self.M) for t in range(self.N))
                                      + gp.quicksum(self.P[i][t] * ui[t] for t in range(self.N))
                                      - gp.quicksum(
                        self.Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(self.K) for j in
                        range(self.M) for t in range(self.N))
                        + self.rho / 2 * gp.quicksum(
                            (self.X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(self.K) for j in range(self.M) for t
                            in
                            range(self.N)))

        prob.optimize()

        if len(self.out_place[i]) > 0:
            if len(self.in_place[i]) > 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X, rC1i.X, rC2i.X]
            elif len(self.in_place[i]) == 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X, rC1i.X]
        elif len(self.out_place[i]) == 0:
            if len(self.in_place[i]) > 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X, rC2i.X]
            elif len(self.in_place[i]) == 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X]

    # z problem
    def model_item_global(self):
        self.X = np.maximum(0, np.sum(self.rho * self.XC - self.Mu, axis=1)) / (self.rho * self.K)

        for i in range(self.K):
            for j in range(self.M):
                for t in range(self.N):
                    self.xx[i, :, j, t] = self.X[i][j][t]

        # analytical solution for an unconstrained quadratic program
        # can try 2n solutions if the z variables are integer
        self.R = np.maximum(0, self.RC1 + self.RC2 - (self.Ksi + self.Eta) / self.rho) / 2

        if self.ite > 0:
            # Update dual variables
            self.Mu += 1.6 * self.rho * (self.xx - self.XC)
            self.Ksi += 1.6 * self.rho * (self.R - self.RC1)
            self.Eta += 1.6 * self.rho * (self.R - self.RC2)

    def comp_obj(self):
        # obtain the objective value when converged
        VV = np.sum(self.V, axis=2)
        ob = np.sum(np.multiply(self.P, self.U)) + np.sum(np.multiply(self.H, VV))

        return ob

"""
Alias for instance method that allows the method to be called in a multiprocessing pool
"""
'''
def model_var_setup_copy(obj, arg):
    return obj.model_var_setup(arg)

def model_item_setObj_copy(obj, arg):
    return obj.model_item_setObj(arg)
'''

def main():
    # 迭代结果
    pri_re = []
    d_re = []
    # gbv.time_model = []
    # gbv.time_op = []
    # gbv.o_err = []  # objective relative error
    # gbv.r_err = []  # solution relative error

    MaxIte = 2000
    ep_abs = 1e-2
    ep_rel = 1e-4

    # 初始化primal/dual residual及primal/dual tolerance
    pr = 10
    dr = 10
    p_tol = 1e-5
    d_tol = 1e-5

    # 随机生成数据和建立变量
    dim = item_dim(1)

    '''use functools.partial to create a new method that always has the
    MyClass object passed as its first argument'''
    '''
    _bound_model_var_setup_copy = functools.partial(model_var_setup_copy, dim)
    _bound_model_item_setObj_copy = functools.partial(model_item_setObj_copy, dim)
    '''

    # p,n for the computation of primal/dual tolerance
    sqrt_dim_p = math.sqrt(dim.M * dim.N * (dim.K ** 2 + dim.A * 2))
    sqrt_dim_n = sqrt_dim_p

    # 开多个进程
    # pool = mp.Pool(processes=min(mp.cpu_count(), dim.K))
    pool = pp.ThreadPool(min(cpu_count(), dim.K))

    # 为每个item首先建立一个model存起来
    lists = list(range(dim.K))
    # tem = pool.map(dim.model_var_setup, lists)
    tem = pool.map(dim.model_var_setup, lists)
    for i in lists:
        dim.item_model[i] = tem[i][0]  # tem[i][0]为item i(i=0,...,K-1)的model
        dim.item_var[i] = tem[i][1:]  # tem[i][1:]为item i(i=0,...,K-1)的变量的列表

    # 初始化变量
    time_start2 = time.time()

    # x-subproblem
    tem = pool.map(dim.model_item_setObj, lists)
    for i in lists:
        dim.U[i] = tem[i][0]
        dim.S[i] = tem[i][1]
        dim.Z[i] = tem[i][2]
        dim.V[i] = tem[i][3]
        dim.YUI[i] = tem[i][4]
        dim.YUO[i] = tem[i][5]
        dim.XC[:, i, :, :] = tem[i][6]
        if len(dim.out_place[i]) > 0:
            if len(dim.in_place[i]) > 0:
                dim.RC1[dim.out_place[i]] = tem[i][7]
                dim.RC2[dim.in_place[i]] = tem[i][8]
            elif len(dim.in_place[i]) == 0:
                dim.RC1[dim.out_place[i]] = tem[i][7]
        elif len(dim.out_place[i]) == 0:
            if len(dim.in_place[i]) > 0:
                dim.RC2[dim.in_place[i]] = tem[i][7]

    # z-subproblem
    dim.model_item_global()

    # begin loop
    while pr > p_tol or dr > d_tol:
        dim.ite += 1
        if dim.ite > MaxIte:
            break

        # x-subproblem
        tem = pool.map(dim.model_item_setObj, lists)
        for i in lists:
            dim.U[i] = tem[i][0]
            dim.S[i] = tem[i][1]
            dim.Z[i] = tem[i][2]
            dim.V[i] = tem[i][3]
            dim.YUI[i] = tem[i][4]
            dim.YUO[i] = tem[i][5]
            dim.XC[:, i, :, :] = tem[i][6]
            if len(dim.out_place[i]) > 0:
                if len(dim.in_place[i]) > 0:
                    dim.RC1[dim.out_place[i]] = tem[i][7]
                    dim.RC2[dim.in_place[i]] = tem[i][8]
                elif len(dim.in_place[i]) == 0:
                    dim.RC1[dim.out_place[i]] = tem[i][7]
            elif len(dim.out_place[i]) == 0:
                if len(dim.in_place[i]) > 0:
                    dim.RC2[dim.in_place[i]] = tem[i][7]

        # pool.map(dim.model_item_setObj, lists)

        # 存储z^k
        xp = dim.X
        rp = dim.R

        # z-subproblem
        dim.model_item_global()

        xx_p = np.zeros((dim.K, dim.K, dim.M, dim.N))
        for i in range(dim.K):
            for j in range(dim.M):
                for t in range(dim.N):
                    xx_p[i, :, j, t] = xp[i][j][t]
        # primal residual
        tem1 = (dim.R - dim.RC1).reshape((-1, 1))
        tem2 = (dim.R - dim.RC2).reshape((-1, 1))
        tem3 = (dim.xx - dim.XC).reshape((-1, 1))
        tem = np.concatenate((tem1, tem2, tem3), axis=0)
        pr = LA.norm(tem)

        # dual residual
        tem1 = (dim.R - rp).reshape((-1, 1))
        tem2 = (dim.xx - xx_p).reshape((-1, 1))
        tem = np.concatenate((tem1, tem1, tem2), axis=0)
        dr = dim.rho * LA.norm(tem)

        # primal tolerance
        A_x = np.concatenate((dim.XC.reshape(-1, 1), dim.RC1.reshape(-1, 1), dim.RC2.reshape(-1, 1)), axis=0)
        B_z = np.concatenate((dim.xx.reshape((-1, 1)), dim.R.reshape((-1, 1)), dim.R.reshape((-1, 1))), axis=0)
        p_tol = sqrt_dim_p * ep_abs + ep_rel * max(LA.norm(A_x), LA.norm(B_z))

        # dual tolerance
        A_y = np.concatenate((dim.Mu.reshape((-1, 1)), dim.Ksi.reshape((-1, 1)), dim.Eta.reshape((-1, 1))), axis=0)
        d_tol = sqrt_dim_n * ep_abs + ep_rel * LA.norm(A_y)

        # adaptively update penalty
        if pr > 10 * dr:
            dim.rho *= 2
        elif dr > 10 * pr:
            dim.rho /= 2

        # 每十次迭代输出一次结果
        if dim.ite % 10 == 0:
            # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
            # gbv.o_err.append(temo_error)

            pri_re.append(pr)
            d_re.append(dr)
            '''
            tem_A = np.concatenate(
                (gbv.X.reshape(-1, 1), gbv.S.reshape(-1, 1), gbv.Z.reshape(-1, 1), gbv.R.reshape(-1, 1),
                 gbv.V.reshape(-1, 1), gbv.U.reshape(-1, 1), gbv.YUI.reshape(-1, 1),
                 gbv.YUO.reshape(-1, 1)), axis=0)
            tem_G = np.concatenate((gbv.X_E.reshape(-1, 1), gbv.S_E.reshape(-1, 1), gbv.Z_E.reshape(-1, 1),
                                    gbv.R_E.reshape(-1, 1), gbv.V_E.reshape(-1, 1), gbv.U_E.reshape(-1, 1),
                                    gbv.YUI_E.reshape(-1, 1), gbv.YUO_E.reshape(-1, 1)), axis=0)

            tem_error = LA.norm(tem_A - tem_G) / LA.norm(tem_G) * 100
            gbv.r_err.append(tem_error)  # 相对误差...%

            wr = "Iteration : " + str(gbv.ite) + '\n' + "Objective : " + str(
                tem_ob) + '\n' + "Primal residual : " + str(
                pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "Relative objective error: " + str(
                temo_error) + '\n' + "Relative solution error: " + str(
                tem_error) + '\n' + '\n'
            '''
            wr = "Iteration : " + str(dim.ite) + '\n' + '\n' + "Primal residual : " + str(
                pr) + '\n' + "Dual residual : " + str(dr) + '\n' + '\n'
            wr_s = open('LS_withstart.txt', 'a')
            wr_s.write(wr)
            wr_s.close()
    pool.close()
    pool.join()
    pool.clear()

    time_end2 = time.time()
    time_A = time_end2 - time_start2

    Ob = dim.comp_obj()

    if dim.ite % 10 != 0:
        # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
        # gbv.o_err.append(temo_error)

        pri_re.append(pr)
        d_re.append(dr)
        '''
        tem_A = np.concatenate((gbv.X.reshape(-1, 1), gbv.S.reshape(-1, 1), gbv.Z.reshape(-1, 1), gbv.R.reshape(-1, 1),
                                gbv.V.reshape(-1, 1), gbv.U.reshape(-1, 1), gbv.YUI.reshape(-1, 1),
                                gbv.YUO.reshape(-1, 1)), axis=0)
        tem_G = np.concatenate((gbv.X_E.reshape(-1, 1), gbv.S_E.reshape(-1, 1), gbv.Z_E.reshape(-1, 1),
                                gbv.R_E.reshape(-1, 1), gbv.V_E.reshape(-1, 1), gbv.U_E.reshape(-1, 1),
                                gbv.YUI_E.reshape(-1, 1), gbv.YUO_E.reshape(-1, 1)), axis=0)
        tem_error = LA.norm(tem_A - tem_G) / LA.norm(tem_G) * 100
        gbv.r_err.append(tem_error)  # 相对误差...%

        wr = "Iteration : " + str(gbv.ite) + '\n' + "Objective : " + str(tem_ob) + '\n' + "Primal residual : " + str(
            pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "Relative objective error: " + str(
            temo_error) + '\n' + "Relative solution error: " + str(
            tem_error) + '\n\n' + "Gurobi extensive formula obj: " + str(
            gbv.ob) + '\n' + "Gurobi extensive formula time: " + str(
            gbv.time_e) + '\n' + "ADMM time: " + str(gbv.time_A) + '\n' + "Finished!"
        '''
        wr = "Iteration : " + str(dim.ite) + '\n' + "Primal residual : " + str(
            pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "Objective : " + str(Ob) + '\n' + "ADMM time: " + str(
            time_A) + '\n' + '\n' + "Finished!"

        wr_s = open('LS_withstart.txt', 'a')
        wr_s.write(wr)
        wr_s.close()
    else:
        '''
        wr = '\n' + "Gurobi extensive formula obj: " + str(
            gbv.ob) + '\n' + "Gurobi extensive formula time: " + str(
            gbv.time_e) + '\n' + "ADMM time: " + str(
            gbv.time_A) + '\n' + '\n' + "Finished!"
        '''
        wr = '\n' + "Objective : " + str(Ob) + '\n' + "ADMM time: " + str(
            time_A) + '\n' + '\n' + "Finished!"
        wr_s = open('LS_withstart.txt', 'a')
        wr_s.write(wr)
        wr_s.close()

    nz = len(pri_re)
    pri_re = np.array(pri_re)
    d_re = np.array(d_re)
    # gbv.o_err = np.array(gbv.o_err)
    # gbv.r_err = np.array(gbv.r_err)

    plt.figure(1)
    plt.plot(range(nz), pri_re, c='red', marker='*', linestyle='-', label='Primal residual')
    plt.plot(range(nz), d_re, c='green', marker='+', linestyle='--', label='Dual residual')
    plt.legend()
    plt.title("primal/dual residual")  # 设置标题
    plt.xlabel("iterations(*10)")  # 设置x轴标注
    plt.ylabel("primal/dual residual")  # 设置y轴标注
    plt.savefig("LS1.png")

    '''
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


if __name__ == "__main__":
    main()
