#!/usr/bin/env python3

# import gurobipy as gp
# from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
# import cvxpy as cp

from GlobalVariable import globalVariables as gbv
from readin import *
# from solve_item import *
# from solve_global import *
from sp_item import *
# from sp_plant import *
# from sp_time import *
from sp_global import *
from comp_obj import *
from Extensive import *
import matplotlib.pyplot as plt

# cvx_solver mosek
# cvx_save_prefs
'''
# parameters
[item_list, holding_cost] = input_item("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_item.csv")
gbv.K = len(item_list)

plant_list = input_plant("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_plant.csv")
gbv.M = len(plant_list)

period_delay = input_periods("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_periods.csv")
gbv.N = len(period_delay.keys())
gbv.epsilonUD = np.zeros(gbv.N)
for i in range(gbv.N):
    gbv.epsilonUD[i] = period_delay.get(i+1)

[bom_key, bom_dict] = input_bom("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_bom.csv")
gbv.cE = bom_key
gbv.E = len(gbv.cE)
gbv.q = np.zeros(gbv.E)
for e in range(gbv.E):
    tem = bom_dict.get(bom_key[e])
    for key, value in tem.items():
        gbv.q[e] = value

alt_multi = input_item_alternate("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_alternate_item.csv")
gbv.cA = []
for key in alt_multi:
    if key[0:2] not in gbv.cA:
        gbv.cA.append(key[0:2])
gbv.A = len(gbv.cA)

[unit_cap, unit_cap_type] = input_unit_capacity("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_unit_capacity.csv")
gbv.nu = np.zeros((gbv.K, gbv.M))
for key, value in unit_cap.items():
    gbv.nu[key[0]-1][key[1]-1] = value

[prod_key, lot_size, lead_time, holding_cost, prod_cost, min_prod, max_prod] = input_production("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_production.csv")
gbv.m = np.zeros(gbv.K)
for key, value in lot_size.items():
    gbv.m[key[0]-1] = value

gbv.dtUP = np.zeros((gbv.K, gbv.M))
for key, value in lead_time.items():
    gbv.dtUP[key[0]-1][key[1]-1] = value

gbv.H = np.zeros((gbv.K, gbv.M))
for key, value in holding_cost.items():
    gbv.H[key[0]-1][key[1]-1] = value

gbv.wLB = np.zeros((gbv.K, gbv.M))
for key, value in min_prod.items():
    gbv.wLB[key[0]-1][key[1]-1] = value

gbv.wUB = np.zeros((gbv.K, gbv.M))
for key, value in max_prod.items():
    gbv.wUB[key[0]-1][key[1]-1] = value

[transit_time, transit_cost] = input_transit("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_transit.csv")
gbv.cL = []
gbv.dtUL = np.zeros((gbv.K, 100))
tem = -1
for key, value in transit_time.items():
    if key[1:3] not in gbv.cL:
        gbv.cL.append(key[1:3])
        tem += 1
    gbv.dtUL[key[0]-1][tem] = value
gbv.L = len(gbv.cL)
gbv.dtUL = gbv.dtUL[:, range(gbv.L)]

init_inv = input_init_inv("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_inv.csv")
gbv.v0 = np.zeros((gbv.K, gbv.M))
for key, value in init_inv.items():
    gbv.v0[key[0]-1][key[1]-1] = value

[period_list, cap_key, max_cap] = input_capacity("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_max_capacity.csv")
gbv.C = np.zeros((gbv.M, gbv.N))
for key, value in max_cap.items():
    gbv.C[key[0]-1][key[1]-1] = value

external_purchase = input_po("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_po.csv")
gbv.Q = np.zeros((gbv.K, gbv.M, gbv.N))
for key, value in external_purchase.items():
    gbv.Q[key[0]-1][key[1]-1][key[2]-1] = value

[real_demand, forecast_demand] = input_demand("/Users/macbookpro/PycharmProjects/pythonProject1/prod_distributed-main/data/pilot_test/dm_df_demand.csv")
gbv.D = np.zeros((gbv.K, gbv.N))
for key, value in real_demand.items():
    gbv.D[key[0]-1][key[1]-1] = value

gbv.P = 200 * np.ones((gbv.K, gbv.N))
'''
gbv.N = 2  # T:time
gbv.M = 3  # J:plant
gbv.K = 3  # I:item
gbv.L = 2  # l
gbv.E = 1
gbv.A = 1  # a

gbv.H = np.array([[2, 1, 3]])
gbv.H = np.tile(gbv.H, (gbv.K, 1))
gbv.P = 200 * np.ones((gbv.K, gbv.N))

gbv.D = np.array([[5, 7], [10, 8], [6, 4]])

gbv.epsilonUD = np.ones(gbv.N) * 0.05
gbv.epsilonUI = 0.8

gbv.v0 = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])

gbv.nu = np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]])

gbv.C = np.array([[50], [50], [50]])
gbv.C = np.tile(gbv.C, (1, gbv.N))

gbv.cL = np.array([(1, 2), (2, 3)])
gbv.cE = np.array([(2, 1)])
gbv.cA = np.array([(3, 2)])

gbv.dtUL = np.ones((gbv.K, gbv.L))  # delta t^L _il

gbv.dtUP = np.zeros((gbv.K, gbv.M))  # delta t^P_ij

gbv.Q = np.zeros((gbv.K, gbv.M, gbv.N))

gbv.q = 2 * np.ones(gbv.E)

gbv.m = 2 * np.ones(gbv.K)  # m_i
gbv.wLB = 1 * np.ones((gbv.K, gbv.M))
gbv.wUB = 20 * np.ones((gbv.K, gbv.M))

# 每个item作为集合A中边的出端(i,i')或者入端(i',i)时，相应边在集合A中对应的序号
gbv.out_place = []
for i in range(gbv.K):
    tem = []
    for a in range(gbv.A):
        if gbv.cA[a][0] == i+1:
            tem.append(a)
    gbv.out_place.append(tem)

gbv.in_place = []
for i in range(gbv.K):
    tem = []
    for a in range(gbv.A):
        if gbv.cA[a][1] == i+1:
            tem.append(a)
    gbv.in_place.append(tem)

# extensive formula
gbv.X_E = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.S_E = np.zeros((gbv.K, gbv.L, gbv.N))
gbv.Z_E = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.R_E = np.zeros((gbv.A, gbv.M, gbv.N))
gbv.V_E = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.U_E = np.zeros((gbv.K, gbv.N))
gbv.YUI_E = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.YUO_E = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.XUb_E = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.W_E = np.zeros((gbv.K, gbv.M, gbv.N))

gbv.ob = solve_extensive()

# Primal Variables
gbv.X = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.S = np.zeros((gbv.K, gbv.L, gbv.N))
gbv.Z = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.R = np.zeros((gbv.A, gbv.M, gbv.N))
gbv.V = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.U = np.zeros((gbv.K, gbv.N))
gbv.YUI = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.YUO = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.XUb = np.zeros((gbv.K, gbv.M, gbv.N))
gbv.W = np.zeros((gbv.K, gbv.M, gbv.N))

gbv.MaxIte = 3000
gbv.rho = 1e4

# dim = input("Decomposition dimension")
dim = 'item'

# item-based decomposition
if dim == 'item':
    # Copies (item)
    # gbv.UC = np.zeros((gbv.K, gbv.K, gbv.N))  # u_{i'(i)t}
    gbv.XC = np.zeros((gbv.K, gbv.K, gbv.M, gbv.N))  # x_{i(i')jt}
    gbv.RC1 = np.zeros((gbv.A, gbv.M, gbv.N))  # r_{a(i)jt}
    gbv.RC2 = np.zeros((gbv.A, gbv.M, gbv.N))  # r_{a(i')jt}
    # Dual variables (item)
    # gbv.Lam = np.zeros((gbv.K, gbv.K, gbv.N))
    gbv.Mu = np.zeros((gbv.K, gbv.K, gbv.M, gbv.N))
    gbv.Ksi = np.zeros((gbv.A, gbv.M, gbv.N))
    gbv.Eta = np.zeros((gbv.A, gbv.M, gbv.N))

# plant-based decomposition
elif dim == 'plant':
    # Copies (plant)
    gbv.ZC = np.zeros((gbv.K, gbv.M, gbv.M, gbv.N))  # z_{ij(j')t}
    gbv.SC = np.zeros((gbv.K, gbv.L, gbv.M, gbv.N))  # s_{il(j)t}
    # Dual variables (plant)
    gbv.Lam = np.zeros((gbv.K, gbv.M, gbv.M, gbv.N))
    gbv.Eta = np.zeros((gbv.K, gbv.L, gbv.M, gbv.N))

# time-based decomposition
elif dim == 'time':
    # Copies (time)
    gbv.UC = np.zeros((gbv.K, gbv.N, gbv.N))     # u_{it(t')}
    gbv.SC = np.zeros((gbv.K, gbv.L, gbv.N, gbv.N))  # s_{ilt(t')}
    gbv.VC = np.zeros((gbv.K, gbv.M, gbv.N, gbv.N))     # v_{ijt(t')}
    gbv.XC = np.zeros((gbv.K, gbv.M, gbv.N, gbv.N))  # x_{ijt(t')}
    # Dual variables (time)
    gbv.Lam = np.zeros((gbv.K, gbv.N, gbv.N))
    gbv.Mu = np.zeros((gbv.K, gbv.L, gbv.N, gbv.N))
    gbv.Ksi = np.zeros((gbv.K, gbv.M, gbv.N, gbv.N))
    gbv.Eta = np.zeros((gbv.K, gbv.M, gbv.N, gbv.N))

# Initialization of decision variables
gbv.ite = 0

if dim == 'item':
    # item (6)
    for i in range(gbv.K):
        sp_item(i)
elif dim == 'plant':
    # plant (13)
    for j in range(gbv.M):
        sp_plant(j)
elif dim == 'time':
    # time (20)
    for t in range(gbv.N):
        sp_time(t)

sp_global(dim)

gbv.ite += 1

# Iterations
gbv.Ob = []
gbv.pri_re = []
gbv.d_re = []
gbv.p_tol = 1e-3
gbv.d_tol = 1e-3
pr = 10
dr = 10

while pr > gbv.p_tol or dr > gbv.d_tol:
#while gbv.ite <= gbv.MaxIte:
    if dim == 'item':
        # item (6)
        for i in range(gbv.K):
            sp_item(i)
    elif dim == 'plant':
        # plant (13)
        for j in range(gbv.M):
            sp_plant(j)
    elif dim == 'time':
        # time (20)
        for t in range(gbv.N):
            sp_time(t)

    #up = gbv.U
    xp = gbv.X
    rp = gbv.R

    sp_global(dim)

    # primal residual
    #tem = np.zeros((gbv.K, gbv.N))
    tem2 = []
    #for i in range(gbv.K):
    #    for t in range(gbv.N):
    #        tem[i][t] = LA.norm(gbv.U[i][t] - gbv.UC[i, :, t])
    #tem2.append(LA.norm(tem))
    tem = np.zeros((gbv.K, gbv.M, gbv.N))
    for i in range(gbv.K):
        for j in range(gbv.M):
            for t in range(gbv.N):
                tem[i][j][t] = LA.norm(gbv.X[i][j][t] - gbv.XC[i, :, j, t])
    tem2.append(LA.norm(tem))
    tem2.append(LA.norm(gbv.R - gbv.RC1))
    tem2.append(LA.norm(gbv.R - gbv.RC2))
    tem2 = np.array(tem2)
    pr = LA.norm(tem2)

    # dual residual
    tem3 = []
    tem4 = LA.norm(gbv.X - xp)
    for i in range(gbv.K):
        tem3.append(tem4)
    tem3.append(LA.norm(gbv.R - rp))
    tem3.append(LA.norm(gbv.R - rp))
    dr = gbv.rho * LA.norm(tem3)

    if pr > 10 * dr:
        gbv.rho *= 2
    elif dr > 10 * pr:
        gbv.rho /= 2

    if gbv.ite % 10 == 0:
        gbv.Ob.append(comp_obj())
        gbv.pri_re.append(pr)
        gbv.d_re.append(dr)

    gbv.ite += 1
    if gbv.ite > gbv.MaxIte:
        break

nz = len(gbv.Ob)
gbv.Ob = np.array(gbv.Ob)
gbv.pri_re = np.array(gbv.pri_re)
gbv.d_re = np.array(gbv.d_re[0:nz])
plt.plot(range(nz), gbv.Ob, c='blue', marker='o', linestyle=':', label='Objective')
plt.plot(range(nz), gbv.pri_re, c='red', marker='*', linestyle='-', label='Primal residual')
plt.plot(range(nz), gbv.d_re, c='green', marker='+', linestyle='--', label='Dual residual')