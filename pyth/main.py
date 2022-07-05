#!/usr/bin/env python3

# import gurobipy as gp
# from gurobipy import GRB
import numpy as np
# import cvxpy as cp

from GlobalVariable import globalVariables as gbv
from readin import *
from solve_item import *
from solve_global import *
from sp_item import *
from sp_global import *
from comp_obj import *
import matplotlib.pyplot as plt

# cvx_solver mosek
# cvx_save_prefs

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

# Copies
gbv.UC = np.zeros((gbv.K, gbv.K, gbv.N))  # u_{i'(i)t}
gbv.XC = np.zeros((gbv.K, gbv.M, gbv.K, gbv.N))   # x_{i'j(i)t}
gbv.RC = np.zeros((gbv.A, gbv.M, gbv.K, gbv.N))   # r_{aj(i)t}

# Dual variables
gbv.Lam = np.zeros((gbv.K, gbv.K, gbv.N))
gbv.Mu = np.zeros((gbv.K, gbv.K, gbv.M, gbv.N))
gbv.Ksi = np.zeros((gbv.A, gbv.M, gbv.K, gbv.N))

# item-based decomposition
# Input
gbv.MaxIte = 1000
gbv.rho = 1e12
# Initialization
gbv.ite = 0
# item (6)
for i in range(gbv.K):
    sp_item(i)
# central planner (7)
sp_global()

gbv.ite += 1

# Iterations
gbv.Ob = np.zeros(gbv.MaxIte)
count = 0

while gbv.ite <= gbv.MaxIte:
    # item (3)
    for i in range(gbv.K):
        sp_item(i)

    # central planner (4) & (6)
    sp_global()

    if gbv.ite % 10 == 1:
        gbv.Ob[count] = comp_obj()
        count += 1

    gbv.ite += 1

nz = int(np.count_nonzero(gbv.Ob))
gbv.Ob = gbv.Ob[0:nz]
plt.plot(range(nz), gbv.Ob, color="red", zorder=1)


