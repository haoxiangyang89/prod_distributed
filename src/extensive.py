# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:58:01 2022

@author: haoxiang
"""

'''
Solve the extensive formulation for the multi-period production problem
'''

from gurobipy import gurobipy as gp
from gurobipy import GRB

def extensive_prob(solve_option = True, relax_option = False):
    # set up the extensive formulation
    global gbv
    prob = gp.Model("extensive_form")
    prob.setParam("Threads", 1)

    # set up model parameters (M: plant, T: time, L: transit,
    u = prob.addVars(gbv.item_list, gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    s = prob.addVars(gbv.item_list, gbv.transit_list, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    z = prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, name="z")  # z_{ijt} for j,t
    v = prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="v")  # v_{ijt} for j,t
    yUI = prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUO = prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, name="yo")  # y^{O}_{ijt} for j,t
    xC = prob.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t

    # initial condition setup
    for i in gbv.item_list:
        u[i, 0] = 0.0  # initial unmet demand set to 0
    for i in gbv.item_list:
        for l in gbv.transit_list:
            for t in range(min(gbv.period_list) - gbv.transit_time[(i,) + l], min(gbv.period_list)):
                s[(i,) + l + (t,)] = 0.0    # initial transportation set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            v[i, j, 0] = gbv.init_inv[i, j]     # initial inventory set to given values
    for i, j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i, j], min(gbv.period_list)):
            xC[i, j, t] = 0.0   # initial production set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if not ((i, j) in gbv.prod_key):
                for t in gbv.period_list:
                    xC[i, j, t] = 0.0   # production for non-existent item-plant pair set to 0

    rC = prob.addVars(gbv.alt_list, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # add constraints for the extensive formulation
    prob.addConstrs((u[i, t] - u[i, t - 1] + gp.quicksum(z[i, j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[i, t] for i in gbv.item_list for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((v[i, j, t] - v[i, j, t - 1] - yUI[i, j, t] + yUO[i, j, t] == 0 \
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list), name='inventory')
    prob.addConstrs((yUI[i, j, t] == gp.quicksum(s[(i,) + l + (t - gbv.transit_time[(i,) + l],)] for l in gbv.transit_list if l[1] == j) +
                     xC[i, j, t - gbv.lead_time[i, j]] + gbv.external_purchase[i, j, t] \
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                    name='input_item')
    prob.addConstrs((yUO[i, j, t] == gp.quicksum(s[(i,) + l + (t,)] for l in gbv.transit_list if l[0] == j) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xC[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == i) + z[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i)) -
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                    name="output_item")
    prob.addConstrs((gp.quicksum(gbv.unit_cap[i_iter, j_iter, ct_iter] * xC[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
                                 if (j_iter == j) and (ct == ct_iter)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list), name='capacity')
    prob.addConstrs((rC[jta] <= v[i, jta[0], jta[1] - 1] for i in gbv.item_list for jta in gbv.alt_list if jta[2][0] == i),
                    name='r_ub')
    prob.addConstrs((yUO[i, j, t] <= v[i, j, t - 1] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')

    if not(relax_option):
        # if we require an integer number of batches
        wi = prob.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t
        prob.addConstrs(
            (xC[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
            name='batch')
        prob.addConstrs(
            (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
            name='w_ub')
    else:
        # if we can relax the integer constraint
        prob.addConstrs(
            (xC[i, j, t] <= gbv.max_prod[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
            name='capacity')

    # set up the subproblem specific objective
    prob.setObjective(
        gp.quicksum(gbv.holding_cost[i] * v[i, j, t] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list) +
        gp.quicksum(gbv.penalty_cost[i] * u[i, t] for i in gbv.item_list for t in gbv.period_list), GRB.MINIMIZE)

    prob.update()

    if solve_option:
        prob.optimize()
        # collect the solution and objective value
        return prob.objVal
    else:
        return prob, [u, s, z, v, yUI, yUO, xC]