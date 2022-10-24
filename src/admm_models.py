'''
problem formulations and solving process
'''

from gurobipy import gurobipy as gp
from gurobipy import GRB

def x_item(gbv, i):
    prob = gp.Model("item_{}".format(i))

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb = 0.0, name = "u")  # u_{it} for t, unmet demand
    ui[0] = 0.0   # initial unmet demand set to 0
    si = prob.addVars(gbv.transit_list, gbv.period_list, lb = 0.0, name = "s")  # s_{ilt} for l,t
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[(i,) + l], min(gbv.period_list)):
            si[l+(t,)] = 0.0
    zi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name = "z")  # z_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name = "v")  # v_{ijt} for j,t
    for j in gbv.plant_list:
        vi[j,0] = gbv.init_inv[i,j]
    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name = "yi")  # y^{I}_{ijt} for j,t
    yUOi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name = "yo")  # y^{O}_{ijt} for j,t
    xCi = prob.addVars(gbv.prod_key, gbv.period_list, lb = 0.0, name = "x")  # x_{ijt} for i,j,t (x copy)
    wi = prob.addVars(gbv.prod_key, gbv.period_list, vtype = GRB.INTEGER, lb = 0.0, name= "w")  # x_{ijt} for i,j,t (x copy)
    for i,j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i,j], min(gbv.period_list)):
            xCi[i,j,t] = 0.0
    for j in gbv.plant_list:
        if not((i,j) in gbv.prod_key):
            for t in gbv.period_list:
                xCi[i,j,t] = 0.0

    rCi = prob.addVars(gbv.alt_list, lb = 0.0, name = "r")  # r_{ajt} for a=(i,i')

    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi[j,t] for j in gbv.plant_list) \
                     == gbv.real_demand[i,t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j,t] - vi[j,t - 1] - yUIi[j,t] + yUOi[j,t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')
    prob.addConstrs((yUIi[j,t] == gp.quicksum(si[l + (t - gbv.transit_time[(i,)+l],)] for l in gbv.transit_list) +
                     xCi[i,j,t - gbv.lead_time[i,j]] + gbv.external_purchase[i,j,t] \
                     for j in gbv.plant_list for t in gbv.period_list), name='input_item')
    prob.addConstrs((yUOi[j,t] == gp.quicksum(si[l + (t,)] for l in gbv.transit_list) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xCi[bk[0],j,t] for bk in gbv.bom_key[j] if bk[1] == i) + zi[j,t] +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
                     for j in gbv.plant_list for t in gbv.period_list), name = "output_item")
    prob.addConstrs((gp.quicksum(gbv.unit_cap[ii,jj] * xCi[ii,jj,t] for ii,jj in gbv.unit_cap.keys() \
                                 if (jj == j) and (gbv.item_set[ii] == ct)) <= gbv.max_cap[ct,j][t]
                     for j in gbv.plant_list for t in gbv.period_list for ct in gbv.set_list), name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0],jta[1]] for jta in gbv.alt_list if jta[2][0] == i), name = 'r_ub')
    prob.addConstrs((yUOi[j,t] <= vi[j,t] for j in gbv.plant_list for t in gbv.period_list), name = 'yo_ub')
    prob.addConstrs((xCi[i,j,t] == wi[i,j,t] * gbv.lot_size[i,j] for i,j in gbv.prod_key for t in gbv.period_list), name = 'batch')
    prob.addConstrs((wi[i,j,t] <= gbv.max_prod[i,j]/gbv.lot_size[i,j] for i,j in gbv.prod_key for t in gbv.period_list), name = 'w_ub')

    prob.update()
    return prob

def x_plant(gbv):
    pass

def x_time(gbv):
    pass

def x_prob(index, x_model, var_pass, var_value, dual_value):
    # change the objective function of x_model

    # solve the x_prob

    # return the solution and the obejctive value

    a = 1

def z_prob(z_model, var_pass, var_value):
    pass

def pi_prob(pi_model, var_pass, var_value, step_size):
    pass

