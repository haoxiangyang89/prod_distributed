'''
problem formulations and solving process
'''

from gurobipy import gurobipy as gp
from gurobipy import GRB
from src.variables import global_var
import itertools
import collections
import time
import numpy as np

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def x_item(i):
    '''
    i: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    global gbv
    prob = gp.Model("item_{}".format(i))
    prob.setParam("Threads", 1)

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb = 0.0, name="u")  # u_{it} for t, unmet demand
    ui[0] = 0.0   # initial unmet demand set to 0
    si = prob.addVars(gbv.transit_list, gbv.period_list, lb = 0.0, name="s")  # s_{ilt} for l,t
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[(i,) + l], min(gbv.period_list)):
            si[l+(t,)] = 0.0
    zi = prob.addVars(gbv.plant_list, gbv.period_list, name="z")  # z_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name="v")  # v_{ijt} for j,t
    for j in gbv.plant_list:
        vi[j,0] = gbv.init_inv[i,j]
    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = prob.addVars(gbv.plant_list, gbv.period_list, name="yo")  # y^{O}_{ijt} for j,t
    xCi = prob.addVars(gbv.prod_key, gbv.period_list, lb = 0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    wi = prob.addVars(gbv.prod_key, gbv.period_list, vtype = GRB.INTEGER, lb = 0.0, name= "w")  # x_{ijt} for i,j,t (x copy)
    for i,j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i,j], min(gbv.period_list)):
            xCi[i,j,t] = 0.0
    for j in gbv.plant_list:
        if not((i,j) in gbv.prod_key):
            for t in gbv.period_list:
                xCi[i,j,t] = 0.0

    rCi = prob.addVars(gbv.alt_list, lb = 0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi[j,t] for j in gbv.plant_list) \
                     == gbv.real_demand[i,t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j,t] - vi[j,t - 1] - yUIi[j,t] + yUOi[j,t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')
    prob.addConstrs((yUIi[j,t] == gp.quicksum(si[l + (t - gbv.transit_time[(i,)+l],)] for l in gbv.transit_list) +
                     xCi[i,j,t - gbv.lead_time[i,j]] + gbv.external_purchase[i,j,t] \
                     for j in gbv.plant_list for t in gbv.period_list if (i,j) in gbv.prod_key), name='input_item')
    prob.addConstrs((yUOi[j,t] == gp.quicksum(si[l + (t,)] for l in gbv.transit_list) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xCi[bk[0],j,t] for bk in gbv.bom_key[j] if bk[1] == i) + zi[j,t] +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
                     for j in gbv.plant_list for t in gbv.period_list if (i,j) in gbv.prod_key), name="output_item")
    prob.addConstrs((gp.quicksum(gbv.unit_cap[ii,jj] * xCi[ii,jj,t] for ii,jj in gbv.unit_cap.keys() \
                                 if (jj == j) and (gbv.item_set[ii] == ct)) <= gbv.max_cap[ct,j][t]
                     for j in gbv.plant_list for t in gbv.period_list for ct in gbv.set_list if (i,j) in gbv.prod_key), name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0],jta[1]] for jta in gbv.alt_list if jta[2][0] == i), name='r_ub')
    prob.addConstrs((yUOi[j,t] <= vi[j,t] for j in gbv.plant_list for t in gbv.period_list), name='yo_ub')
    prob.addConstrs((xCi[i,j,t] == wi[i,j,t] * gbv.lot_size[i,j] for i,j in gbv.prod_key for t in gbv.period_list), name='batch')
    prob.addConstrs((wi[i,j,t] <= gbv.max_prod[i,j]/gbv.lot_size[i,j] for i,j in gbv.prod_key for t in gbv.period_list), name='w_ub')

    # set up the subproblem specific objective
    theta = prob.addVar(lb = 0.0, name = 'theta')
    prob.addConstr(theta == gp.quicksum(gbv.holding_cost[i] * vi[j,t] for j in gbv.plant_list for t in gbv.period_list) +
                      gp.quicksum(gbv.penalty_cost[i] * ui[t] for t in gbv.period_list), name = 'obj_local')

    prob.update()
    return prob

def x_item_init():
    '''
    initialize the global variables for the item decomposition
    return: global variables with their primal/dual values initialized
    '''

    global gbv
    global_vars = []

    # initialize the variable x
    x_keys = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    x_var = global_var("x", x_keys)
    x_var.init_dual(gbv.item_list)

    # initialize the variable r
    r_keys = ["r[{},{},{}]".format(*r_rel) for r_rel in gbv.alt_list]
    r_var = global_var("r", r_keys)
    r_var.init_dual(gbv.item_list)

    global_vars = [x_var, r_var]
    return global_vars

def x_item_dual_obtain_name(i):
    '''
    for each item subproblem, obtain the index of the global variables that require value passing
    return: a list of list containing global variables' indices of item subproblem i
    '''
    global gbv

    x_prod_list = [item for item in gbv.prod_key if item[1] in gbv.item_plant[i]]
    # obtain the variable names: x
    x_name = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in x_prod_list for t in gbv.period_list]

    r_list = [item for item in gbv.alt_list if i in item[2]]
    # obtain the variable names: r
    r_name = ["r[{},{},{}]".format(*r_rel) for r_rel in r_list]

    global_names_i = [x_name, r_name]

    return global_names_i

def x_plant(gbv):
    pass

def x_time(gbv):
    pass

def dual_obtain(global_vars, i, global_names_i):
    global_ind = []

    assert len(global_names_i) == len(global_vars)

    # obtain the global variable index
    for varI in range(len(global_vars)):
        var_ind = [global_vars[varI].keys.index(global_names_i[varI][j]) for j in range(len(global_names_i[varI]))]
        global_ind.append(var_ind)

    return global_ind


def x_solve(i, global_ind_i, global_vars, rho):
    '''
    i: the index of sub-problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, prob_list

    prob = prob_list[i]

    # set the linear expression of the local part for the objective
    obj_local_part = prob.getVarByName("theta")

    time_init = time.time()

    # objective function
    prob.setObjective(obj_local_part -
                      gp.quicksum(global_vars[gvar_ind].dual[i][vi] * prob.getVarByName(global_vars[gvar_ind].keys[vi]) \
                                  for gvar_ind in range(len(global_vars)) \
                                  for vi in global_ind_i[gvar_ind]) + # add all global variables for relaxation part
                      rho / 2 * gp.quicksum((global_vars[gvar_ind].value[vi] - prob.getVarByName(global_vars[gvar_ind].keys[vi])) ** 2 \
                                            for gvar_ind in range(len(global_vars)) \
                                            for vi in global_ind_i[gvar_ind]), GRB.MINIMIZE)

    # solve the problem
    prob.update()

    time_obj = time.time()
    time_setObj = time_obj - time_init
    prob.optimize()
    time_solution = time.time() - time_obj
    print("Objective setup time: {} sec; Solution time: {} sec.".format(time_setObj, time_solution))

    # obtain the solutions and output results
    local_output = []
    for gvar_ind in range(len(global_vars)):
        local_val = []
        for vi in global_ind_i[gvar_ind]:
            local_val.append(prob.getVarByName(global_vars[gvar_ind].keys[vi]).X)
        local_output.append(local_val)

    # obtain the solution and return the subproblem solution
    return local_output

def z_solve(local_results, global_ind, global_vars, rho):
    '''
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    I = len(local_results)

    # update the global variables
    for gvar_ind in range(len(global_vars)):
        global_array = np.zeros(len(global_vars[gvar_ind].value))
        for i in range(I):
            local_array = np.zeros(len(global_vars[gvar_ind].value))
            local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]
            global_array += 1 / (rho * I) * (rho * local_array - global_vars[gvar_ind].dual[i])
        global_vars[gvar_ind].value = np.maximum(global_array,0.0)

def pi_solve(local_results, global_ind, global_vars, rho, adj_coeff = 1.6):
    '''
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    I = len(local_results)

    # update the dual variables
    for gvar_ind in range(len(global_vars)):
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                list_ind = global_ind[i][gvar_ind]
                global_vars[gvar_ind].dual[i][list_ind] += adj_coeff * rho * (global_vars[gvar_ind].value[list_ind] - np.array(local_results[i][gvar_ind][list_ind]))
