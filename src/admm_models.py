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
from numpy import linalg as LA

global gbv, prob_list, global_vars, global_ind

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def x_item(item_ind, relax_option = True):
    '''
    item_ind: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    prod_plant_i = [(i,j) for i in gbv.item_list for j in gbv.plant_list if \
                    ((j in gbv.item_plant[item_ind]) and (i,j) in gbv.prod_key) or ((i,item_ind) in gbv.bom_key[j])]
    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb = 0.0, name="u")  # u_{it} for t, unmet demand
    si = prob.addVars(gbv.transit_list, gbv.period_list, lb = 0.0, name="s")  # s_{ilt} for l,t
    zi = prob.addVars(gbv.plant_list, gbv.period_list, name="z")  # z_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name="v")  # v_{ijt} for j,t
    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = prob.addVars(gbv.plant_list, gbv.period_list, name="yo")  # y^{O}_{ijt} for j,t
    xCi = prob.addVars(prod_plant_i, gbv.period_list, lb = 0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = prob.addVars(prod_plant_i, gbv.period_list, vtype = GRB.INTEGER, lb = 0.0, name= "w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j,0] = gbv.init_inv[item_ind,j]     # initial inventory set to given values
    for i,j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i,j], min(gbv.period_list)):
            xCi[i,j,t] = 0.0   # initial production set to 0

    rCi = prob.addVars(alt_i, lb = 0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi[j,t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind,t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j,t] - vi[j,t - 1] - yUIi[j,t] + yUOi[j,t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')
    prob.addConstrs((yUIi[j,t] == gp.quicksum(si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in gbv.transit_list if l[1] == j) +
                     xCi[item_ind,j,t - gbv.lead_time[item_ind,j]] + gbv.external_purchase[item_ind,j,t] \
                     for j in gbv.plant_list for t in gbv.period_list if j in gbv.item_plant[item_ind]), name='input_item_prod')
    prob.addConstrs((yUIi[j,t] == gp.quicksum(si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in gbv.transit_list if l[1] == j) +
                     gbv.external_purchase[item_ind,j,t] for j in gbv.plant_list for t in gbv.period_list if not(j in gbv.item_plant[item_ind])),
                    name='input_item_non_prod')
    prob.addConstrs((yUOi[j,t] == gp.quicksum(si[l + (t,)] for l in gbv.transit_list if l[0] == j) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xCi[bk[0],j,t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + zi[j,t] +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list), name="output_item")
    prob.addConstrs((gp.quicksum(gbv.unit_cap[ii,jj] * xCi[ii,jj,t] for ii,jj in gbv.unit_cap.keys() \
                                 if (jj == j) and (gbv.item_set[ii] == ct)) <= gbv.max_cap[ct,j][t]
                     for j in gbv.plant_list for t in gbv.period_list for ct in gbv.set_list if (item_ind,j) in gbv.prod_key), name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0],jta[1]-1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    prob.addConstrs((rCi[jta] <= gbv.init_inv[jta[2][0],jta[0]] for jta in gbv.alt_list if (jta[2][1] == item_ind) and (jta[1] == 1)), name='r_ub_rev_ini')
    prob.addConstrs((yUOi[j,t] <= vi[j,t-1] for j in gbv.plant_list for t in gbv.period_list), name='yo_ub')
    prob.addConstrs((xCi[i,j,t] == wi[i,j,t] * gbv.lot_size[i,j] for i,j in prod_plant_i for t in gbv.period_list), name='batch')
    prob.addConstrs((wi[i,j,t] <= gbv.max_prod[i,j]/gbv.lot_size[i,j] for i,j in prod_plant_i for t in gbv.period_list), name='w_ub')

    # set up the subproblem specific objective
    theta = prob.addVar(lb = 0.0, name = 'theta')
    prob.addConstr(theta == gp.quicksum(gbv.holding_cost[item_ind] * vi[j,t] for j in gbv.plant_list for t in gbv.period_list) +
                      gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list), name = 'obj_local')

    prob.update()
    return prob

def x_item_init(x_prob_model):
    '''
    initialize the global variables for the item decomposition
    return: global variables with their primal/dual values initialized
    '''

    # initialize the variable x
    x_keys = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    x_var = global_var("x", x_keys)
    x_var.init_dual(gbv.item_list)

    # initialize the variable r
    r_keys = ["r[{},{},{}]".format(*r_rel) for r_rel in gbv.alt_list]
    r_var = global_var("r", r_keys)
    r_var.init_dual(gbv.item_list)

    global_vars_init = [x_var, r_var]
    global_ind_init = []
    prob_list_init = []

    # for each item subproblem, obtain:
    # global_ind: the index of the global variables that require value passing
    # prob_list: the constructed problem
    for i in gbv.item_list:
        x_prod_list = [(it_id, j) for it_id in gbv.item_list for j in gbv.plant_list if \
                        ((j in gbv.item_plant[i]) and (it_id,j) in gbv.prod_key) or ((it_id, i) in gbv.bom_key[j])]

        # obtain the variable names: x
        x_name = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in x_prod_list for t in gbv.period_list]

        r_list = [item for item in gbv.alt_list if i in item[2]]
        # obtain the variable names: r
        r_name = ["r[{},{},{}]".format(*r_rel) for r_rel in r_list]

        global_names_i = [x_name, r_name]
        assert len(global_names_i) == len(global_vars_init)

        # obtain the global variable index
        global_ind_i = []
        for varI in range(len(global_vars_init)):
            var_ind = [global_vars_init[varI].keys.index(global_names_i[varI][j]) for j in range(len(global_names_i[varI]))]
            global_ind_i.append(var_ind)
        global_ind_init.append(global_ind_i)

        # construct the local problems
        prob_list_init.append(x_prob_model(i))
    return global_vars_init, global_ind_init, prob_list_init

def x_plant(gbv):
    pass

def x_time(gbv):
    pass

def x_solve(i, rho, quad_penalty = True):
    '''
    i: the index of sub-problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''

    prob = prob_list[i]
    global_ind_i = global_ind[i]

    # set the linear expression of the local part for the objective
    obj_local_part = prob.getVarByName("theta")

    time_init = time.time()

    # objective function
    if quad_penalty:
        # if the objective contains the quadratic penalty term (bundle method)
        prob.setObjective(obj_local_part -
                          gp.quicksum(global_vars[gvar_ind].dual[i][vi] * prob.getVarByName(global_vars[gvar_ind].keys[vi]) \
                                      for gvar_ind in range(len(global_vars)) \
                                      for vi in global_ind_i[gvar_ind]) + # add all global variables for relaxation part
                          rho / 2 * gp.quicksum((global_vars[gvar_ind].value[vi] - prob.getVarByName(global_vars[gvar_ind].keys[vi])) ** 2 \
                                                for gvar_ind in range(len(global_vars)) \
                                                for vi in global_ind_i[gvar_ind]), GRB.MINIMIZE)
    else:
        # if the objective does not contain the quadratic penalty term (Lagrangian method)
        prob.setObjective(obj_local_part -
                          gp.quicksum(global_vars[gvar_ind].dual[i][vi] * prob.getVarByName(global_vars[gvar_ind].keys[vi]) \
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

def z_solve(local_results, rho, var_threshold = 1e-6):
    '''
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''

    I = len(local_results)
    # initiate the dual residual calculation
    dual_residual_sqr = 0
    primal_tolerance = 0
    local_combined = []
    global_combined = []

    # update the global variables
    for gvar_ind in range(len(global_vars)):
        global_array = np.zeros(len(global_vars[gvar_ind].value))
        global_counter = np.zeros(len(global_vars[gvar_ind].value))
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                # calculate the local information to update the global variables
                local_array = np.zeros(len(global_vars[gvar_ind].value))
                local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]

                # the dual value for the non-participating global variables is zero
                global_array += (rho * local_array - global_vars[gvar_ind].dual[i])

                # count whether the global variable occurs in the i-th subproblem
                global_counter[global_ind[i][gvar_ind]] += 1

                # record the combined global/local vars to calculate the epsilon
                local_combined.extend(local_results[i][gvar_ind])
                global_combined.extend(global_vars[gvar_ind].value[global_ind[i][gvar_ind]])

        global_var_pre_proc = np.maximum(global_array / (rho * global_counter), 0.0)
        # eliminate values that are very close to zero
        global_var_post_proc = np.where(global_var_pre_proc < var_threshold, 0.0, global_var_pre_proc)
        # calculate the dual residual
        for i in range(I):
            residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars[gvar_ind].value[global_ind[i][gvar_ind]])
            dual_residual_sqr += sum(residual_list**2)

        global_vars[gvar_ind].value = global_var_post_proc

    primal_tolerance = np.maximum(LA.norm(local_combined), LA.norm(global_combined))
    return np.sqrt(dual_residual_sqr), primal_tolerance

def pi_solve(local_results, rho, var_threshold = 1e-6, adj_coeff = 1.6):
    '''
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, prob_list, global_vars, global_ind

    I = len(local_results)
    # initiate the primal residual calculation
    primal_residual_sqr = 0
    dual_tolerance = 0
    dual_combined = []

    # update the dual variables
    for gvar_ind in range(len(global_vars)):
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                local_array = np.zeros(len(global_vars[gvar_ind].value))
                local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]

                # obtain the residual and update the dual
                residual_list = global_vars[gvar_ind].value - np.where(local_array < var_threshold, 0.0, local_array)
                global_vars[gvar_ind].dual[i] += adj_coeff * rho * residual_list

                # calculate the primal residual
                primal_residual_sqr += sum(residual_list**2)

                # record the combined dual vars to calculate the epsilon
                dual_combined.extend(global_vars[gvar_ind].dual[i][global_ind[i][gvar_ind]])

    dual_tolerance = LA.norm(dual_combined)

    return np.sqrt(primal_residual_sqr), dual_tolerance
