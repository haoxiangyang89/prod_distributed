# -*- coding: utf-8 -*-
"""
Created on Tue June 13, 2023

@author: haoxiang
"""

'''
main structure of running decomposition instances in parallel
'''

import os
import sys
from argparse import ArgumentParser
import numpy as np
from numpy import linalg as LA
import time
import math
import matplotlib.pyplot as plt
import random
import multiprocessing
from functools import partial

from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from multiprocessing import Manager
from multiprocessing import SimpleQueue
from multiprocessing.pool import Pool

from src.params import ADMMparams
from src.readin import *
from src.GlobalVariable import *
from src.variables import global_var

from random import random
from time import sleep
import collections

from gurobipy import gurobipy as gp
from gurobipy import GRB


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def parse_arguments():
    parser = ArgumentParser("Parameters for production planning optimization ADMM process")
    parser.add_argument("-t", "--type",
                        dest='type',
                        default='x_item',
                        type=str,
                        help='The type of x problem.')
    parser.add_argument("-n", "--number",
                        dest="number",
                        default=1,
                        type=int,
                        help="The number of x problems."
                        )
    parser.add_argument("-x", "--var",
                        dest="var",
                        default='./data/vars.json',
                        type=str,
                        help='Global variables specifications')
    parser.add_argument("-pf", "--params_file",
                        default="./data/params.json",
                        dest='params_file',
                        type=str,
                        help='Parameter json file location')
    parser.add_argument("-df", "--data_folder",
                        default="./data/pilot_test",
                        dest='data_folder',
                        type=str,
                        help='CSV data folder location')
    parser.add_argument("-th", "--threads",
                        default=1,
                        dest='threads',
                        type=int,
                        help='The number of threads')

    args = parser.parse_args()
    return args


def create_gbv(path_file):
    '''
       This function creates a global data structure to contain all the problem parameters.
    '''
    gbv = GlobalVariable()

    # load the data and form a data structure and create the global instance data frame
    # gbv.item_list, gbv.holding_cost, gbv.penalty_cost = input_item(os.path.join(path_file, "dm_df_item.csv"),
    #                                                               os.path.join(path_file, "dm_df_production.csv"),
    #                                                               os.path.join(path_file, "dm_df_bom.csv"),
    #                                                               os.path.join(path_file, "dm_df_transit.csv"),
    #                                                               os.path.join(path_file, "dm_df_alternate_item.csv"))
    gbv.item_list, gbv.holding_cost, gbv.penalty_cost = input_item(os.path.join(path_file, "dm_df_item.csv"))
    gbv.K = len(gbv.item_list)
    gbv.set_dict = input_set(os.path.join(path_file, "dm_df_set.csv"))
    gbv.item_set, gbv.set_list = input_item_set(os.path.join(path_file, "dm_df_item_set.csv"))
    gbv.alt_list, gbv.alt_dict, gbv.alt_cost = input_item_alternate(os.path.join(path_file, "dm_df_alternate_item.csv"))
    gbv.unit_cap_set = input_unit_capacity(os.path.join(path_file, "dm_df_unit_capacity.csv"))
    gbv.item_plant = input_item_plant(os.path.join(path_file, "dm_df_item_plant.csv"))
    gbv.bom_key, gbv.bom_dict = input_bom(os.path.join(path_file, "dm_df_bom.csv"))
    gbv.prod_key, gbv.lot_size, gbv.lead_time, gbv.component_holding_cost, \
    gbv.prod_cost, gbv.min_prod, gbv.max_prod = input_production(os.path.join(path_file, "dm_df_production.csv"))
    # need a penalty cost for demand mismatch

    gbv.plant_list = input_plant(os.path.join(path_file, "dm_df_plant.csv"))
    gbv.M = len(gbv.plant_list)
    gbv.cap_period_list, gbv.cap_key, gbv.max_cap = input_capacity(os.path.join(path_file, "dm_df_max_capacity.csv"))
    gbv.transit_list, gbv.transit_time, gbv.transit_cost = input_transit(os.path.join(path_file, "dm_df_transit.csv"))
    gbv.L = len(gbv.transit_list)
    gbv.init_inv = input_init_inv(os.path.join(path_file, "dm_df_inv.csv"))

    gbv.period_list = input_periods(os.path.join(path_file, "dm_df_periods.csv"))
    gbv.T = len(gbv.period_list)
    gbv.external_purchase = input_po(os.path.join(path_file, "dm_df_po.csv"))
    gbv.real_demand, gbv.forecast_demand = input_demand(os.path.join(path_file, "dm_df_demand.csv"))

    gbv = timeline_adjustment(gbv)
    gbv = cap_adjustment(gbv)
    gbv = obtain_bounds(gbv)
    gbv = analyze_bom(gbv)

    for j in gbv.plant_list:
        if not (j in gbv.bom_key.keys()):
            gbv.bom_key[j] = []
            gbv.bom_dict[j] = {}

    # zero padding for the demand and external purchase
    # we need to change the data into a numpy array form, or maybe sparse tensor
    for i in gbv.item_list:
        for j in gbv.plant_list:
            for t in gbv.period_list:
                if not ((i, j, t) in gbv.external_purchase.keys()):
                    gbv.external_purchase[i, j, t] = 0.0
        for t in gbv.period_list:
            if not ((i, t) in gbv.real_demand.keys()):
                gbv.real_demand[i, t] = 0.0

    for i in gbv.item_list:
        for j in gbv.plant_list:
            if not ((i, j) in gbv.prod_key):
                gbv.lead_time[i, j] = 0

    return gbv


def global_var_init_item(gbv):
    x_keys = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    r_keys = ["r[{},{},{}]".format(*r_rel) for r_rel in gbv.alt_list]

    global_vars = dict()

    # initialize the variable values
    global_vars["value"] = [np.zeros(len(x_keys)), np.zeros(len(r_keys))]

    # initialize the variable fixing status
    global_vars["fixed"] = [np.zeros(len(x_keys)), np.zeros(len(r_keys))]

    # initialize the variable duals
    global_vars["dual"] = [[np.zeros(len(x_keys)) for i in gbv.item_list],
                           [np.zeros(len(r_keys)) for i in gbv.item_list]]

    return global_vars


def global_const_init(gbv):
    global_var_const = {}
    global_var_const["name"] = ["x", "r"]
    x_keys = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    x_ind = [(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    r_keys = ["r[{},{},{}]".format(*r_rel) for r_rel in gbv.alt_list]
    global_var_const["keys"] = [x_keys, r_keys]
    global_var_const["indices"] = [x_ind, gbv.alt_list]

    global_ind = []

    # for each item subproblem, obtain:
    # global_ind: the index of the global variables that require value passing
    # prob_list: the constructed problem
    for i in gbv.item_list:
        '''
        x_prod_list = []
        if gbv.item_plant.get(i, -1) != -1:
            x_prod_list.extend([(it_id, j) for it_id in gbv.item_list for j in gbv.plant_list if \
                                (j in gbv.item_plant[i]) and ((it_id, j) in gbv.prod_key)])
        for j in gbv.plant_list:
            if gbv.bom_key.get(j, -1) != -1:
                x_prod_list.extend([(it_id, j) for it_id in gbv.item_list if
                                    ((it_id, i) in gbv.bom_key[j]) and ((it_id, j) in gbv.prod_key)])
        x_prod_list = [*set(x_prod_list)]
        # x_prod_list = [(it_id, j) for it_id in gbv.item_list for j in gbv.plant_list if \
        #                ((j in gbv.item_plant[i]) and (it_id,j) in gbv.prod_key) or ((it_id, i) in gbv.bom_key[j])]
        '''
        x_prod_list = [(x_prod[0], x_prod[1]) for x_prod in gbv.prod_key]

        # obtain the variable names: x
        x_name = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in x_prod_list for t in gbv.period_list]

        '''
        r_list = [item for item in gbv.alt_list if i in item[2]]
        '''
        r_list = [item for item in gbv.alt_list]
        # obtain the variable names: r
        r_name = ["r[{},{},{}]".format(*r_rel) for r_rel in r_list]

        global_names_i = [x_name, r_name]
        assert len(global_names_i) == len(global_var_const["name"])

        # obtain the global variable index
        global_ind_i = []
        for varI in range(len(global_var_const["name"])):
            var_ind = [global_var_const["keys"][varI].index(global_names_i[varI][j]) for j in
                       range(len(global_names_i[varI]))]
            global_ind_i.append(var_ind)
        global_ind.append(global_ind_i)

    global sqrt_dim
    sqrt_dim = np.sqrt(sum(len(global_ind[i][gvar_ind]) for gvar_ind in range(len(global_var_const["name"])) for i in
                           range(len(global_ind))))

    global_counter = []
    for gvar_ind in range(len(global_var_const["name"])):
        local_counter = np.zeros(len(global_var_const["keys"][gvar_ind]))
        for i in range(len(gbv.item_list)):
            local_counter[global_ind[i][gvar_ind]] += 1

        global_counter.append(local_counter)

    return global_var_const, global_ind, sqrt_dim, global_counter


def x_item_init(global_info, global_dict_const, global_ind_list, dim, gvar_counter):
    '''
    initialize the global variables for the item decomposition
    return: global variables with their primal/dual values initialized
    '''

    # declare scope of a new global variable
    global gbv
    # store argument in the global variable for this process
    gbv = global_info

    global g_time
    g_time = time.time()

    # create global variable and global indices
    global global_var_const, global_ind, sqrt_dim, global_counter
    global_var_const = global_dict_const
    global_ind = global_ind_list

    sqrt_dim = dim

    global_counter = gvar_counter


def x_solve(dummy_item_index, rho, global_vars, quad_penalty=True, relax_option=True, penalty_mag=1e5, fixed_param=True,
            fixing_round=10):
    '''
    A reformulation of the sub problem to guarantee feasibility for the sub problems
    item_ind: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    global gbv, global_ind, global_var_const, global_counter
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\tSetting up Model " + str(item_ind) + "!\n")

    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)
    prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    '''
    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]
    '''
    prod_plant_i = [(i, j) for i, j in gbv.prod_key]
    alt_i = [alt_item for alt_item in gbv.alt_list]
    prod_plant_i = [*set(prod_plant_i)]
    alt_i = [*set(alt_i)]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t

    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = prob.addVars(prod_plant_i, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')

    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                    name='input_item_prod')
    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) + gbv.external_purchase[
                         item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                    name='input_item_prod')

    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                     + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (t in gbv.max_cap[ct, j].keys())),
                    name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[1] == 1)),
        name='r_ub_rev_ini')
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    print(multiprocessing.current_process(), "\tSolving Model " + str(dummy_item_index) + "!\n")

    global_ind_i = global_ind[dummy_item_index]

    time_init = time.time()
    # set up the subproblem specific objective
    # set the linear expression of the local part for the objective
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)

    gvar_ind = 0
    obj_local_part -= gp.quicksum(
        global_vars["dual"][gvar_ind][dummy_item_index][xi_ind] * xCi[global_var_const["indices"][gvar_ind][xi_ind]] for
        xi_ind
        in global_ind_i[gvar_ind])

    gvar_ind = 1
    obj_local_part -= gp.quicksum(
        global_vars["dual"][gvar_ind][dummy_item_index][ri_ind] * rCi[global_var_const["indices"][gvar_ind][ri_ind]] for
        ri_ind
        in global_ind_i[gvar_ind])

    # objective function
    if quad_penalty:
        # if the objective contains the quadratic penalty term (bundle method)
        gvar_ind = 0
        obj_local_part += rho / 2 * gp.quicksum(
            (global_vars["value"][gvar_ind][xi_ind] - xCi[global_var_const["indices"][gvar_ind][xi_ind]]) ** 2 \
            for xi_ind in global_ind_i[gvar_ind])

        gvar_ind = 1
        obj_local_part += rho / 2 * gp.quicksum(
            (global_vars["value"][gvar_ind][ri_ind] - rCi[global_var_const["indices"][gvar_ind][ri_ind]]) ** 2 \
            for ri_ind in global_ind_i[gvar_ind])

    prob.setObjective(obj_local_part, GRB.MINIMIZE)

    # fixing the local variables
    if fixed_param:
        gvar_ind = 0
        for xi_ind in global_ind_i[gvar_ind]:
            if global_vars["fixed"][gvar_ind][xi_ind] == fixing_round:   #global_counter[gvar_ind][xi_ind] * fixed_lag
                fixed_var = xCi[global_var_const["indices"][gvar_ind][xi_ind]]
                fixed_var.lb = global_vars["value"][gvar_ind][xi_ind]
                fixed_var.ub = global_vars["value"][gvar_ind][xi_ind]

        gvar_ind = 1
        for ri_ind in global_ind_i[gvar_ind]:
            if global_vars["fixed"][gvar_ind][ri_ind] == fixing_round:
                fixed_var = rCi[global_var_const["indices"][gvar_ind][ri_ind]]
                fixed_var.lb = global_vars["value"][gvar_ind][ri_ind]
                fixed_var.ub = global_vars["value"][gvar_ind][ri_ind]

    # solve the problem
    prob.update()

    time_obj = time.time()
    time_setObj = time_obj - time_init
    prob.optimize()
    time_solution = time.time() - time_obj
    print("Objective setup time: {} sec; Solution time: {} sec.".format(time_setObj, time_solution))

    # obtain the solutions and output results
    local_output = []

    gvar_ind = 0
    local_val = []
    for xi_ind in global_ind_i[gvar_ind]:
        local_val.append(xCi[global_var_const["indices"][gvar_ind][xi_ind]].X)

    local_output.append(local_val)

    gvar_ind = 1
    local_val = []
    for ri_ind in global_ind_i[gvar_ind]:
        local_val.append(rCi[global_var_const["indices"][gvar_ind][ri_ind]].X)
    local_output.append(local_val)

    # obtain the solution and return the subproblem solution
    return local_output


def x_solve_dual(dummy_item_index, dual_vars, rho, primal_vars, quad_penalty=True, relax_option=True, penalty_mag=1e5,
                 fixed_param=True, fixing_round=10):
    '''
    A reformulation of the sub problem to guarantee feasibility for the sub problems
    item_ind: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    global gbv, global_ind, global_var_const, global_counter
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\tSetting up Model " + str(item_ind) + "!\n")

    x_prob = gp.Model("item_{}".format(item_ind))
    x_prob.setParam("Threads", 1)
    x_prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i

    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    # prod_plant_i = [(i, j) for i in gbv.item_list for j in gbv.plant_list if \
    #                ((j in gbv.item_plant[item_ind]) and (i, j) in gbv.prod_key) or ((i, item_ind) in gbv.bom_key[j])]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = x_prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = x_prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = x_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = x_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = x_prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t

    yUIi = x_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = x_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = x_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = x_prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = x_prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = x_prob.addVars(prod_plant_i, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = x_prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    x_prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                       == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    x_prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                       for t in gbv.period_list), name='inventory')

    x_prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                       xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                       for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                      name='input_item_prod')
    x_prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) + gbv.external_purchase[
                           item_ind, j, t] \
                       for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                      name='input_item_prod')

    x_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                       gp.quicksum(
                           gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                               zi_p[j, t] - zi_m[j, t]) +
                       gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                   (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                       gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                   (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                       for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                      name="output_item")
    x_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                       + (
                               zi_p[j, t] - zi_m[j, t]) +
                       gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                   (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                       gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                   (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                       for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                      name="output_item")

    x_prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                       for ct, j in gbv.max_cap.keys() for t in gbv.period_list if
                       ((item_ind, j) in gbv.prod_key) and (t in gbv.max_cap[ct, j].keys())), name='capacity')
    x_prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind),
                      name='r_ub')
    x_prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[2][1] == item_ind) and (jta[1] == 1)),
        name='r_ub_rev_ini')
    x_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                      name='yo_ub')
    x_prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    x_prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    print(multiprocessing.current_process(), "\tSolving Model " + str(dummy_item_index) + "!\n")

    global_ind_i = global_ind[dummy_item_index]

    time_init = time.time()
    # set up the subproblem specific objective
    # set the linear expression of the local part for the objective
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)

    gvar_ind = 0
    obj_local_part -= gp.quicksum(
        dual_vars[gvar_ind][xi_ind] * xCi[global_var_const["indices"][gvar_ind][global_ind_i[gvar_ind][xi_ind]]] for
        xi_ind
        in range(len(global_ind_i[gvar_ind])))

    gvar_ind = 1
    obj_local_part -= gp.quicksum(
        dual_vars[gvar_ind][ri_ind] * rCi[global_var_const["indices"][gvar_ind][global_ind_i[gvar_ind][ri_ind]]] for
        ri_ind
        in range(len(global_ind_i[gvar_ind])))

    # objective function
    if quad_penalty:
        # if the objective contains the quadratic penalty term (bundle method)
        gvar_ind = 0
        obj_local_part += rho / 2 * gp.quicksum(
            (primal_vars[gvar_ind][xi_ind] - xCi[global_var_const["indices"][gvar_ind][xi_ind]]) ** 2 \
            for xi_ind in global_ind_i[gvar_ind])

        gvar_ind = 1
        obj_local_part += rho / 2 * gp.quicksum(
            (primal_vars[gvar_ind][ri_ind] - rCi[global_var_const["indices"][gvar_ind][ri_ind]]) ** 2 \
            for ri_ind in global_ind_i[gvar_ind])

    x_prob.setObjective(obj_local_part, GRB.MINIMIZE)

    # solve the x_problem
    x_prob.update()

    time_obj = time.time()
    time_setObj = time_obj - time_init
    x_prob.optimize()
    time_solution = time.time() - time_obj
    print("Objective setup time: {} sec; Solution time: {} sec.".format(time_setObj, time_solution))

    # obtain the solutions and output results
    local_output = []

    gvar_ind = 0
    local_val = []
    for xi_ind in global_ind_i[gvar_ind]:
        local_val.append(xCi[global_var_const["indices"][gvar_ind][xi_ind]].X)

    local_output.append(local_val)

    gvar_ind = 1
    local_val = []
    for ri_ind in global_ind_i[gvar_ind]:
        local_val.append(rCi[global_var_const["indices"][gvar_ind][ri_ind]].X)
    local_output.append(local_val)

    # obtain the solution and return the subproblem solution
    return local_output

def x_solve_int_BD_cut(dummy_item_index, rho, global_vars, BD_cut, UB, LB, quad_penalty=True, relax_option=True, penalty_mag=1e5,
            var_threshold=1e-5,
            fixed_param=True,
            fixing_round=10):
    '''
            A reformulation of the sub problem to guarantee feasibility for the sub problems
            item_ind: the index of sub-problem
            return: the subproblem built with Gurobi
            '''
    global gbv, global_ind, global_var_const, global_counter
    '''
    BD_cut[ite_num] = {}
    BD_cut[ite_num]["global_vars_value"] = [np.zeros(len(x_keys)), np.zeros(len(r_keys))] # x_hat,r_hat
    BD_cut[ite_num]["second_stage_obj"] = np.zeros(len(gbv.item_list))   # eta_hat
    for i in gbv.item_list:
        BD_cut[ite_num][i] = [gapi, np.zeros(len(x_keys)), np.zeros(len(r_keys)), dual_eta]  # gapi, dual_x, dual_r, dual_eta
    '''
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\tSetting up Model " + str(item_ind) + "!\n")

    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)
    prob.setParam("OutputFlag", 0)
    #prob.setParam("Method", 1)
    prob.setParam("DualReductions", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    '''
    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]
    '''
    prod_plant_i = [(i, j) for i, j in gbv.prod_key]
    alt_i = [alt_item for alt_item in gbv.alt_list]
    prod_plant_i = [*set(prod_plant_i)]
    alt_i = [*set(alt_i)]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t

    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = prob.addVars(prod_plant_i, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # second stage objective approximation
    eta = prob.addVars(gbv.item_list, lb=0.0, name='second_stage_obj')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')

    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                    name='input_item_prod')
    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     gbv.external_purchase[
                         item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                    name='input_item_prod')

    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                     + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (t in gbv.max_cap[ct, j].keys())),
                    name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind),
                    name='r_ub')
    prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[1] == 1)),
        name='r_ub_rev_ini')
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    print(multiprocessing.current_process(), "\tSolving Model " + str(dummy_item_index) + "!\n")

    global_ind_i = global_ind[dummy_item_index]

    time_init = time.time()
    # set up the subproblem specific objective
    # set the linear expression of the local part for the objective
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)

    gvar_ind = 0
    obj_local_part -= gp.quicksum(
        global_vars["dual"][gvar_ind][dummy_item_index][xi_ind] * xCi[global_var_const["indices"][gvar_ind][xi_ind]]
        for
        xi_ind
        in global_ind_i[gvar_ind])

    gvar_ind = 1
    obj_local_part -= gp.quicksum(
        global_vars["dual"][gvar_ind][dummy_item_index][ri_ind] * rCi[global_var_const["indices"][gvar_ind][ri_ind]]
        for
        ri_ind
        in global_ind_i[gvar_ind])

    # objective function
    if quad_penalty:
        # if the objective contains the quadratic penalty term (bundle method)
        gvar_ind = 0
        obj_local_part += rho / 2 * gp.quicksum(
            (global_vars["value"][gvar_ind][xi_ind] - xCi[global_var_const["indices"][gvar_ind][xi_ind]]) ** 2 \
            for xi_ind in global_ind_i[gvar_ind])

        gvar_ind = 1
        obj_local_part += rho / 2 * gp.quicksum(
            (global_vars["value"][gvar_ind][ri_ind] - rCi[global_var_const["indices"][gvar_ind][ri_ind]]) ** 2 \
            for ri_ind in global_ind_i[gvar_ind])

    prob.setObjective(obj_local_part, GRB.MINIMIZE)

    # fixing the local variables
    if fixed_param:
        gvar_ind = 0
        for xi_ind in global_ind_i[gvar_ind]:
            if global_vars["fixed"][gvar_ind][xi_ind] == fixing_round:
                fixed_var = xCi[global_var_const["indices"][gvar_ind][xi_ind]]
                fixed_var.lb = global_vars["value"][gvar_ind][xi_ind]
                fixed_var.ub = global_vars["value"][gvar_ind][xi_ind]

        gvar_ind = 1
        for ri_ind in global_ind_i[gvar_ind]:
            if global_vars["fixed"][gvar_ind][ri_ind] == fixing_round:
                fixed_var = rCi[global_var_const["indices"][gvar_ind][ri_ind]]
                fixed_var.lb = global_vars["value"][gvar_ind][ri_ind]
                fixed_var.ub = global_vars["value"][gvar_ind][ri_ind]

    # constraints on eta
    prob.addConstr(eta[item_ind] == gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list))
    # integrate Benders cut
    if len(BD_cut.keys()) > 0:
        max_ite = max(BD_cut.keys())
        for ite_num in BD_cut.keys():  # BD_cut.keys(), only add the newest cut
            x_hat = BD_cut[ite_num]["global_vars_value"][0]
            r_hat = BD_cut[ite_num]["global_vars_value"][1]
            eta_hat = BD_cut[ite_num]["second_stage_obj"]

            prob.addConstrs(BD_cut[ite_num][i][0] + gp.quicksum(
                BD_cut[ite_num][i][1][xi_ind] * (xCi[global_var_const["indices"][0][xi_ind]] - x_hat[xi_ind])
                for xi_ind
                in
                range(x_hat.size))
                            + gp.quicksum(
                BD_cut[ite_num][i][2][ri_ind] * (rCi[global_var_const["indices"][1][ri_ind]] - r_hat[ri_ind])
                for ri_ind
                in
                range(r_hat.size))
                            + (BD_cut[ite_num][i][3] * (eta[i] - eta_hat[gbv.item_list.index(i)])) <= 0 for i in
                            gbv.item_list if (i in BD_cut[ite_num].keys()) and (i != item_ind))

    # integrate the best upper/lower bound so far to induce another cut
    prob.addConstr(gp.quicksum(eta[i] for i in gbv.item_list) <= UB)
    prob.addConstr(gp.quicksum(eta[i] for i in gbv.item_list) >= LB)

    # solve the problem
    prob.update()

    time_obj = time.time()
    time_setObj = time_obj - time_init
    prob.optimize()
    time_solution = time.time() - time_obj
    print("Objective setup time: {} sec; Solution time: {} sec.".format(time_setObj, time_solution))

    # obtain the solutions and output results
    if prob.Status == 2:
        local_output = []

        gvar_ind = 0
        local_val = []
        for xi_ind in global_ind_i[gvar_ind]:
            if xCi[global_var_const["indices"][gvar_ind][xi_ind]].X > var_threshold:
                local_val.append(xCi[global_var_const["indices"][gvar_ind][xi_ind]].X)
            else:
                local_val.append(0.0)

        local_output.append(local_val)

        gvar_ind = 1
        local_val = []
        for ri_ind in global_ind_i[gvar_ind]:
            if rCi[global_var_const["indices"][gvar_ind][ri_ind]].X > var_threshold:
                local_val.append(rCi[global_var_const["indices"][gvar_ind][ri_ind]].X)
            else:
                local_val.append(0.0)
        local_output.append(local_val)

        local_val = []
        for i in gbv.item_list:
            if eta[i].X > var_threshold:
                local_val.append(eta[i].X)
            else:
                local_val.append(0.0)
        local_output.append(local_val)

        # obtain the solution and return the subproblem solution
        return local_output

    else:
        wr = "Sub " + str(item_ind) + " unsolvable! Error: " + str(prob.Status)
        wr_s = open("sub_unsolve", "a")
        wr_s.write(wr)
        wr_s.close()
        if prob.Status == 3:
            print(str(item_ind) + " infeasible")
            prob.computeIIS()
            prob.write('iismodel.ilp')
        elif prob.Status == 5:
            print(str(item_ind) + " unbounded")


def BD_x_sub(dummy_item_index, hat_global, hat_eta, relax_option=True, penalty_mag=1e5, var_threshold=1e-5):
    '''
    generate benders cuts
    '''
    global gbv, global_ind, global_var_const
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\t(BD sub) Setting up Model " + str(item_ind) + "!\n")

    BD_sub_prob = gp.Model("item_{}".format(item_ind))
    BD_sub_prob.setParam("Threads", 1)
    BD_sub_prob.setParam("OutputFlag", 0)
    BD_sub_prob.setParam("DualReductions", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    '''
    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    # prod_plant_i = [(i, j) for i in gbv.item_list for j in gbv.plant_list if \
    #                ((j in gbv.item_plant[item_ind]) and (i, j) in gbv.prod_key) or ((i, item_ind) in gbv.bom_key[j])]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]
    '''
    prod_plant_i = [(i, j) for i, j in gbv.prod_key]
    alt_i = [alt_item for alt_item in gbv.alt_list]
    prod_plant_i = [*set(prod_plant_i)]
    alt_i = [*set(alt_i)]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = BD_sub_prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = BD_sub_prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = BD_sub_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = BD_sub_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = BD_sub_prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t
    # inv_ub = (sum([gbv.max_prod[item_ind, j] for j in gbv.plant_list if (item_ind, j) in gbv.max_prod.keys()]) * len(
    #    gbv.period_list) + sum(gbv.external_purchase[item_ind, j, t] for j in gbv.plant_list for t in gbv.period_list)) * 10
    # if inv_ub > 0:
    #    for j in gbv.plant_list:
    #        for t in gbv.period_list:
    #            vi[j, t].ub = inv_ub

    yUIi = BD_sub_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = BD_sub_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = BD_sub_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = BD_sub_prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = BD_sub_prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = BD_sub_prob.addVars(prod_plant_i, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # lower approximation of second stage objective
    etai = BD_sub_prob.addVar(lb=0.0, name='eta')
    # gap between approximation eta and the actual objective, if > 0, then return a cut
    gapi = BD_sub_prob.addVar(lb=0.0, name="approx_gap")


    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = BD_sub_prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    BD_sub_prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    BD_sub_prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')

    BD_sub_prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                    name='input_item_prod')
    BD_sub_prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) + gbv.external_purchase[
                         item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                    name='input_item_prod')

    BD_sub_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    BD_sub_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                     + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    BD_sub_prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (t in gbv.max_cap[ct, j].keys())),
                    name='capacity')
    BD_sub_prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    BD_sub_prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[1] == 1)),
        name='r_ub_rev_ini')
    BD_sub_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    BD_sub_prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    BD_sub_prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    # put second stage objective into constraint
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)
    BD_sub_prob.addConstr(obj_local_part <= etai + gapi, name='second_obj_approx')

    # given first stage solution and the approximation of second stage objective of scenario s, fix them
    global_ind_i = global_ind[dummy_item_index]
    x_Constrs_dict = {}
    r_Constrs_dict = {}
    # fixing the local variables
    gvar_ind = 0
    for xi_ind in global_ind_i[gvar_ind]:
        x_Constrs_dict[xi_ind] = BD_sub_prob.addConstr(
            xCi[global_var_const["indices"][gvar_ind][xi_ind]] == hat_global[gvar_ind][xi_ind])

    gvar_ind = 1
    for ri_ind in global_ind_i[gvar_ind]:
        r_Constrs_dict[ri_ind] = BD_sub_prob.addConstr(
            rCi[global_var_const["indices"][gvar_ind][ri_ind]] == hat_global[gvar_ind][ri_ind])

    # second stage objective approximation
    eta_approx = BD_sub_prob.addConstr(etai == hat_eta[dummy_item_index])

    time_init = time.time()

    BD_sub_prob.setObjective(gapi, GRB.MINIMIZE)

    BD_sub_prob.update()
    time_var = time.time()
    time_setVar = time_var - time_init
    BD_sub_prob.optimize()
    time_solution = time.time() - time_var
    print("BD-sub problem variable setup time: {} sec; Solution time: {} sec.".format(time_setVar, time_solution))

    # obtain dual solutions for generating cut if gap > 0
    if BD_sub_prob.Status == 2:
        if BD_sub_prob.ObjVal > var_threshold:
            local_dual_output = [BD_sub_prob.ObjVal]

            gvar_ind = 0
            local_dual_val = []
            for xi_ind in global_ind_i[gvar_ind]:
                local_dual_val.append(x_Constrs_dict[xi_ind].Pi)
            local_dual_output.append(local_dual_val)

            gvar_ind = 1
            local_dual_val = []
            for ri_ind in global_ind_i[gvar_ind]:
                local_dual_val.append(r_Constrs_dict[ri_ind].Pi)
            local_dual_output.append(local_dual_val)

            local_dual_output.append(eta_approx.Pi)

            return local_dual_output
    elif BD_sub_prob.Status == 3:
        print("BD_sub " + str(item_ind) + " infeasible")
        BD_sub_prob.computeIIS()
        BD_sub_prob.write('BD_sub_iismodel.ilp')
        return("unsolvable")
    elif BD_sub_prob.Status == 5:
        print("BD_sub " + str(item_ind) + " unbounded")
        return("unsolvable")


def x_solve_ub(dummy_item_index, global_vars, relax_option=True, penalty_mag=1e5):
    '''
    A reformulation of the sub problem to guarantee feasibility for the sub problems
    item_ind: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    global gbv, global_ind, global_var_const
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\t(UB) Setting up Model " + str(item_ind) + "!\n")

    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)
    prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    '''
    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    # prod_plant_i = [(i, j) for i in gbv.item_list for j in gbv.plant_list if \
    #                ((j in gbv.item_plant[item_ind]) and (i, j) in gbv.prod_key) or ((i, item_ind) in gbv.bom_key[j])]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]
    '''
    prod_plant_i = [(i, j) for i, j in gbv.prod_key]
    alt_i = [alt_item for alt_item in gbv.alt_list]
    prod_plant_i = [*set(prod_plant_i)]
    alt_i = [*set(alt_i)]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t
    # inv_ub = (sum([gbv.max_prod[item_ind, j] for j in gbv.plant_list if (item_ind, j) in gbv.max_prod.keys()]) * len(
    #    gbv.period_list) + sum(gbv.external_purchase[item_ind, j, t] for j in gbv.plant_list for t in gbv.period_list)) * 10
    # if inv_ub > 0:
    #    for j in gbv.plant_list:
    #        for t in gbv.period_list:
    #            vi[j, t].ub = inv_ub

    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = prob.addVars(prod_plant_i, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')

    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                    name='input_item_prod')
    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) + gbv.external_purchase[
                         item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                    name='input_item_prod')

    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                     + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (t in gbv.max_cap[ct, j].keys())),
                    name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[1] == 1)),
        name='r_ub_rev_ini')
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    # set up the subproblem specific objective
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)

    global_ind_i = global_ind[dummy_item_index]

    time_init = time.time()

    # fix the local variables to their global counterparts' values
    gvar_ind = 0
    for xi_ind in global_ind_i[gvar_ind]:
        local_var = xCi[global_var_const["indices"][gvar_ind][xi_ind]]
        local_var.lb = global_vars["value"][gvar_ind][xi_ind]
        local_var.ub = global_vars["value"][gvar_ind][xi_ind]

    gvar_ind = 1
    for ri_ind in global_ind_i[gvar_ind]:
        local_var = rCi[global_var_const["indices"][gvar_ind][ri_ind]]
        local_var.lb = global_vars["value"][gvar_ind][ri_ind]
        local_var.ub = global_vars["value"][gvar_ind][ri_ind]

    prob.setObjective(obj_local_part, GRB.MINIMIZE)

    prob.update()
    time_var = time.time()
    time_setVar = time_var - time_init
    prob.optimize()
    time_solution = time.time() - time_var
    print("UB problem variable setup time: {} sec; Solution time: {} sec.".format(time_setVar, time_solution))

    # obtain the objective value (local upper bound) and return
    if prob.Status == 2:
        local_obj = prob.ObjVal
    else:
        local_obj = np.infty
    return local_obj


def x_solve_lb(dummy_item_index, global_vars, relax_option=True, penalty_mag=1e5):
    global gbv, global_ind, global_var_const
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\t(LB) Setting up Model " + str(item_ind) + "!\n")

    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)
    prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    '''
    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    # prod_plant_i = [(i, j) for i in gbv.item_list for j in gbv.plant_list if \
    #                ((j in gbv.item_plant[item_ind]) and (i, j) in gbv.prod_key) or ((i, item_ind) in gbv.bom_key[j])]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]
    '''
    prod_plant_i = [(i, j) for i, j in gbv.prod_key]
    alt_i = [alt_item for alt_item in gbv.alt_list]
    prod_plant_i = [*set(prod_plant_i)]
    alt_i = [*set(alt_i)]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t
    # inv_ub = (sum([gbv.max_prod[item_ind, j] for j in gbv.plant_list if (item_ind, j) in gbv.max_prod.keys()]) * len(
    #    gbv.period_list) + sum(gbv.external_purchase[item_ind, j, t] for j in gbv.plant_list for t in gbv.period_list)) * 10
    # if inv_ub > 0:
    #    for j in gbv.plant_list:
    #        for t in gbv.period_list:
    #            vi[j, t].ub = inv_ub

    yUIi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    if relax_option:
        wi = prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t
    else:
        wi = prob.addVars(prod_plant_i, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')

    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                    name='input_item_prod')
    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) + gbv.external_purchase[
                         item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                    name='input_item_prod')

    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                     + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (t in gbv.max_cap[ct, j].keys())),
                    name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[1] == 1)),
        name='r_ub_rev_ini')
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    '''
    solve x problems: solve for local variables
    i: the index of sub-problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    print(multiprocessing.current_process(), "\tSolving Model " + str(dummy_item_index) + "!\n")

    global_ind_i = global_ind[dummy_item_index]

    time_init = time.time()
    # set up the subproblem specific objective
    # set the linear expression of the local part for the objective
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + \
                     gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)

    gvar_ind = 0
    obj_local_part -= gp.quicksum(
        global_vars["dual"][gvar_ind][dummy_item_index][xi_ind] * xCi[global_var_const["indices"][gvar_ind][xi_ind]] for
        xi_ind
        in global_ind_i[gvar_ind])

    gvar_ind = 1
    obj_local_part -= gp.quicksum(
        global_vars["dual"][gvar_ind][dummy_item_index][ri_ind] * rCi[global_var_const["indices"][gvar_ind][ri_ind]] for
        ri_ind
        in global_ind_i[gvar_ind])

    prob.setObjective(obj_local_part, GRB.MINIMIZE)

    # solve the problem
    prob.update()
    time_var = time.time()
    time_setVar = time_var - time_init
    prob.optimize()
    time_solution = time.time() - time_var
    print("LB problem variable setup time: {} sec; Solution time: {} sec.".format(time_setVar, time_solution))

    # obtain the objective value (local upper bound) and return
    if prob.Status == 2:
        local_obj = prob.ObjVal
    else:
        local_obj = -np.infty
    return local_obj


def lb_add(global_vars):
    global gbv, global_var_const, global_ind

    I = gbv.K
    # initiate the dual residual calculation

    solver_start = time.time()
    # construct a MIP to solve for the global variables
    gvar_ind = 0
    lb_add_mip = gp.Model("global_mip")
    lb_add_mip.setParam("Threads", 1)
    lb_add_mip.setParam("OutputFlag", 0)

    # set up the decision variables to be the global variables and the capacity constraints
    # here this constraint is specific to item decomposition!!!
    x_ind = global_var_const["indices"][gvar_ind]
    x_keys = global_var_const["keys"][gvar_ind]
    assert len(global_var_const["keys"][gvar_ind]) == len(x_keys)
    x_vars = lb_add_mip.addVars(range(len(x_keys)), lb=0.0, name="x_var")  # x_keys should match x_ind
    w_vars = lb_add_mip.addVars(range(len(x_keys)), lb=0.0, vtype=GRB.INTEGER, name="glb_var")

    # set up the capacity constraints
    lb_add_mip.addConstrs((gp.quicksum(
        gbv.unit_cap.get((x_ind[glb_ind][0], x_ind[glb_ind][1], ct), 0) * x_vars[glb_ind] for glb_ind in
        range(len(x_keys)) \
        if (x_ind[glb_ind][1] == j) and (x_ind[glb_ind][2] == t)) <= gbv.max_cap[ct, j][t]
                           for ct, j in gbv.max_cap.keys() for t in gbv.period_list if t in gbv.max_cap[ct, j].keys()),
                          name='capacity')
    lb_add_mip.addConstrs(
        (x_vars[glb_ind] == w_vars[glb_ind] * gbv.lot_size[x_ind[glb_ind][0], x_ind[glb_ind][1]] for glb_ind in
         range(len(x_keys))), name='batch')
    lb_add_mip.addConstrs((w_vars[glb_ind] <= gbv.max_prod[x_ind[glb_ind][0], x_ind[glb_ind][1]] / gbv.lot_size[
        x_ind[glb_ind][0], x_ind[glb_ind][1]] for glb_ind in range(len(x_keys))), name='w_ub')

    lb_add_mip.setObjective(gp.quicksum(x_vars[glb_ind] *
                                        global_vars["dual"][gvar_ind][i][glb_ind]
                                        for glb_ind in range(len(x_keys)) for i in range(I)), GRB.MINIMIZE)

    # update the x variables' values
    lb_add_mip.update()
    lb_add_mip.optimize()

    # obtain the objective value (local lower bound) and return
    if lb_add_mip.Status == 2:
        lb_add_obj = lb_add_mip.ObjVal
    else:
        lb_add_obj = -math.inf
    return lb_add_obj


def z_solve_int_solver(local_results, rho, global_vars, var_threshold=1e-5, fixing_round=10):
    '''
    solve z problem with integer variables: solve for the global variables
    local_results: the output from the x problem
    rho: second order penalty coefficient
    '''
    global gbv, global_var_const, global_ind, global_counter

    I = len(local_results)
    # initiate the dual residual calculation
    dual_residual_sqr = 0
    primal_tolerance = 0
    local_combined = []
    global_combined = []

    # create a local dictionary to update global_vars
    local_dict = {}
    local_dict["value"] = [np.zeros(len(item)) for item in global_var_const["keys"]]
    local_dict["dual"] = global_vars["dual"]
    local_dict["fixed"] = global_vars["fixed"]

    solver_start = time.time()
    # construct a MIP to solve for the global variables
    gvar_ind = 0
    global_mip = gp.Model("global_mip")
    global_mip.setParam("Threads", 1)
    # global_mip.setParam("OutputFlag", 0)

    # set up the decision variables to be the global variables and the capacity constraints
    # here this constraint is specific to item decomposition!!!
    x_ind = global_var_const["indices"][gvar_ind]
    x_keys = global_var_const["keys"][gvar_ind]
    assert len(global_var_const["keys"][gvar_ind]) == len(x_keys)
    x_vars = global_mip.addVars(range(len(x_keys)), lb=0.0, name="x_var")  # x_keys should match x_ind
    w_vars = global_mip.addVars(range(len(x_keys)), lb=0.0, vtype=GRB.INTEGER, name="glb_var")

    # set up the capacity constraints
    global_mip.addConstrs((gp.quicksum(
        gbv.unit_cap.get((x_ind[glb_ind][0], x_ind[glb_ind][1], ct), 0) * x_vars[glb_ind] for glb_ind in
        range(len(x_keys)) \
        if (x_ind[glb_ind][1] == j) and (x_ind[glb_ind][2] == t)) <= gbv.max_cap[ct, j][t]
                           for ct, j in gbv.max_cap.keys() for t in gbv.period_list if t in gbv.max_cap[ct, j].keys()),
                          name='capacity')
    global_mip.addConstrs(
        (x_vars[glb_ind] == w_vars[glb_ind] * gbv.lot_size[x_ind[glb_ind][0], x_ind[glb_ind][1]] for glb_ind in
         range(len(x_keys))), name='batch')
    global_mip.addConstrs((w_vars[glb_ind] <= gbv.max_prod[x_ind[glb_ind][0], x_ind[glb_ind][1]] / gbv.lot_size[
        x_ind[glb_ind][0], x_ind[glb_ind][1]] for glb_ind in range(len(x_keys))), name='w_ub')

    # set up the objective function
    local_array_dict = {}
    for i in range(I):
        local_array_dict[i] = np.zeros(len(local_dict["value"][gvar_ind]))
        local_array_dict[i][global_ind[i][gvar_ind]] = local_results[i][gvar_ind]
        local_combined.extend(local_results[i][gvar_ind])
    global_mip.setObjective(gp.quicksum(rho / 2 * (x_vars[glb_ind] - local_array_dict[i][glb_ind]) ** 2 +
                                        (x_vars[glb_ind] - local_array_dict[i][glb_ind]) *
                                        global_vars["dual"][gvar_ind][i][glb_ind]
                                        for glb_ind in range(len(x_keys)) for i in range(I) if
                                        glb_ind in global_ind[i][gvar_ind]), GRB.MINIMIZE)

    # update the x variables' values
    global_mip.update()
    global_mip.optimize()

    solver_time = time.time() - solver_start

    for glb_ind in range(len(x_keys)):
        x_value = x_vars[glb_ind].X
        if x_value > var_threshold:
            local_dict["value"][gvar_ind][glb_ind] = x_value
    global_var_post_proc = local_dict["value"][gvar_ind]
    x_value_consistent = np.zeros(len(x_keys))
    for i in range(I):
        residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars["value"][gvar_ind][
            global_ind[i][gvar_ind]])
        dual_residual_sqr += sum(residual_list ** 2)
        global_combined.extend(global_var_post_proc[global_ind[i][gvar_ind]])

        tem = np.zeros(len(local_dict["value"][gvar_ind]))
        tem[global_ind[i][gvar_ind]] = np.abs(
            local_dict["value"][gvar_ind][global_ind[i][gvar_ind]] - local_array_dict[i][global_ind[i][gvar_ind]])
        x_value_consistent += tem

    # perform the fixing procedure
    # if the x_value is not consistent with local_results
    x_value_consistent /= global_counter[gvar_ind]
    x_bool_consistent = (x_value_consistent <= var_threshold)
    local_dict["fixed"][gvar_ind] = np.minimum(local_dict["fixed"][gvar_ind] * x_bool_consistent + x_bool_consistent,
                                               np.ones(len(x_keys)) * fixing_round)

    gvar_ind = 1
    global_array = np.zeros(len(local_dict["value"][gvar_ind]))
    # global_counter = np.zeros(len(local_dict["value"][gvar_ind]))
    local_array_dict = {}
    for i in range(I):
        if len(global_ind[i][gvar_ind]) > 0:
            # calculate the local information to update the global variables
            local_array = np.zeros(len(local_dict["value"][gvar_ind]))
            local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]
            local_array_dict[i] = local_array

            # the dual value for the non-participating global variables is zero
            global_array += (rho * local_array - global_vars["dual"][gvar_ind][i])

            # count whether the global variable occurs in the i-th subproblem
            # global_counter[global_ind[i][gvar_ind]] += 1

            # record the combined global/local vars to calculate the epsilon
            local_combined.extend(local_results[i][gvar_ind])

    global_var_pre_proc = np.maximum(global_array / (rho * global_counter[gvar_ind]), 0.0)
    # eliminate values that are very close to zero
    global_var_post_proc = np.where(global_var_pre_proc < var_threshold, 0.0, global_var_pre_proc)

    local_dict["value"][gvar_ind] = global_var_post_proc

    r_keys = global_var_const["keys"][gvar_ind]
    r_value_consistent = np.zeros(len(r_keys))
    # calculate the dual residual
    for i in range(I):
        residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars["value"][gvar_ind][
            global_ind[i][gvar_ind]])
        dual_residual_sqr += sum(residual_list ** 2)
        global_combined.extend(global_var_post_proc[global_ind[i][gvar_ind]])

        if len(global_ind[i][gvar_ind]) > 0:
            tem = np.zeros(len(local_dict["value"][gvar_ind]))
            tem[global_ind[i][gvar_ind]] = np.abs(
                local_dict["value"][gvar_ind][global_ind[i][gvar_ind]] - local_array_dict[i][global_ind[i][gvar_ind]])
            r_value_consistent += tem

    r_value_consistent /= global_counter[gvar_ind]
    r_bool_consistent = (r_value_consistent <= var_threshold)
    local_dict["fixed"][gvar_ind] = np.minimum(local_dict["fixed"][gvar_ind] * r_bool_consistent + r_bool_consistent,
                                               np.ones(len(r_keys)) * fixing_round)

    primal_tolerance = np.maximum(LA.norm(local_combined), LA.norm(global_combined))
    return np.sqrt(dual_residual_sqr), primal_tolerance, local_dict, solver_time


def z_solve_lp(local_results, rho, global_vars, var_threshold=1e-5):
    '''
    solve z problem: solve for the global variables
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, global_var_const, global_ind, global_counter

    I = len(local_results)
    # initiate the dual residual calculation
    dual_residual_sqr = 0
    primal_tolerance = 0
    local_combined = []
    global_combined = []

    # create a local dictionary to update global_vars
    local_dict = {}
    local_dict["value"] = [np.zeros(len(item)) for item in global_var_const["keys"]]
    local_dict["dual"] = global_vars["dual"]
    local_dict["fixed"] = global_vars["fixed"]

    # update the global variables
    for gvar_ind in range(len(global_var_const["name"])):
        global_array = np.zeros(len(local_dict["value"][gvar_ind]))
        # global_counter = np.zeros(len(local_dict["value"][gvar_ind]))
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                # calculate the local information to update the global variables
                local_array = np.zeros(len(local_dict["value"][gvar_ind]))
                local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]

                # the dual value for the non-participating global variables is zero
                global_array += (rho * local_array - global_vars["dual"][gvar_ind][i])

                # count whether the global variable occurs in the i-th subproblem
                # global_counter[global_ind[i][gvar_ind]] += 1

                # record the combined global/local vars to calculate the epsilon
                local_combined.extend(local_results[i][gvar_ind])

        global_var_pre_proc = np.maximum(global_array / (rho * global_counter[gvar_ind]), 0.0)
        # eliminate values that are very close to zero
        global_var_post_proc = np.where(global_var_pre_proc < var_threshold, 0.0, global_var_pre_proc)
        # calculate the dual residual
        for i in range(I):
            residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars["value"][gvar_ind][
                global_ind[i][gvar_ind]])
            dual_residual_sqr += sum(residual_list ** 2)
            global_combined.extend(global_var_post_proc[global_ind[i][gvar_ind]])

        local_dict["value"][gvar_ind] = global_var_post_proc

    primal_tolerance = np.maximum(LA.norm(local_combined), LA.norm(global_combined))

    return np.sqrt(dual_residual_sqr), primal_tolerance, local_dict, 0

def int_sub_sol(local_results, global_vars_value):
    global gbv, global_var_const, global_ind

    I = len(local_results)

    hat_global = []
    for gvar_ind in range(len(global_var_const["name"])):
        hat_var = np.zeros(len(global_var_const["keys"][gvar_ind]))

        max_dev = -math.inf
        for i in range(I):
            local_array = np.zeros(len(global_var_const["keys"][gvar_ind]))
            local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]

            local_dev = LA.norm(local_array - global_vars_value[gvar_ind])
            if local_dev > max_dev:
                max_dev = local_dev
                hat_var = local_array.copy()

        hat_global.append(hat_var)

    hat_eta = np.ones(I) * math.inf
    for i in range(I):
        local_eta = np.zeros(I)
        local_eta[:] = local_results[i][2]
        local_eta[i] = math.inf

        hat_eta = np.minimum(hat_eta, local_eta)

    return hat_global, hat_eta

def pi_solve(local_results, rho, global_vars, var_threshold=1e-5, adj_coeff=1.6):
    '''
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, global_ind, global_var_const

    I = len(local_results)
    # initiate the primal residual calculation
    primal_residual_sqr = 0
    dual_tolerance = 0
    dual_combined = []

    # create a local dictionary to update global_vars
    local_dict = {}
    local_dict["value"] = global_vars["value"]
    local_dict["dual"] = global_vars["dual"]
    local_dict["fixed"] = global_vars["fixed"]

    # update the dual variables
    for gvar_ind in range(len(global_var_const["name"])):
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                local_array = np.zeros(len(local_dict["value"][gvar_ind]))
                local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]
                local_array = np.where(local_array < var_threshold, 0.0, local_array)

                # obtain the residual and update the dual
                residual_list = np.zeros(len(local_dict["value"][gvar_ind]))
                residual_list[global_ind[i][gvar_ind]] = global_vars["value"][gvar_ind][global_ind[i][gvar_ind]] - \
                                                         local_array[global_ind[i][gvar_ind]]
                local_dict["dual"][gvar_ind][i] += adj_coeff * rho * residual_list

                # calculate the primal residual
                primal_residual_sqr += sum(residual_list ** 2)

                # record the combined dual vars to calculate the epsilon
                dual_combined.extend(local_dict["dual"][gvar_ind][i][global_ind[i][gvar_ind]])

    dual_tolerance = LA.norm(dual_combined)

    return np.sqrt(primal_residual_sqr), dual_tolerance, local_dict

def extensive_prob(solve_option=True, relax_option=False, solve_remaining_after_fixing=False, global_vars=None, penalty_mag=1e5, fixing_round=10, timelimit=10 * 60):
    # set up the extensive formulation
    # set up the extensive formulation
    global gbv
    ext_prob = gp.Model("extensive_form")
    ext_prob.setParam("Threads", 1)
    ext_prob.setParam("TimeLimit", timelimit)

    # set up model parameters (M: plant, T: time, L: transit,
    u = ext_prob.addVars(gbv.item_list, gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    s = ext_prob.addVars(gbv.transit_list, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    z_p = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z_{ijt} for j,t
    z_m = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z_{ijt} for j,t
    v = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="v")  # v_{ijt} for j,t
    yUI = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUO_p = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O}_{ijt} for j,t
    yUO_m = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O}_{ijt} for j,t
    xC = ext_prob.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t

    # initial condition setup
    for i in gbv.item_list:
        u[i, 0] = 0.0  # initial unmet demand set to 0
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[l], min(gbv.period_list)):
            s[l + (t,)] = 0.0  # initial transportation set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if (i, j) in gbv.init_inv.keys():
                v[i, j, 0] = gbv.init_inv[i, j]  # initial inventory set to given values
            else:
                v[i, j, 0] = 0.0
    for i, j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i, j], min(gbv.period_list)):
            xC[i, j, t] = 0.0  # initial production set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if not ((i, j) in gbv.prod_key):
                for t in gbv.period_list:
                    xC[i, j, t] = 0.0  # production for non-existent item-plant pair set to 0

    rC = ext_prob.addVars(gbv.alt_list, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # add constraints for the extensive formulation
    ext_prob.addConstrs((u[i, t] - u[i, t - 1] + gp.quicksum(z_p[i, j, t] - z_m[i, j, t] for j in gbv.plant_list) \
                         == gbv.real_demand[i, t] for i in gbv.item_list for t in gbv.period_list), name='unmet_demand')
    ext_prob.addConstrs((v[i, j, t] - v[i, j, t - 1] - yUI[i, j, t] + yUO_p[i, j, t] - yUO_m[i, j, t] == 0 \
                         for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list), name='inventory')
    ext_prob.addConstrs((yUI[i, j, t] == gp.quicksum(
        s[l + (t - gbv.transit_time[l],)] for l in gbv.transit_list if (l[0] == i) and (l[2] == j)) +
                         (xC[i, j, t - gbv.lead_time[i, j]] if (i, j) in gbv.prod_key else 0.0) + gbv.external_purchase[
                             i, j, t] \
                         for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                        name='input_item')
    ext_prob.addConstrs(
        (yUO_p[i, j, t] - yUO_m[i, j, t] == gp.quicksum(s[l + (t,)] for l in gbv.transit_list if (l[0] == i) and (l[1] == j)) +
         gp.quicksum(gbv.bom_dict[j][bk] * xC.get((bk[0], j, t), 0.0) for bk in gbv.bom_key[j] if bk[1] == i) + z_p[
             i, j, t] - z_m[i, j, t] +
         gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                     (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i)) -
         gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                     (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
         for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
        name="output_item")
    ext_prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xC[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter)) <= gbv.max_cap[ct, j][t]
                         for ct, j in gbv.max_cap.keys() for t in gbv.period_list), name='capacity')
    ext_prob.addConstrs(
        (rC[jta] <= v[i, jta[0], jta[1] - 1] for i in gbv.item_list for jta in gbv.alt_list if jta[2][0] == i),
        name='r_ub')
    ext_prob.addConstrs(
        (yUO_p[i, j, t] - yUO_m[i, j, t] <= v[i, j, t - 1] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
        name='yo_ub')

    if not (relax_option):
        # if we require an integer number of batches
        wi = ext_prob.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t
        ext_prob.addConstrs(
            (xC[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
            name='batch')
        ext_prob.addConstrs(
            (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
            name='w_ub')
    else:
        # if we can relax the integer constraint
        ext_prob.addConstrs(
            (xC[i, j, t] <= gbv.max_prod[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
            name='capacity')

    # set up the subproblem specific objective
    ext_prob.setObjective(
        gp.quicksum(
            gbv.holding_cost[i] * v[i, j, t] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list) +
        gp.quicksum(gbv.penalty_cost[i] * u[i, t] for i in gbv.item_list for t in gbv.period_list) +
        gp.quicksum(gbv.transit_cost[l] * s[l + (t,)] for l in gbv.transit_list for t in gbv.period_list) +
                    penalty_mag * gp.quicksum(
            z_m[i, j, t] + yUO_m[i, j, t] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
        GRB.MINIMIZE)

    if solve_remaining_after_fixing:
        gvar_ind = 0
        x_ind = [(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
        for xi_ind in range(len(x_ind)):
            if global_vars["fixed"][gvar_ind][xi_ind] == fixing_round:
                fixed_var = xC[x_ind[xi_ind]]
                fixed_var.lb = global_vars["value"][gvar_ind][xi_ind]
                fixed_var.ub = global_vars["value"][gvar_ind][xi_ind]

        gvar_ind = 1
        for ri_ind in range(len(gbv.alt_list)):
            if global_vars["fixed"][gvar_ind][ri_ind] == fixing_round:
                fixed_var = rC[gbv.alt_list[ri_ind]]
                fixed_var.lb = global_vars["value"][gvar_ind][ri_ind]
                fixed_var.ub = global_vars["value"][gvar_ind][ri_ind]

    ext_prob.update()

    if solve_option:
        ext_prob.optimize()
        # collect the solution and objective value
        return ext_prob.objVal
    else:
        return ext_prob, [u, s, z_p, z_m, v, yUI, yUO_p, yUO_m, xC, rC]


def dual_LP_item(dummy_item_index, global_var_value, penalty_mag=1e5):
    global gbv, global_ind, global_var_const, sce_num
    item_ind = gbv.item_list[dummy_item_index]
    print(multiprocessing.current_process(), "\tSetting up Model " + str(item_ind) + "!\n")

    dual_prob = gp.Model("item_{}".format(item_ind))
    dual_prob.setParam("Threads", 1)
    dual_prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    '''
    prod_plant_i = []
    if gbv.item_plant.get(item_ind, -1) != -1:
        prod_plant_i.extend([(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                             (j in gbv.item_plant[item_ind]) and ((i, j) in gbv.prod_key)])
    for j in gbv.plant_list:
        if gbv.bom_key.get(j, -1) != -1:
            prod_plant_i.extend(
                [(i, j) for i in gbv.item_list if ((i, item_ind) in gbv.bom_key[j]) and ((i, j) in gbv.prod_key)])
    prod_plant_i = [*set(prod_plant_i)]

    # prod_plant_i = [(i, j) for i in gbv.item_list for j in gbv.plant_list if \
    #                ((j in gbv.item_plant[item_ind]) and (i, j) in gbv.prod_key) or ((i, item_ind) in gbv.bom_key[j])]

    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]
    '''
    prod_plant_i = [(i, j) for i, j in gbv.prod_key]
    alt_i = [alt_item for alt_item in gbv.alt_list]
    prod_plant_i = [*set(prod_plant_i)]
    alt_i = [*set(alt_i)]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = dual_prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    transit_list_i = [(i_trans[1], i_trans[2]) for i_trans in gbv.transit_list if i_trans[0] == item_ind]
    si = dual_prob.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = dual_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = dual_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = dual_prob.addVars(gbv.plant_list, gbv.period_list, lb=0, name="v")  # v_{ijt} for j,t

    yUIi = dual_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_p = dual_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    yUOi_m = dual_prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,-}_{ijt} for j,t
    xCi = dual_prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    wi = dual_prob.addVars(prod_plant_i, gbv.period_list, lb=0.0, name="w")  # w_{ijt} for i,j,t

    # initial condition setup
    ui[0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv.get((item_ind, j), 0)  # initial inventory set to given values
    for i, j in prod_plant_i:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = dual_prob.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    dual_prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    dual_prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')

    dual_prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if (item_ind, j) in gbv.prod_key),
                    name='input_item_prod')
    dual_prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in transit_list_i if l[1] == j) + gbv.external_purchase[
                         item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if not ((item_ind, j) in gbv.prod_key)),
                    name='input_item_prod')

    dual_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    dual_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == j)
                     + (
                             zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    dual_prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_plant_i)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (t in gbv.max_cap[ct, j].keys())),
                    name='capacity')
    dual_prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    dual_prob.addConstrs(
        (rCi[jta] <= gbv.init_inv.get((jta[2][0], jta[0]), 0) for jta in gbv.alt_list if
         (jta[1] == 1)),
        name='r_ub_rev_ini')
    dual_prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    dual_prob.addConstrs(
        (xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='batch')
    dual_prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    global_ind_i = global_ind[dummy_item_index]

    x_Constrs_dict = {}
    r_Constrs_dict = {}
    # fixing the local variables
    gvar_ind = 0
    for xi_ind in global_ind_i[gvar_ind]:
        x_Constrs_dict[xi_ind] = dual_prob.addConstr(
            xCi[global_var_const["indices"][gvar_ind][xi_ind]] == global_var_value[gvar_ind][xi_ind])

    gvar_ind = 1
    for ri_ind in global_ind_i[gvar_ind]:
        r_Constrs_dict[ri_ind] = dual_prob.addConstr(
            rCi[global_var_const["indices"][gvar_ind][ri_ind]] == global_var_value[gvar_ind][ri_ind])

    time_init = time.time()
    # set up the subproblem specific objective
    # set the linear expression of the local part for the objective
    obj_local_part = gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                                 for j in gbv.plant_list for t in gbv.period_list) + gp.quicksum(
        gbv.penalty_cost[item_ind] * ui[t] for t in gbv.period_list) + \
                     gp.quicksum(gbv.transit_cost[(item_ind,) + l] * si[l + (t,)] for l in transit_list_i for t in
                                 gbv.period_list)

    # objective functio
    dual_prob.setObjective(obj_local_part, GRB.MINIMIZE)

    # solve the dual_problem
    dual_prob.update()

    print(multiprocessing.current_process(), "\tSolving Model " + str(dummy_item_index) + "!\n")

    time_obj = time.time()
    time_setObj = time_obj - time_init
    dual_prob.optimize()
    time_solution = time.time() - time_obj
    print("Objective setup time: {} sec; Solution time: {} sec.".format(time_setObj, time_solution))

    # obtain the solutions and output results
    local_dual_output = []

    gvar_ind = 0
    local_dual_val = []
    for xi_ind in global_ind_i[gvar_ind]:
        local_dual_val.append(x_Constrs_dict[xi_ind].Pi)

    local_dual_output.append(local_dual_val)

    gvar_ind = 1
    local_dual_val = []
    for ri_ind in global_ind_i[gvar_ind]:
        local_dual_val.append(r_Constrs_dict[ri_ind].Pi)
    local_dual_output.append(local_dual_val)

    # obtain the solution and return the subproblem solution
    return local_dual_output


def z_solve_int_solver_bound(local_results, rho, global_vars, var_threshold=1e-5, fixing_param=True, fixing_round=10):
    '''
    solve z problem with integer variables: solve for the global variables
    local_results: the output from the x problem
    rho: second order penalty coefficient
    '''
    global gbv, global_var_const, global_ind, global_counter

    I = len(local_results)
    # initiate the dual residual calculation
    dual_residual_sqr = 0
    primal_tolerance = 0
    local_combined = []
    global_combined = []

    # create a local dictionary to update global_vars
    local_dict = {}
    local_dict["value"] = [np.zeros(len(item)) for item in global_var_const["keys"]]
    local_dict["dual"] = global_vars["dual"]
    local_dict["fixed"] = global_vars["fixed"]

    solver_start = time.time()
    # construct a MIP to solve for the global variables
    gvar_ind = 0
    global_mip = gp.Model("global_mip")
    global_mip.setParam("Threads", 1)
    # global_mip.setParam("OutputFlag", 0)

    # set up the decision variables to be the global variables and the capacity constraints
    # here this constraint is specific to item decomposition!!!
    x_ind = global_var_const["indices"][gvar_ind]
    x_keys = global_var_const["keys"][gvar_ind]
    assert len(global_var_const["keys"][gvar_ind]) == len(x_keys)
    x_vars = global_mip.addVars(range(len(x_keys)), lb=0.0, name="x_var")  # x_keys should match x_ind
    w_vars = global_mip.addVars(range(len(x_keys)), lb=0.0, vtype=GRB.INTEGER, name="glb_var")

    # set up the capacity constraints
    global_mip.addConstrs((gp.quicksum(
        gbv.unit_cap.get((x_ind[glb_ind][0], x_ind[glb_ind][1], ct), 0) * x_vars[glb_ind] for glb_ind in
        range(len(x_keys)) \
        if (x_ind[glb_ind][1] == j) and (x_ind[glb_ind][2] == t)) <= gbv.max_cap[ct, j][t]
                           for ct, j in gbv.max_cap.keys() for t in gbv.period_list if t in gbv.max_cap[ct, j].keys()),
                          name='capacity')
    global_mip.addConstrs(
        (x_vars[glb_ind] == w_vars[glb_ind] * gbv.lot_size[x_ind[glb_ind][0], x_ind[glb_ind][1]] for glb_ind in
         range(len(x_keys))), name='batch')
    global_mip.addConstrs((w_vars[glb_ind] <= gbv.max_prod[x_ind[glb_ind][0], x_ind[glb_ind][1]] / gbv.lot_size[
        x_ind[glb_ind][0], x_ind[glb_ind][1]] for glb_ind in range(len(x_keys))), name='w_ub')

    # set up the upper bound for production at each location
    global_mip.addConstrs(
        (gbv.composition[i, ip] * gbv.lot_size[i] * gp.quicksum(
            w_vars[glb_ind] for glb_ind in range(len(x_keys)) for tau in gbv.period_list if
            (x_ind[glb_ind][0] == i) and (tau <= t)) <= \
         gbv.X_outside_ub_plant[x_ind[glb_ind][0], x_ind[glb_ind][1]][gbv.period_list.index(t)] + gp.quicksum(
            x_vars[item[0], item[1], tp] for item in gbv.prod_key for tp in gbv.period_list if
            (item[0] == ip) and (tp <= t - gbv.lead_time[item])) \
         for i, j in gbv.prod_key for t in gbv.period_list for ip in gbv.subsidiary_list[i]), name='plant_ub'
    )

    global_mip.addConstrs(gp.quicksum(
        gbv.composition[x_ind[glb_ind][0], i] * gbv.lot_size[x_ind[glb_ind][0]] * w_vars[glb_ind] for glb_ind in
        range(len(x_keys)) \
        if (x_ind[glb_ind][0] in gbv.precursor_list[i]) and (x_ind[glb_ind][2] == t)) <= gbv.X_outside_ub[i][
                              gbv.period_list.index(t)] + \
                          gp.quicksum(x_vars[glb_ind] for glb_ind in range(len(x_keys)) if
                                      (x_ind[glb_ind][0] == i) and (x_ind[glb_ind][2] <= t))
                          for i in gbv.item_list for t in gbv.period_list
                          )

    # set up the objective function
    local_array_dict = {}
    for i in range(I):
        local_array_dict[i] = np.zeros(len(local_dict["value"][gvar_ind]))
        local_array_dict[i][global_ind[i][gvar_ind]] = local_results[i][gvar_ind]
        local_combined.extend(local_results[i][gvar_ind])
    global_mip.setObjective(gp.quicksum(rho / 2 * (x_vars[glb_ind] - local_array_dict[i][glb_ind]) ** 2 +
                                        (x_vars[glb_ind] - local_array_dict[i][glb_ind]) *
                                        global_vars["dual"][gvar_ind][i][glb_ind]
                                        for glb_ind in range(len(x_keys)) for i in range(I) if
                                        glb_ind in global_ind[i][gvar_ind]), GRB.MINIMIZE)

    # update the x variables' values
    global_mip.update()
    global_mip.optimize()

    solver_time = time.time() - solver_start

    for glb_ind in range(len(x_keys)):
        x_value = x_vars[glb_ind].X
        if x_value > var_threshold:
            local_dict["value"][gvar_ind][glb_ind] = x_value
    global_var_post_proc = local_dict["value"][gvar_ind]
    x_value_consistent = np.zeros(len(x_keys))
    for i in range(I):
        residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars["value"][gvar_ind][
            global_ind[i][gvar_ind]])
        dual_residual_sqr += sum(residual_list ** 2)
        global_combined.extend(global_var_post_proc[global_ind[i][gvar_ind]])

        tem = np.zeros(len(local_dict["value"][gvar_ind]))
        tem[global_ind[i][gvar_ind]] = np.abs(
            local_dict["value"][gvar_ind][global_ind[i][gvar_ind]] - local_array_dict[i][global_ind[i][gvar_ind]])
        x_value_consistent += tem

    # perform the fixing procedure
    # if the x_value is not consistent with local_results
    x_value_consistent /= global_counter[gvar_ind]
    x_bool_consistent = (x_value_consistent <= var_threshold)
    local_dict["fixed"][gvar_ind] = np.minimum(local_dict["fixed"][gvar_ind] * x_bool_consistent + x_bool_consistent,
                                               np.ones(len(x_keys)) * fixing_round)

    gvar_ind = 1
    global_array = np.zeros(len(local_dict["value"][gvar_ind]))
    # global_counter = np.zeros(len(local_dict["value"][gvar_ind]))
    local_array_dict = {}
    for i in range(I):
        if len(global_ind[i][gvar_ind]) > 0:
            # calculate the local information to update the global variables
            local_array = np.zeros(len(local_dict["value"][gvar_ind]))
            local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]
            local_array_dict[i] = local_array

            # the dual value for the non-participating global variables is zero
            global_array += (rho * local_array - global_vars["dual"][gvar_ind][i])

            # count whether the global variable occurs in the i-th subproblem
            # global_counter[global_ind[i][gvar_ind]] += 1

            # record the combined global/local vars to calculate the epsilon
            local_combined.extend(local_results[i][gvar_ind])

    global_var_pre_proc = np.maximum(global_array / (rho * global_counter[gvar_ind]), 0.0)
    # eliminate values that are very close to zero
    global_var_post_proc = np.where(global_var_pre_proc < var_threshold, 0.0, global_var_pre_proc)

    local_dict["value"][gvar_ind] = global_var_post_proc

    r_keys = global_var_const["keys"][gvar_ind]
    r_value_consistent = np.zeros(len(r_keys))
    # calculate the dual residual
    for i in range(I):
        residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars["value"][gvar_ind][
            global_ind[i][gvar_ind]])
        dual_residual_sqr += sum(residual_list ** 2)
        global_combined.extend(global_var_post_proc[global_ind[i][gvar_ind]])

        if len(global_ind[i][gvar_ind]) > 0:
            tem = np.zeros(len(local_dict["value"][gvar_ind]))
            tem[global_ind[i][gvar_ind]] = np.abs(
                local_dict["value"][gvar_ind][global_ind[i][gvar_ind]] - local_array_dict[i][global_ind[i][gvar_ind]])
            r_value_consistent += tem

    r_value_consistent /= global_counter[gvar_ind]
    r_bool_consistent = (r_value_consistent <= var_threshold)
    local_dict["fixed"][gvar_ind] = np.minimum(local_dict["fixed"][gvar_ind] * r_bool_consistent + r_bool_consistent,
                                               np.ones(len(r_keys)) * fixing_round)

    primal_tolerance = np.maximum(LA.norm(local_combined), LA.norm(global_combined))
    return np.sqrt(dual_residual_sqr), primal_tolerance, local_dict, solver_time


if __name__ == '__main__':
    data_folder = "data/fine_tune"
    # process the hyperparameters
    hparams = ADMMparams("src/params.json")
    x_solve = eval("x_solve_int_BD_cut")
    z_solve = eval("z_solve_int_solver")
    pi_solve = eval("pi_solve")

    # record the primal/dual residual
    pri_re = []
    d_re = []

    # initialize the iteration, primal residual, dual residual
    iter_no = 0
    primal_residual = hparams.init_pr
    dual_residual = hparams.init_dr

    # create shared information and manager
    gbv = create_gbv(data_folder)

    global_vars = global_var_init_item(gbv)
    global_var_const, global_ind, sqrt_dim, global_counter = global_const_init(gbv)

    # objective of remaining problem after fixing plenty enough variables
    remaining_after_fixed_solver_obj = math.inf

    # create and configure the process pool
    pool = Pool(processes=min(len(gbv.item_list), 30), initializer=x_item_init,
                initargs=(gbv, global_var_const, global_ind, sqrt_dim, global_counter))

    UB = math.inf
    LB = -math.inf
    rho = 10.0  # rho = 10.0
    rel_tol = hparams.rel_tol

    admm_time = 0

    ext_prob, ext_prob_handle = extensive_prob(solve_option=False, relax_option=True)
    ext_prob.optimize()
    xC = ext_prob_handle[8]
    rC = ext_prob_handle[9]

    x_ind = [(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    r_ind = gbv.alt_list

    x_val = np.zeros(len(x_ind))
    for i in range(len(x_ind)):
        x_val[i] = xC[x_ind[i]].X
        global_vars["value"][0][i] = xC[x_ind[i]].X

    r_val = np.zeros(len(r_ind))
    for i in range(len(r_ind)):
        r_val[i] = rC[r_ind[i]].X
        global_vars["value"][1][i] = rC[r_ind[i]].X

    global_var_sol = [x_val, r_val]
    local_LP_dual_vals = pool.map(partial(dual_LP_item, global_var_value=global_var_sol), range(len(gbv.item_list)))

    global_vars_dual = []
    for gvar_ind in range(len(global_var_const["keys"])):
        gvar_dual = []
        dual_size = len(global_var_const["keys"][gvar_ind])
        for i in range(len(gbv.item_list)):
            local_gvar_dual = np.zeros(dual_size)
            local_gvar_dual[global_ind[i][gvar_ind]] = local_LP_dual_vals[i][gvar_ind]

            gvar_dual.append(local_gvar_dual)

        global_vars_dual.append(gvar_dual)
    global_vars["dual"] = global_vars_dual

    # Benders cut
    BD_cut = dict()

    # initial round of solution of ADMM
    ini_start = time.time()
    local_results = pool.map(partial(x_solve, rho=rho, global_vars=global_vars, BD_cut=BD_cut, UB=UB, LB=LB, quad_penalty=True),
                             range(len(gbv.item_list)))
    global_vars = z_solve(local_results, rho, global_vars)[2]
    ini_end = time.time()

    # update bounds
    ub_results = pool.map(partial(x_solve_ub, global_vars=global_vars), range(len(gbv.item_list)))
    if not (np.inf in ub_results):
        UB = np.minimum(sum(ub_results), UB)

    lb_results = pool.map(partial(x_solve_lb, global_vars=global_vars), range(len(gbv.item_list)))
    if not ((-np.inf) in lb_results):
        lb_add_obj = lb_add(global_vars)
        if lb_add_obj > -math.inf:
            LB = np.maximum(sum(lb_results) + lb_add_obj, LB)

    # generate hat_x, hat_r and hat_eta from local results
    hat_global, hat_eta = int_sub_sol(local_results, global_vars["value"])
    # solve benders-sub with respect to hat_x, hat_r, hat_eta to generate BD cuts
    local_BD_dual = pool.map(partial(BD_x_sub, hat_global=hat_global, hat_eta=hat_eta), range(len(gbv.item_list)))
    BD_cut[iter_no] = dict()
    BD_cut[iter_no]["global_vars_value"] = hat_global # x_hat,r_hat
    BD_cut[iter_no]["second_stage_obj"] = hat_eta   # eta_hat
    for i in range(len(gbv.item_list)):
        if local_BD_dual[i] != None:
            # BD cuts coefficients
            gapi = local_BD_dual[i][0]

            x_dual = np.zeros(len(global_var_const["keys"][0]))
            x_dual[:] = local_BD_dual[i][1]

            r_dual = np.zeros(len(global_var_const["keys"][1]))
            r_dual[:] = local_BD_dual[i][2]

            BD_cut[iter_no][gbv.item_list[i]] = [gapi, x_dual, r_dual, local_BD_dual[i][3]]  # gapi, dual_x, dual_r, dual_eta

    admm_time += ini_end - ini_start

    primal_residual_record = []
    dual_residual_record = []

    primal_tol = hparams.p_tol
    dual_tol = hparams.d_tol

    while primal_residual > 1e-2 or dual_residual > 1e-2:
        iter_no += 1

        # set up the termination criterion
        if iter_no > hparams.iter_tol:
            break
        else:
            if UB < math.inf and LB > -math.inf:
                rel_err = (UB - LB) / (UB + 1e-10)
                if rel_err < rel_tol:
                    break

        iter_start_time = time.time()

        # solve x problem in parallel
        #local_results = pool.map(partial(x_solve, rho=rho, global_vars=global_vars), range(len(gbv.item_list)))
        local_results = pool.map(
            partial(x_solve, rho=rho, global_vars=global_vars, BD_cut=BD_cut, UB=UB, LB=LB, quad_penalty=True),
            range(len(gbv.item_list)))

        if None in local_results:
            print("Sub unsolvable!")
            break
        # admm_obj = 0
        # for i in range(len(local_results)):
        #    admm_obj += local_results[i][2]

        # solve z problem, obtain dual residual
        dual_residual, primal_tol_norm, global_vars, z_time = z_solve(local_results, rho, global_vars)
        # dp_time = z_solve_int_dp(local_results, rho, global_vars)[-1]

        # update dual variables, obtain primal residual
        primal_residual, dual_tol_norm, global_vars = pi_solve(local_results, rho, global_vars)

        # upper bound and lower bound calculation
        if iter_no > hparams.LP_warm_start_rounds:
            ub_results = pool.map(partial(x_solve_ub, global_vars=global_vars), range(len(gbv.item_list)))
            if not (np.inf in ub_results):
                UB = np.minimum(sum(ub_results), UB)

            lb_results = pool.map(partial(x_solve_lb, global_vars=global_vars), range(len(gbv.item_list)))
            if not ((-np.inf) in lb_results):
                lb_add_obj = lb_add(global_vars)
                if lb_add_obj > -math.inf:
                    LB = np.maximum(sum(lb_results) + lb_add_obj, LB)

                # to calculate the lower bound, the global part of the objective includes the global variables
                # the sum of dual variables should be 0 for LP, for IP?

                # lambda_list = []
                # for gvar_ind in range(len(global_var_const["name"])):
                #    lambda_list.append(np.sum(global_vars["dual"][i][gvar_ind] for i in range(len(gbv.item_list))))
                # assert np.sum(lambda_list) < 1e-4

        # generate hat_x, hat_r and hat_eta from local results
        hat_global, hat_eta = int_sub_sol(local_results, global_vars["value"])
        # generate BD cuts from hat_x, hat_r, hat_eta
        local_BD_dual = pool.map(partial(BD_x_sub, hat_global=hat_global, hat_eta=hat_eta),
                                    range(len(gbv.item_list)))
        if "unsolvable" in local_BD_dual:
            print("BD dual unsolvable!")
            break

        BD_cut[iter_no] = dict()
        BD_cut[iter_no]["global_vars_value"] = hat_global  # x_hat,r_hat
        BD_cut[iter_no]["second_stage_obj"] = hat_eta
        for i in range(len(gbv.item_list)):
            if local_BD_dual[i] != None:
                gapi = local_BD_dual[i][0]

                x_dual = np.zeros(len(global_var_const["keys"][0]))
                x_dual[:] = local_BD_dual[i][1]

                r_dual = np.zeros(len(global_var_const["keys"][1]))
                r_dual[:] = local_BD_dual[i][2]

                BD_cut[iter_no][gbv.item_list[i]] = [gapi, x_dual, r_dual,
                                                     local_BD_dual[i][3]]  # gapi, dual_x, dual_r, dual_eta

        # calculate the primal/dual tolerance
        primal_tol = sqrt_dim * hparams.eps_abs + hparams.eps_rel * primal_tol_norm
        dual_tol = sqrt_dim * hparams.eps_abs + hparams.eps_rel * dual_tol_norm
        primal_residual_record.append(primal_residual)
        dual_residual_record.append(dual_residual)
        iter_elapsed_time = time.time() - iter_start_time

        admm_time += iter_elapsed_time
        print("------- Iteration {}, Primal Residual = {}, Dual Residual = {}, Rho = {}, iter_time = {}".format(iter_no,
                                                                                                                np.round(
                                                                                                                    primal_residual,
                                                                                                                    2),
                                                                                                                np.round(
                                                                                                                    dual_residual,
                                                                                                                    2),
                                                                                                                rho,
                                                                                                                np.round(
                                                                                                                    iter_elapsed_time,
                                                                                                                    2)))

        # update the rho penalty term
        if (iter_no <= hparams.LP_warm_start_rounds) and (
                primal_residual > 10 * dual_residual):  # and (primal_residual > primal_tol):
            # pass
            rho *= 2
        elif (iter_no <= hparams.LP_warm_start_rounds) and (
                dual_residual > 10 * primal_residual):  # and (dual_residual > dual_tol):
            # pass
            rho /= 2

        if iter_no > hparams.LP_warm_start_rounds:
            z_solve = eval("z_solve_int_solver")

        # fixed rate
        fixed_var_num = 0
        total_var_size = 0
        for gvar_ind in range(len(global_var_const["name"])):
            fixed_var_num += np.count_nonzero(
                global_vars["fixed"][gvar_ind] == hparams.fixing_rounds)
            total_var_size += global_vars["fixed"][gvar_ind].size

        fixed_rate = fixed_var_num / total_var_size * 100
        # if fixed variables are plenty enough, solve the remaining problem
        if fixed_rate > 80:
            remaining_after_fixed_solver_obj = extensive_prob(solve_option=True, relax_option=False,
                                                              solve_remaining_after_fixing=True,
                                                              global_vars=global_vars)
            break

        time_str = "Solver"  # "Solver"
        if iter_no % 5 == 0:
            wr = "\nIteration: " + str(iter_no) + "\nUpper bound : " + str(UB) + "\nLower bound : " + str(
                LB) + "\nRho :" + str(rho) + "\nPrimal residual : " + str(
                np.round(primal_residual, 2)) + '\nDual residual : ' + str(
                np.round(dual_residual, 2)) + "\nIteration time : " + str(
                np.round(iter_elapsed_time, 2)) + "\n" + time_str + " time : " + str(np.round(z_time, 2)) + \
                 "\nADMM time used : " + str(np.round(admm_time, 2)) + \
                 "\nVariable fixed rate(%) : " + str(np.round(fixed_rate, 2)) + "\n"
            wr_s = open('results.txt', 'a')
            wr_s.write(wr)
            wr_s.close()

    if iter_no % 5 != 1:
        wr = "\nIteration: " + str(iter_no - 1) + "\nUpper bound : " + str(UB) + "\nLower bound : " + str(
            LB) + "\nRho :" + str(rho) + "\nPrimal residual : " + str(
            np.round(primal_residual, 2)) + '\nDual residual : ' + str(
            np.round(dual_residual, 2)) + "\nIteration time : " + str(
            np.round(iter_elapsed_time, 2)) + "\n" + time_str + " time : " + str(np.round(z_time, 2)) + \
             "\nVariable fixed rate(%) : " + str(
            np.round(fixed_rate, 2)) + "\nFinished !\n"
        wr_s = open('results.txt', 'a')
        wr_s.write(wr)
        wr_s.close()

    # wr_s = open('global_var.txt', 'a')
    # wr_s.write(wr)
    # wr_s.close()

    # MIP solver solution
    # solver_obj = extensive_prob(True, False, False, round(admm_time))

    wr = "\nADMM time :" + str(np.round(admm_time, 2))
    # + "\nGurobi objective: " + str(solver_obj)
    if remaining_after_fixed_solver_obj < math.inf:
        wr += "\nObjective of remaining problem after fixing plenty enough variables : " + str(
            remaining_after_fixed_solver_obj)
    wr_s = open('results.txt', 'a')
    wr_s.write(wr)
    wr_s.close()

    wr = ""
    for max_BD_ite in BD_cut.keys():
        wr += "\nIteration :" + str(max_BD_ite)
        for i in BD_cut[max_BD_ite].keys():
            if (i != "global_vars_value") and (i != "second_stage_obj"):
                wr += "\nItem " + str(i) + "\nx_dual: "
                for x_dual in BD_cut[max_BD_ite][i][1]:
                    if abs(x_dual) > 0.0:
                        wr += str(x_dual) + ",\t"
                wr += "\nr_dual: "
                for r_dual in BD_cut[max_BD_ite][i][2]:
                    if abs(r_dual) > 0.0:
                        wr += str(r_dual) + ",\t"
                wr += "\neta_dual: " + str(BD_cut[max_BD_ite][i][3]) + "\n"

    wr_s = open('results.txt', 'a')
    wr_s.write(wr)
    wr_s.close()


    solver_start = time.time()
    #solver_obj = extensive_prob(solve_option=True, relax_option=False, solve_remaining_after_fixing=False)
    solver_obj = math.inf
    solver_end = time.time()

    wr = "\nGurobi objective: " + str(solver_obj)
    wr_s = open('results.txt', 'a')
    wr_s.write(wr)
    wr_s.close()
