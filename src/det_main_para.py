# -*- coding: utf-8 -*-
"""
Created on Tue June 13, 2023

@author: haoxiang
"""

'''
main structure of running decomposition instances in parallel
'''

import os
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

    # zero padding for the demand and external purchase
    # we need to change the data into a numpy array form, or maybe sparse tensor
    for i in gbv.item_list:
        for j in gbv.plant_list:
            for t in gbv.period_list:
                if not((i,j,t) in gbv.external_purchase.keys()):
                    gbv.external_purchase[i,j,t] = 0.0
        for t in gbv.period_list:
            if not((i,t) in gbv.real_demand.keys()):
                gbv.real_demand[i,t] = 0.0

    return gbv


def global_var_init_item(gbv):
    x_keys = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    r_keys = ["r[{},{},{}]".format(*r_rel) for r_rel in gbv.alt_list]

    # initialize the variable values
    global_vars["value"] = [np.zeros(len(x_keys)), np.zeros(len(r_keys))]

    # initialize the variable duals
    global_vars["dual"] = [[np.zeros(len(x_keys)) for i in gbv.item_list], [np.zeros(len(r_keys)) for i in gbv.item_list]]


def global_const_init(gbv):
    global_var_const = {}
    global_var_const["name"] = ["x", "r"]
    x_keys = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in gbv.prod_key for t in gbv.period_list]
    r_keys = ["r[{},{},{}]".format(*r_rel) for r_rel in gbv.alt_list]
    global_var_const["keys"] = [x_keys, r_keys]

    global_ind = []

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
        assert len(global_names_i) == len(global_var_const["name"])

        # obtain the global variable index
        global_ind_i = []
        for varI in range(len(global_var_const["name"])):
            var_ind = [global_var_const["keys"][varI].index(global_names_i[varI][j]) for j in range(len(global_names_i[varI]))]
            global_ind_i.append(var_ind)
        global_ind.append(global_ind_i)

    global sqrt_dim
    sqrt_dim = np.sqrt(sum(len(global_ind[i][gvar_ind]) for gvar_ind in range(len(global_var_const["name"])) for i in range(len(global_ind))))

    return global_var_const, global_ind, sqrt_dim

def x_item(item_ind, relax_option = True):
    '''
    item_ind: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    global gbv
    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)
    prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    prod_plant_i = [(i,j) for i in gbv.item_list for j in gbv.plant_list if \
                    ((j in gbv.item_plant[item_ind]) and (i,j) in gbv.prod_key) or ((i,item_ind) in gbv.bom_key[j])]
    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb = 0.0, name="u")  # u_{it} for t, unmet demand
    si = prob.addVars(gbv.transit_list, gbv.period_list, lb = 0.0, name="s")  # s_{ilt} for l,t
    zi = prob.addVars(gbv.plant_list, gbv.period_list, name="z")  # z_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb = 0.0,
                      ub = sum([gbv.max_prod[item_ind,j] for j in gbv.plant_list]) * len(gbv.period_list) * 10, name="v")  # v_{ijt} for j,t
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
    prob.addConstrs((gp.quicksum(gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
                                 if (j_iter == j) and (ct == ct_iter)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (item_ind, j) in gbv.prod_key), name='capacity')
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

def x_item_feas(item_ind, relax_option = True, penalty_mag = 1e5):
    '''
    A reformulation of the sub problem to guarantee feasibility for the sub problems
    item_ind: the index of sub-problem
    return: the subproblem built with Gurobi
    '''
    global gbv
    prob = gp.Model("item_{}".format(item_ind))
    prob.setParam("Threads", 1)
    prob.setParam("OutputFlag", 0)

    # find the prod_key (i_share,j) that shares the plant with item i or has bom relationship with item i
    prod_plant_i = [(i, j) for i in gbv.item_list for j in gbv.plant_list if \
                    ((j in gbv.item_plant[item_ind]) and (i, j) in gbv.prod_key) or ((i, item_ind) in gbv.bom_key[j])]
    alt_i = [alt_item for alt_item in gbv.alt_list if item_ind in alt_item[2]]

    # set up model parameters (M: plant, T: time, L: transit,
    ui = prob.addVars(gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    si = prob.addVars(gbv.transit_list, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    zi_p = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t
    zi_m = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^-_{ijt} for j,t
    vi = prob.addVars(gbv.plant_list, gbv.period_list, lb=0.0,
    ub = sum([gbv.max_prod[item_ind, j] for j in gbv.plant_list]) * len(
        gbv.period_list) * 10, name = "v")  # v_{ijt} for j,t
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
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[(item_ind,) + l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for j in gbv.plant_list:
        vi[j, 0] = gbv.init_inv[item_ind, j]  # initial inventory set to given values
    for i, j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i, j], min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = prob.addVars(alt_i, lb = 0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    prob.addConstrs((ui[t] - ui[t - 1] + gp.quicksum(zi_p[j, t] - zi_m[j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[item_ind, t] for t in gbv.period_list), name='unmet_demand')
    prob.addConstrs((vi[j, t] - vi[j, t - 1] - yUIi[j, t] + yUOi_p[j, t] - yUOi_m[j, t] == 0 for j in gbv.plant_list \
                     for t in gbv.period_list), name='inventory')
    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in gbv.transit_list if l[1] == j) +
                     xCi[item_ind, j, t - gbv.lead_time[item_ind, j]] + gbv.external_purchase[item_ind, j, t] \
                     for j in gbv.plant_list for t in gbv.period_list if j in gbv.item_plant[item_ind]),
                    name='input_item_prod')
    prob.addConstrs((yUIi[j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[(item_ind,) + l],)] for l in gbv.transit_list if l[1] == j) +
                     gbv.external_purchase[item_ind, j, t] for j in gbv.plant_list for t in gbv.period_list if
                     not (j in gbv.item_plant[item_ind])), name = 'input_item_non_prod')
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] == gp.quicksum(si[l + (t,)] for l in gbv.transit_list if l[0] == j) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == item_ind) + (zi_p[j, t] - zi_m[j, t]) +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == item_ind)) -
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == item_ind))
                     for j in gbv.plant_list for t in gbv.period_list), name="output_item")
    prob.addConstrs((gp.quicksum(gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
                                 if (j_iter == j) and (ct == ct_iter)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list if (item_ind, j) in gbv.prod_key), name='capacity')
    prob.addConstrs((rCi[jta] <= vi[jta[0], jta[1] - 1] for jta in gbv.alt_list if jta[2][0] == item_ind), name='r_ub')
    prob.addConstrs(
        (rCi[jta] <= gbv.init_inv[jta[2][0], jta[0]] for jta in gbv.alt_list if (jta[2][1] == item_ind) and (jta[1] == 1)),
        name='r_ub_rev_ini')
    prob.addConstrs((yUOi_p[j, t] - yUOi_m[j, t] <= vi[j, t - 1] for j in gbv.plant_list for t in gbv.period_list), name='yo_ub')
    prob.addConstrs((xCi[i, j, t] == wi[i, j, t] * gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
                    name='batch')
    prob.addConstrs(
        (wi[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in prod_plant_i for t in gbv.period_list),
        name='w_ub')

    # set up the subproblem specific objective
    theta = prob.addVar(lb=0.0, name='theta')
    prob.addConstr(theta == gp.quicksum(gbv.holding_cost[item_ind] * vi[j, t] + penalty_mag * (zi_m[j, t] + yUOi_m[j, t]) \
                    for j in gbv.plant_list for t in gbv.period_list) + gp.quicksum(gbv.penalty_cost[item_ind] * ui[t] \
                    for t in gbv.period_list), name = 'obj_local')

    prob.update()
    return prob


def x_item_init(global_info, global_dict, global_dict_const, global_ind_list, x_prob_model, dim):
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
    global global_var_const, global_vars, global_ind, prob_list, prob_list_lb, prob_list_ub, sqrt_dim
    global_vars = global_dict
    global_var_const = global_dict_const
    global_ind = global_ind_list
    prob_list = []
    prob_list_lb = []
    prob_list_ub = []
    for i in gbv.item_list:
        # construct the local problems
        prob_list.append(x_prob_model(i))
        prob_list_lb.append(x_prob_model(i))
        prob_list_ub.append(x_prob_model(i))

    sqrt_dim = dim


def x_solve(i, rho, quad_penalty = True):
    '''
    solve x problems: solve for local variables
    i: the index of sub-problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, global_ind, prob_list, global_var_const

    prob = prob_list[i]
    global_ind_i = global_ind[i]

    # set the linear expression of the local part for the objective
    obj_local_part = prob.getVarByName("theta")

    time_init = time.time()

    # objective function
    if quad_penalty:
        # if the objective contains the quadratic penalty term (bundle method)
        prob.setObjective(obj_local_part -
                          gp.quicksum(global_vars["dual"][gvar_ind][i][vi] * prob.getVarByName(global_var_const["keys"][gvar_ind][vi]) \
                                      for gvar_ind in range(len(global_var_const["name"])) \
                                      for vi in global_ind_i[gvar_ind]) + # add all global variables for relaxation part
                          rho / 2 * gp.quicksum((global_vars["value"][gvar_ind][vi] - prob.getVarByName(global_var_const["keys"][gvar_ind][vi])) ** 2 \
                                                for gvar_ind in range(len(global_var_const["name"])) \
                                                for vi in global_ind_i[gvar_ind]), GRB.MINIMIZE)
    else:
        # if the objective does not contain the quadratic penalty term (Lagrangian method)
        prob.setObjective(obj_local_part -
                          gp.quicksum(global_vars["dual"][gvar_ind][i][vi] * prob.getVarByName(global_var_const["keys"][gvar_ind][vi]) \
                              for gvar_ind in range(len(global_var_const["name"])) \
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
    for gvar_ind in range(len(global_var_const["name"])):
        local_val = []
        for vi in global_ind_i[gvar_ind]:
            local_val.append(prob.getVarByName(global_var_const["keys"][gvar_ind][vi]).X)
        local_output.append(local_val)

    # obtain the solution and return the subproblem solution
    return local_output


def x_solve_lb(i):
    '''
    solve the x problems' relaxation: obtain the lower bound
    i: the index of sub-problem
    '''
    global gbv, global_ind, prob_list_lb, global_var_const
    prob = prob_list_lb[i]
    global_ind_i = global_ind[i]

    # set the linear expression of the local part for the objective
    obj_local_part = prob.getVarByName("theta")

    time_init = time.time()
    prob.setObjective(obj_local_part - gp.quicksum(global_vars["dual"][gvar_ind][i][vi] * prob.getVarByName(
                          global_var_const["keys"][gvar_ind][vi]) for gvar_ind in range(len(global_var_const["name"])) \
                                                   for vi in global_ind_i[gvar_ind]), GRB.MINIMIZE)
    # solve the problem
    prob.update()
    time_obj = time.time()
    time_setObj = time_obj - time_init
    prob.optimize()
    time_solution = time.time() - time_obj
    print("LB problem objective setup time: {} sec; Solution time: {} sec.".format(time_setObj, time_solution))

    # obtain the objective value (local lower bound) and return
    if prob.Status == 2:
        local_obj = prob.ObjVal
    else:
        local_obj = -np.infty
    return local_obj


def x_solve_ub(i):
    '''
        solve the x problems' by fixing the global variables' values: obtain the upper bound
        i: the index of sub-problem
    '''
    global gbv, global_ind, prob_list_ub, global_var_const
    prob = prob_list_ub[i]
    global_ind_i = global_ind[i]

    time_init = time.time()

    # fix the local variables to their global counterparts' values
    for gvar_ind in range(len(global_var_const["name"])):
        for vi in global_ind_i[gvar_ind]:
            local_var = prob.getVarByName(global_var_const["keys"][gvar_ind][vi])
            local_var.lb = global_vars["value"][gvar_ind][vi]
            local_var.ub = global_vars["value"][gvar_ind][vi]

    obj_local_part = prob.getVarByName("theta")
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


def z_solve(local_results, rho, var_threshold = 1e-6):
    '''
    solve z problem: solve for the global variables
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, global_var_const, global_ind

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

    # update the global variables
    for gvar_ind in range(len(global_var_const["name"])):
        global_array = np.zeros(len(local_dict["value"][gvar_ind]))
        global_counter = np.zeros(len(local_dict["value"][gvar_ind]))
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                # calculate the local information to update the global variables
                local_array = np.zeros(len(local_dict["value"][gvar_ind]))
                local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]

                # the dual value for the non-participating global variables is zero
                global_array += (rho * local_array - global_vars["dual"][gvar_ind][i])

                # count whether the global variable occurs in the i-th subproblem
                global_counter[global_ind[i][gvar_ind]] += 1

                # record the combined global/local vars to calculate the epsilon
                local_combined.extend(local_results[i][gvar_ind])

        global_var_pre_proc = np.maximum(global_array / (rho * global_counter), 0.0)
        # eliminate values that are very close to zero
        global_var_post_proc = np.where(global_var_pre_proc < var_threshold, 0.0, global_var_pre_proc)
        # calculate the dual residual
        for i in range(I):
            residual_list = rho * (global_var_post_proc[global_ind[i][gvar_ind]] - global_vars["value"][gvar_ind][global_ind[i][gvar_ind]])
            dual_residual_sqr += sum(residual_list**2)
            global_combined.extend(global_var_post_proc[global_ind[i][gvar_ind]])

        local_dict["value"][gvar_ind] = global_var_post_proc

    primal_tolerance = np.maximum(LA.norm(local_combined), LA.norm(global_combined))
    global_vars.update(local_dict)
    return np.sqrt(dual_residual_sqr), primal_tolerance


def pi_solve(local_results, rho, var_threshold = 1e-6, adj_coeff = 1.6):
    '''
    local_results: the output from the x problem
    global_ind: a list with each item corresponding to the indices of the global variables selected for the sub
    global_var: a list global variable handles
    rho: second order penalty coefficient
    '''
    global gbv, prob_list, global_ind, global_var_const

    I = len(local_results)
    # initiate the primal residual calculation
    primal_residual_sqr = 0
    dual_tolerance = 0
    dual_combined = []

    # create a local dictionary to update global_vars
    local_dict = {}
    local_dict["value"] = global_vars["value"]
    local_dict["dual"] = global_vars["dual"]

    # update the dual variables
    for gvar_ind in range(len(global_var_const["name"])):
        for i in range(I):
            if len(global_ind[i][gvar_ind]) > 0:
                local_array = np.zeros(len(local_dict["value"][gvar_ind]))
                local_array[global_ind[i][gvar_ind]] = local_results[i][gvar_ind]

                # obtain the residual and update the dual
                residual_list = global_vars["value"][gvar_ind] - np.where(local_array < var_threshold, 0.0, local_array)
                local_dict["dual"][gvar_ind][i] += adj_coeff * rho * residual_list

                # calculate the primal residual
                primal_residual_sqr += sum(residual_list**2)

                # record the combined dual vars to calculate the epsilon
                dual_combined.extend(local_dict["dual"][gvar_ind][i][global_ind[i][gvar_ind]])

    dual_tolerance = LA.norm(dual_combined)
    global_vars.update(local_dict)

    return np.sqrt(primal_residual_sqr), dual_tolerance


# protect the entry point
if __name__ == '__main__':
    data_folder = "../data/small_test"
    # process the hyperparameters
    hparams = ADMMparams("params.json")
    x_prob_model = eval("x_item_feas")
    x_solve = eval("x_solve")
    z_solve = eval("z_solve")
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
    global_vars = Manager().dict()

    global_var_const, global_ind, sqrt_dim = global_const_init(gbv)
    global_var_init_item(gbv)

    # create and configure the process pool
    pool = Pool(processes = len(gbv.item_list), initializer=x_item_init, initargs=(gbv, global_vars, global_var_const, global_ind, x_prob_model, sqrt_dim))

    UB = math.inf
    LB = -math.inf
    rho = 10.0
    rel_tol = hparams.rel_tol

    # obtain the initial z solution
    local_results = pool.map(partial(x_solve, rho = rho, quad_penalty = False), range(len(gbv.item_list)))
    z_solve(local_results, rho)

    primal_residual_record = []
    dual_residual_record = []

    while primal_residual > hparams.p_tol or dual_residual > hparams.d_tol:
       iter_no += 1
       iter_start_time = time.time()

       # set up the termination criterion
       if iter_no > hparams.iter_tol:
           break
       else:
           if UB < math.inf and LB > -math.inf:
               rel_err = (UB - LB) / (UB + 1e-10)
               if rel_err < rel_tol:
                   break

       # solve x problem in parallel
       local_results = pool.map(partial(x_solve, rho=rho), range(len(gbv.item_list)))

       # solve z problem, obtain dual residual
       dual_residual, primal_tol_norm = z_solve(local_results, rho)

       # update dual variables, obtain primal residual
       primal_residual, dual_tol_norm = pi_solve(local_results, rho)

       # upper bound and lower bound calculation
       lb_results = pool.map(partial(x_solve_lb), range(len(gbv.item_list)))
       ub_results = pool.map(partial(x_solve_ub), range(len(gbv.item_list)))
       if not((-np.inf) in lb_results):
           LB = np.maximum(sum(lb_results), LB)
       if not(np.inf in ub_results):
           UB = np.minimum(sum(ub_results), UB)

       # calculate the primal/dual tolerance
       primal_tol = sqrt_dim * hparams.eps_abs + hparams.eps_rel * primal_tol_norm
       dual_tol = sqrt_dim * hparams.eps_abs + hparams.eps_rel * dual_tol_norm
       primal_residual_record.append(primal_residual)
       dual_residual_record.append(dual_residual)
       iter_elapsed_time = time.time() - iter_start_time
       print("------- Iteration {}, Primal Residual = {}, Dual Residual = {}, Rho = {}, iter_time = {}".format(iter_no, np.round(primal_residual, 2), np.round(dual_residual, 2), rho, np.round(iter_elapsed_time,2)))

       # update the rho penalty term
       if (primal_residual > 10 * dual_residual) and (primal_residual > primal_tol):
           rho *= 2
       elif (dual_residual > 10 * primal_residual) and (dual_residual > dual_tol):
           rho /= 2

    print(LB, UB)