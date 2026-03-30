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

#from src.params import ADMMparams
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
    gbv.real_demand = input_demand(os.path.join(path_file, "dm_df_demand.csv"))
    # gbv.real_demand, gbv.forecast_demand

    gbv = timeline_adjustment(gbv)
    gbv = cap_adjustment(gbv)

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

    # bundle items with replacement relationship together to create new decomposition units (before decompose wrt single items)
    a_list = []
    for jta in gbv.alt_list:
        if jta[2] not in a_list:
            a_list.append(jta[2])

    replace_items = []
    bundle_list = []
    for a in a_list:
        replace_items.extend([a[0], a[1]])
        if [min(a[0], a[1]), max(a[0], a[1])] not in bundle_list:
            bundle_list.append([min(a[0], a[1]), max(a[0], a[1])])
    replace_items = [*set(replace_items)]

    decomp_units = []
    for i in gbv.item_list:
        if i not in replace_items:
            decomp_units.append([i])
    decomp_units.extend(bundle_list)
    gbv.decomp_units = decomp_units

    return gbv


def extensive_prob(relax_option=False, penalty_mag=1e5):
    # set up the extensive formulation
    # set up the extensive formulation
    global gbv
    ext_prob = gp.Model("extensive_form")
    ext_prob.setParam("Threads", 1)
    ext_prob.Params.TimeLimit = 30 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    u = ext_prob.addVars(gbv.item_list, gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    s = ext_prob.addVars(gbv.transit_list, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    z_m = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z_{ijt} for j,t
    z_p = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z_{ijt} for j,t
    v = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="v")  # v_{ijt} for j,t
    yUI = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUO_m = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O}_{ijt} for j,t
    yUO_p = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O}_{ijt} for j,t
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
                             i, j, t] + \
                         gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                     (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
                         for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                        name='input_item')

    ext_prob.addConstrs((yUO_p[i, j, t] - yUO_m[i, j, t] == gp.quicksum(
        s[l + (t,)] for l in gbv.transit_list if (l[0] == i) and (l[1] == j)) +
                         gp.quicksum(
                             gbv.bom_dict[j][bk] * xC.get((bk[0], j, t), 0.0) for bk in gbv.bom_key[j] if bk[1] == i) +
                         z_p[i, j, t] - z_m[i, j, t] +
                         gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                     (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                         for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list if
                         gbv.bom_key.get(j, -1) != -1),
                        name="output_item")
    ext_prob.addConstrs((yUO_p[i, j, t] - yUO_m[i, j, t] == gp.quicksum(
        s[l + (t,)] for l in gbv.transit_list if (l[0] == i) and (l[1] == j)) + z_p[i, j, t] - z_m[i, j, t] +
                         gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                     (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                         for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list if
                         gbv.bom_key.get(j, -1) == -1),
                        name="output_item")

    ext_prob.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xC[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter)) <= gbv.max_cap[ct, j][t]
                         for ct, j in gbv.max_cap.keys() for t in gbv.period_list), name='capacity')
    ext_prob.addConstrs(
        (rC[jta] <= v[i, jta[0], jta[1] - 1] for i in gbv.item_list for jta in gbv.alt_list if jta[2][0] == i),
        name='r_ub')
    ext_prob.addConstrs(
        (yUO_p[i, j, t] - yUO_m[i, j, t] <= v[i, j, t - 1] for i in gbv.item_list for j in gbv.plant_list for t in
         gbv.period_list),
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
        gp.quicksum(
            penalty_mag * (z_m[i, j, t] + yUO_m[i, j, t]) for i in gbv.item_list for j in gbv.plant_list for t in
            gbv.period_list)
        , GRB.MINIMIZE)

    ext_prob.update()
    ext_prob.optimize()

    opt_x = {}
    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            opt_x[i, j, t] = xC[i, j, t].X

    return ext_prob.ObjVal, opt_x


def master_prob():
    global gbv
    global sol_method, reg_method, level_obj_norm

    mp = gp.Model("master_prob")

    # mp.Params.TimeLimit = 10 * 60

    x = mp.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="x")
    w = mp.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")

    theta = mp.addVars(list(range(len(gbv.decomp_units))), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")

    mp.addConstrs(
        (x[i, j, t] == w[i, j, t] * gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='batch')
    mp.addConstrs(
        (w[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='w_ub')
    mp.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in gbv.prod_key) <= gbv.max_cap[ct, j][t]
                   for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()), name='capacity')

    if sol_method == "gb_callback":
        mp.setObjective(gp.quicksum(theta[i] for i in range(len(gbv.decomp_units))))
    elif sol_method == "master_sub_loop":
        if reg_method == "level_set" and level_obj_norm == 1:
            x_dev_p = mp.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x_dev_p")
            x_dev_m = mp.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x_dev_m")

    mp.update()

    return mp, [x, theta]


def level_set_aux(level_obj_norm):
    global gbv

    mp_aux = gp.Model("master_auxiliary_prob")

    mp_aux.setParam("DualReductions", 0)
    # mp_aux.Params.TimeLimit = 10 * 60

    x = mp_aux.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="x")
    w = mp_aux.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")

    theta = mp_aux.addVars(list(range(len(gbv.decomp_units))), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")

    mp_aux.addConstrs(
        (x[i, j, t] == w[i, j, t] * gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='batch')
    mp_aux.addConstrs(
        (w[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='w_ub')
    mp_aux.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in gbv.prod_key) <= gbv.max_cap[ct, j][t]
                       for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()), name='capacity')

    if level_obj_norm == 1:
        x_dev_p = mp_aux.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x_dev_p")
        x_dev_m = mp_aux.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x_dev_m")

    mp_aux.update()

    return mp_aux


def solve_level_set_aux(level, best_incum, cut_coeff, level_obj_norm):
    global gbv

    mp_aux = gp.Model("master_auxiliary_prob")

    mp_aux.setParam("DualReductions", 0)
    # mp_aux.Params.TimeLimit = 10 * 60

    x = mp_aux.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="x")
    w = mp_aux.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")

    theta = mp_aux.addVars(list(range(len(gbv.decomp_units))), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")

    mp_aux.addConstrs(
        (x[i, j, t] == w[i, j, t] * gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='batch')
    mp_aux.addConstrs(
        (w[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='w_ub')
    mp_aux.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in gbv.prod_key) <= gbv.max_cap[ct, j][t]
                       for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()), name='capacity')

    # level
    mp_aux.addConstr(gp.quicksum(theta[i] for i in range(len(gbv.decomp_units))) <= level)

    # bender's cut
    if len(cut_coeff.keys()) > 0:
        mp_aux.addConstrs(theta[k] >= cut_coeff[iter, k][0] + gp.quicksum(
            cut_coeff[iter, k][i, j, t][0] * (x[i, j, t] - cut_coeff[iter, k][i, j, t][1])
            for i, j in gbv.prod_key for t in gbv.period_list)
                          for iter, k in cut_coeff.keys())
    # objective
    if level_obj_norm == 2:
        obj = gp.quicksum((x[i, j, t] - best_incum[i, j, t]) ** 2 for i, j in gbv.prod_key for t in gbv.period_list)
    elif level_obj_norm == 1:
        x_dev_p = mp_aux.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x_dev_p")
        x_dev_m = mp_aux.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x_dev_m")

        mp_aux.addConstrs(x[i, j, t] + x_dev_p[i, j, t]
                          - x_dev_m[i, j, t] == best_incum[i, j, t] for i, j in gbv.prod_key for t in gbv.period_list)

        obj = gp.quicksum((x_dev_p[i, j, t] + x_dev_m[i, j, t]) for i, j in gbv.prod_key for t in gbv.period_list)

    mp_aux.setObjective(obj)

    mp_aux.update()
    mp_aux.optimize()

    x_vals = {(i, j, t): x[i, j, t].X for i, j in gbv.prod_key for t in gbv.period_list}
    theta_vals = {k: theta[k].X for k in range(len(gbv.decomp_units))}

    return x_vals, theta_vals


def sub_prob(dummy_unit_index, x_vals, penalty_mag=1e5):
    global gbv

    unit_ind_list = gbv.decomp_units[dummy_unit_index]
    print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand

    transit_list_i = [i_trans for i_trans in gbv.transit_list if i_trans[0] in unit_ind_list]
    si = sp.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_m = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z_m")  # z^+_{ijt} for j,t
    zi_p = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_m = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo_m")  # y^{O,+}_{ijt} for j,t
    yUOi_p = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    xCi = sp.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, 0] = 0.0  # initial unmet demand set to 0
    for l in transit_list_i:
        for t in range(min(gbv.period_list) - gbv.transit_time[l], min(gbv.period_list)):
            si[l + (t,)] = 0.0  # initial transportation set to 0
    for i in unit_ind_list:
        for j in gbv.plant_list:
            vi[i, j, 0] = gbv.init_inv.get((i, j), 0)  # initial inventory set to given values
    for i, j in gbv.prod_key:
        for t in range(min(gbv.period_list) - int(gbv.lead_time[i, j]), min(gbv.period_list)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    for i in gbv.item_list:
        for j in gbv.plant_list:
            if not ((i, j) in gbv.prod_key):
                for t in gbv.period_list:
                    xCi[i, j, t] = 0.0

    alt_i = [jta for jta in gbv.alt_list if (jta[2][0] in unit_ind_list) or (jta[2][1] in unit_ind_list)]
    rCi = sp.addVars(alt_i, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] - zi_m[i, j, t] for j in gbv.plant_list) \
                   == gbv.real_demand[i, t] for i in unit_ind_list for t in gbv.period_list), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi_p[i, j, t] - yUOi_m[i, j, t] == 0 for i in unit_ind_list \
         for j in gbv.plant_list for t in gbv.period_list), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[l],)] for l in transit_list_i if l[0] == i and l[2] == j) +
                   xCi[i, j, t - gbv.lead_time[i, j]] + gbv.external_purchase[i, j, t] +
                   gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                               (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i)) \
                   for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list),
                  name='input_item')

    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] == gp.quicksum(
        si[l + (t,)] for l in transit_list_i if l[0] == i and l[1] == j) +
                   gp.quicksum(
                       gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == i) + zi_p[i, j, t] -
                   zi_m[i, j, t] +
                   gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                               (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                   for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list if
                   gbv.bom_key.get(j, -1) != -1),
                  name="output_item")
    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] == gp.quicksum(
        si[l + (t,)] for l in transit_list_i if l[0] == i and l[1] == j) + zi_p[i, j, t] - zi_m[i, j, t] +
                   gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                               (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                   for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list if
                   gbv.bom_key.get(j, -1) == -1),
                  name="output_item")

    sp.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in gbv.prod_key)) <= gbv.max_cap[ct, j][t]
                   for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()), name='capacity')
    sp.addConstrs((rCi[jta] <= vi[jta[2][0], jta[0], jta[1] - 1] for jta in alt_i), name='r_ub')

    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in gbv.plant_list for t in gbv.period_list),
                  name='yo_ub')
    sp.addConstrs(
        (xCi[i, j, t] <= gbv.max_prod[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='x_ub')

    # fix the local variables to their global counterparts' values
    sub_fix_x = {}
    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] == x_vals[i, j, t])
    # set up the subproblem specific objective
    obj = gp.quicksum(
        gbv.holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list) + \
          gp.quicksum(gbv.penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in gbv.period_list) + \
          gp.quicksum(gbv.transit_cost[l] * si[l + (t,)] for l in transit_list_i for t in gbv.period_list) + \
          gp.quicksum(
              penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in gbv.plant_list for t in
              gbv.period_list)

    sp.setObjective(obj, GRB.MINIMIZE)

    sp.update()

    sp.optimize()

    # print("Subprob " + str(unit_ind_list) + " Status " + str(sp.Status))
    # if dummy_unit_index == 0 and sp.Status == 3:
    #    sp.computeIIS()
    #    sp.write("model1.ilp")

    dual_coeff = {}

    # obtain the objective value and return
    local_obj = sp.ObjVal

    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi

    return local_obj, dual_coeff


def compute_sub_dual(dummy_unit_index, x_vals, penalty_mag=1e5):
    global gbv

    unit_ind_list = gbv.decomp_units[dummy_unit_index]
    print(multiprocessing.current_process(), "\tSolving SubDualproblem " + str(unit_ind_list) + "!\n")

    sd = gp.Model("item_{}".format(unit_ind_list))
    sd.setParam("OutputFlag", 0)
    sd.setParam("DualReductions", 0)
    # sd.Params.TimeLimit = 5 * 60

    # set dual variables
    dual_u = sd.addVars(unit_ind_list, gbv.period_list, lb=-float('inf'), ub=float('inf'), name="dual_unmet_demand")
    dual_v = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=float('inf'),
                        name="dual_inventory")
    dual_I = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=float('inf'),
                        name="dual_input_item")
    dual_O = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=float('inf'),
                        name="dual_output_item")

    ct_j_t_list = [(ct, j, t) for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()]
    dual_C = sd.addVars(ct_j_t_list, lb=-float('inf'), ub=0.0, name="dual_capacity")

    alt_i = [jta for jta in gbv.alt_list if (jta[2][0] in unit_ind_list) or (jta[2][1] in unit_ind_list)]
    dual_rUB = sd.addVars(alt_i, lb=-float('inf'), ub=0.0, name="dual_r_ub")

    dual_OUB = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=0.0, name="dual_yo_ub")
    dual_xUB = sd.addVars(gbv.prod_key, gbv.period_list, lb=-float('inf'), ub=0.0, name="dual_x_ub")

    pi = sd.addVars(gbv.prod_key, gbv.period_list, lb=-float('inf'), ub=float('inf'), name="dual_x_hat")

    transit_list_i = [i_trans for i_trans in gbv.transit_list if i_trans[0] in unit_ind_list]

    # zero-padding
    for i, j in gbv.prod_key:
        if i in unit_ind_list:
            for t in range(max(gbv.period_list) + 1, max(gbv.period_list) + gbv.lead_time[i, j] + 1):
                dual_I[i, j, t] = 0.0
        else:
            for t in range(min(gbv.period_list), max(gbv.period_list) + gbv.lead_time[i, j] + 1):
                dual_I[i, j, t] = 0.0
                dual_O[i, j, t] = 0.0

    # dual constraints
    # coeff_v >= 0
    sd.addConstrs(gbv.holding_cost[i] - dual_v[i, j, t] + dual_v[i, j, t + 1] + dual_OUB[i, j, t + 1] +
                  gp.quicksum(dual_rUB[jta] for jta in alt_i if jta[0] == j and jta[1] == t + 1 and jta[2][0] == i) >= 0
                  for i in unit_ind_list for j in gbv.plant_list for t in
                  range(min(gbv.period_list), max(gbv.period_list)))
    sd.addConstrs(
        gbv.holding_cost[i] - dual_v[i, j, max(gbv.period_list)] >= 0 for i in unit_ind_list for j in gbv.plant_list)

    # coeff_u >= 0
    sd.addConstrs(gbv.penalty_cost[i] - dual_u[i, t] + dual_u[i, t + 1] >= 0 for i in unit_ind_list for t in
                  range(min(gbv.period_list), max(gbv.period_list)))
    sd.addConstrs(gbv.penalty_cost[i] - dual_u[i, max(gbv.period_list)] >= 0 for i in unit_ind_list)

    # coeff_s >= 0
    sd.addConstrs(gbv.transit_cost[l] + dual_I[l[0], l[2], t + gbv.transit_time[l]] + dual_O[l[0], l[1], t] >= 0
                  for l in transit_list_i for t in
                  range(min(gbv.period_list), max(gbv.period_list) - gbv.transit_time[l] + 1)
                  if gbv.transit_time[l] <= max(gbv.period_list) - min(gbv.period_list))
    sd.addConstrs(gbv.transit_cost[l] + dual_O[l[0], l[1], t] >= 0
                  for l in transit_list_i for t in
                  range(max(gbv.period_list) - gbv.transit_time[l] + 1, max(gbv.period_list) + 1))

    # coeff_z >= 0
    sd.addConstrs(
        dual_u[i, t] - dual_O[i, j, t] <= 0 for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list)
    sd.addConstrs(-dual_u[i, t] + dual_O[i, j, t] <= penalty_mag for i in unit_ind_list for j in gbv.plant_list for t in
                  gbv.period_list)

    # coeff_yUI >= 0
    sd.addConstrs(
        dual_v[i, j, t] - dual_I[i, j, t] >= 0 for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list)

    # coeff_yUO >= 0
    sd.addConstrs(
        dual_v[i, j, t] + dual_OUB[i, j, t] + dual_O[i, j, t] <= 0 for i in unit_ind_list for j in gbv.plant_list for t
        in gbv.period_list)
    sd.addConstrs(-dual_v[i, j, t] - dual_OUB[i, j, t] - dual_O[i, j, t] <= penalty_mag for i in unit_ind_list for j in
                  gbv.plant_list for t in gbv.period_list)

    # coeff_r >= 0
    sd.addConstrs(-dual_rUB[jta] + gbv.alt_dict[jta] * (
                dual_I[jta[2][1], jta[0], jta[1]] + dual_O[jta[2][0], jta[0], jta[1]]) >= 0 for jta in alt_i)

    # coeff_x >= 0
    sd.addConstrs(-dual_xUB[i, j, t] - pi[i, j, t] + dual_I[i, j, t + gbv.lead_time[i, j]]
                  - gp.quicksum(gbv.unit_cap.get((i, j, ct_j_t[0]), 0.0) * dual_C[ct_j_t] for ct_j_t in ct_j_t_list if
                                ct_j_t[1] == j and ct_j_t[2] == t)
                  + gp.quicksum(gbv.bom_dict[j].get((i, ii), 0.0) * dual_O[ii, j, t] for ii in unit_ind_list)
                  >= 0 for i, j in gbv.prod_key for t in gbv.period_list if j in gbv.bom_dict.keys())
    sd.addConstrs(-dual_xUB[i, j, t] - pi[i, j, t] + dual_I[i, j, t + gbv.lead_time[i, j]]
                  - gp.quicksum(gbv.unit_cap.get((i, j, ct_j_t[0]), 0.0) * dual_C[ct_j_t] for ct_j_t in ct_j_t_list if
                                ct_j_t[1] == j and ct_j_t[2] == t)
                  >= 0 for i, j in gbv.prod_key for t in gbv.period_list if not j in gbv.bom_dict.keys())

    obj = gp.quicksum(gbv.real_demand[i, t] * dual_u[i, t] for i in unit_ind_list for t in gbv.period_list) \
          + gp.quicksum(
        gbv.init_inv.get((i, j), 0.0) * (dual_v[i, j, 1] + dual_OUB[i, j, 1]) for i in unit_ind_list for j in
        gbv.plant_list) \
          + gp.quicksum(gbv.max_cap[ct, j][t] * dual_C[ct, j, t] for ct, j, t in ct_j_t_list) \
          + gp.quicksum(gbv.max_prod[i, j] * dual_xUB[i, j, t] for i, j in gbv.prod_key for t in gbv.period_list) \
          + gp.quicksum(x_vals[i, j, t] * pi[i, j, t] for i, j in gbv.prod_key for t in gbv.period_list) \
          + gp.quicksum(
        gbv.init_inv.get((jta[2][0], jta[0]), 0) * dual_rUB[jta] for jta in alt_i if jta[1] == min(gbv.period_list)) \
          + gp.quicksum(
        gbv.external_purchase[i, j, t] * dual_I[i, j, t] for i in unit_ind_list for j in gbv.plant_list for t in
        gbv.period_list)

    sd.setObjective(obj, GRB.MAXIMIZE)

    sd.update()

    sd.optimize()

    # print("SubDualprob " + str(unit_ind_list) + " : " + "Status " + str(sd.Status) + ", Objective " + str(sd.ObjVal))
    wr = "\nSubDualprob " + str(unit_ind_list) + " : " + "Status " + str(sd.Status) + ", Objective " + str(sd.ObjVal)
    print(wr)
    # wr_s = open("benders.txt", "a")
    # wr_s.write(wr)
    # wr_s.close()

    cut_coeff = {}
    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            if abs(pi[i, j, t].X) > 1e-5:
                cut_coeff[i, j, t] = pi[i, j, t].X
            else:
                cut_coeff[i, j, t] = 0

    local_obj = sd.ObjVal

    return local_obj, cut_coeff


def mycallback(model, where):
    # pool = Pool(min(len(gbv.decomp_units), 30))

    # initialization of best solution x
    if where == GRB.Callback.MIPSOL:  # MIPSOL()

        global max_iter, iter_num
        global dual_opt
        #global reg_method
        #global level_fractile, level_obj_norm
        #global cut_coeff
        #global best_cand_ub, best_incum

        # pool = Pool(min(len(gbv.decomp_units), 30))

        iter_num += 1

        best_ub = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        best_lb = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

        wr = "\n\nIteration: " + str(iter_num) + \
             "\nCallback best objective: " + str(best_ub) + "\nCallback best lower bound: " + str(best_lb)
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        x_vals = model.cbGetSolution(model._xVars)
        # x_vals = {(i, j, t): round(x_vals[i, j, t]) for i, j, t in x_vals.keys()}
        theta_vals = model.cbGetSolution(model._thetaVars)

        wr = "\nMaster x_val:"
        for i, j in gbv.prod_key:
            for t in gbv.period_list:
                wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
                # wr += str(int(x_vals[i, j, t])) + ","
        wr += "\nMaster theta_val:"
        for k in range(len(gbv.decomp_units)):
            wr += str(k) + " : " + str(theta_vals[k]) + "\t,"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        '''
        if reg_method == "level_set":
            # update level set auxiliary problem
            if best_lb >= 0 and best_ub < 1e10:
                level = best_lb + level_fractile * (best_ub - best_lb)
            else:
                level = math.inf

            wr = "\nCut coefficient num: " + str(len(cut_coeff.keys()))
            for ss in cut_coeff.keys():
                wr += "\t" + str(ss) + ","
            # wr_s = open("benders.txt", "a")
            # wr_s.write(wr)

            x_vals, theta_vals = solve_level_set_aux(level, best_incum, cut_coeff, level_obj_norm)

            if iter_num % 1 == 0:
                wr = "\nMaster level-set auxiliary x_val:"
                for i, j in gbv.prod_key:
                    for t in gbv.period_list:
                        wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
                wr += "\nMaster level-set auxiliary theta_val:"
                for k in range(len(gbv.decomp_units)):
                    wr += str(k) + " : " + str(theta_vals[k]) + ",\t"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)
                wr_s.close()
        '''
        # optimality cut
        if dual_opt == "gurobi_dual":
            x_vals_dict = {(i, j, t): x_vals[i, j, t] for i, j, t in x_vals.keys()}
            with Pool(min(len(gbv.decomp_units), 4)) as pool:
                sub_opt_results = pool.map(partial(sub_prob, x_vals=x_vals_dict), range(len(gbv.decomp_units)))

            #sub_opt_results = []
            #for k in range(len(gbv.decomp_units)):
            #    sub_opt_results.append(sub_prob(k, x_vals))
        elif dual_opt == "direct_solve":
            x_vals_dict = {(i, j, t): x_vals[i, j, t] for i, j, t in x_vals.keys()}
            with Pool(min(len(gbv.decomp_units), 4)) as pool:
                sub_opt_results = pool.map(partial(compute_sub_dual, x_vals=x_vals_dict), range(len(gbv.decomp_units)))

            #sub_opt_results = []
            #for k in range(len(gbv.decomp_units)):
            #    sub_opt_results.append(compute_sub_dual(k, x_vals))

        #ub_cand = 0
        for k in range(len(gbv.decomp_units)):
            sub_result = sub_opt_results[k]

            sub_obj = sub_result[0]
            sub_dual = sub_result[1]

            #ub_cand += sub_obj

            if theta_vals[k] < sub_obj:
                # optimality cut
                wr = "\nSub " + str(k) + ":\nDual variables:"
                for i, j, t in sub_dual.keys():
                    wr += str((i, j, t)) + ": " + str(sub_dual[i, j, t]) + ",\t"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)

                model.cbLazy(model._thetaVars[k] >= sub_obj + gp.quicksum(
                    sub_dual[i, j, t] * (model._xVars[i, j, t] - x_vals[i, j, t])
                    for i, j in gbv.prod_key for t in gbv.period_list))

                #cut_coeff[iter_num, k] = {(i, j, t): (sub_dual[i, j, t], x_vals[i, j, t]) for i, j in gbv.prod_key for t
                #                          in gbv.period_list}
                #cut_coeff[iter_num, k][0] = sub_obj

                wr = "\nOptimality cut:" + "\n\ttheta" + str([k]) + " >= " + str(sub_obj)
                for i, j in gbv.prod_key:
                    for t in gbv.period_list:
                        wr += " + (" + str(sub_dual[i, j, t]) + ")*" + "(x" + str([i, j, t]) + "-" + str(
                            x_vals[i, j, t]) + ")"
                # wr_s = open("benders.txt", "a")
                wr_s.write(wr)
                wr_s.close()

        #if ub_cand < best_cand_ub:
        #    best_cand_ub = ub_cand
        #    best_incum = copy.deepcopy(x_vals)

        # model.write("out_" + str(iter_num) + ".lp")
        if best_ub < math.inf and best_lb > -math.inf:
            gap = (best_ub - best_lb) / best_ub
        else:
            gap = math.inf

        if iter_num >= max_iter or gap < 0.05:
            model.terminate()


def master_sub_loop():
    global mp
    global max_iter, iter_num, best_lb, best_ub
    global dual_opt, reg_method
    global level_fractile, level_obj_norm
    global best_incum

    pool = Pool(min(len(gbv.decomp_units), 4))   # 30

    # level set initialization
    level = math.inf
    level_constr = mp.addConstr(
        gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(gbv.decomp_units))) <= level)
    mp.update()

    while True:
        iter_num += 1

        wr = "\nIteration: " + str(iter_num)
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        if reg_method == None:
            mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(gbv.decomp_units))))
            mp.update()
            mp.optimize()

            x_vals = {(i, j, t): mp.getVarByName(f"x[{i},{j},{t}]").X for i, j in gbv.prod_key for t in gbv.period_list}
            theta_vals = {k: mp.getVarByName(f"theta[{k}]").X for k in range(len(gbv.decomp_units))}

            lb_cand = sum(theta_vals[k] for k in range(len(gbv.decomp_units)))
            if lb_cand > best_lb:
                best_lb = lb_cand
        elif reg_method == "level_set":
            # master
            level_constr.rhs = math.inf
            mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(gbv.decomp_units))))
            mp.update()
            mp.optimize()

            theta_vals = {k: mp.getVarByName(f"theta[{k}]").X for k in range(len(gbv.decomp_units))}
            lb_cand = sum(theta_vals[k] for k in range(len(gbv.decomp_units)))
            if lb_cand > best_lb:
                best_lb = lb_cand

            # auxiliary problem, refinement to master
            # update level set auxiliary problem
            if best_ub < math.inf and best_lb > -math.inf:
                level = best_lb + level_fractile * (best_ub - best_lb)
                level_constr.rhs = level

            if level_obj_norm == 2:
                obj = gp.quicksum(
                    (mp.getVarByName(f"x[{i},{j},{t}]") - best_incum[i, j, t]) ** 2 for i, j in gbv.prod_key for t in
                    gbv.period_list)
            elif level_obj_norm == 1:
                if iter_num == 1:
                    x_dev_constr = mp.addConstrs(
                        mp.getVarByName(f"x[{i},{j},{t}]") + mp.getVarByName(f"x_dev_p[{i},{j},{t}]")
                        - mp.getVarByName(f"x_dev_m[{i},{j},{t}]") == best_incum[i, j, t] for i, j in gbv.prod_key for t
                        in gbv.period_list)
                elif iter_num > 1:
                    for i, j in gbv.prod_key:
                        for t in gbv.period_list:
                            x_dev_constr[i, j, t].rhs = best_incum[i, j, t]

                obj = gp.quicksum(
                    (mp.getVarByName(f"x_dev_p[{i},{j},{t}]") + mp.getVarByName(f"x_dev_m[{i},{j},{t}]")) for i, j in
                    gbv.prod_key for t in gbv.period_list)

            mp.setObjective(obj)
            mp.update()
            mp.optimize()

            x_vals = {(i, j, t): mp.getVarByName(f"x[{i},{j},{t}]").X for i, j in gbv.prod_key for t in gbv.period_list}
            theta_vals = {k: mp.getVarByName(f"theta[{k}]").X for k in range(len(gbv.decomp_units))}

            best_incum = copy.deepcopy(x_vals)
            '''
            if mp.Status == 3:
                wr = " infeasible\n"
                mp_aux.computeIIS()
                mp_aux.write("model1.ilp")

                f = open("model1.ilp")
                while True:
                    line = f.readline()
                    line = line[:-1]
                    if (line):
                        wr += line + "\n"
                    else:
                        break
                f.close()

                wr_s = open("benders.txt", "a")
                wr_s.write(wr)
            elif mp_aux.Status == 5:
                    wr = " unbounded"
                    print(wr)
            '''

        if iter_num % 1 == 0:
            wr = "\nMaster x_val:"
            for i, j in gbv.prod_key:
                for t in gbv.period_list:
                    wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
            wr += "\nMaster theta_val:"
            for k in range(len(gbv.decomp_units)):
                wr += str(k) + " : " + str(theta_vals[k]) + ",\t"
            wr_s = open("benders.txt", "a")
            wr_s.write(wr)
            wr_s.close()

        # subproblem solution given a production level

        # optimality cut
        if dual_opt == "gurobi_dual":
            sub_opt_results = pool.map(partial(sub_prob, x_vals=x_vals), range(len(gbv.decomp_units)))
        elif dual_opt == "direct_solve":
            sub_opt_results = pool.map(partial(compute_sub_dual, x_vals=x_vals), range(len(gbv.decomp_units)))

        # sub_opt_results = []
        # for k in range(len(gbv.decomp_units)):
        #    sub_opt_results.append(sub_prob(k, x_vals, "opt_cut"))

        wr = "\nPrimal objective: "
        for k in range(len(gbv.decomp_units)):
            wr += "\n\tSubPrimalprob " + str(gbv.decomp_units[k]) + " : Objective " + str(sub_opt_results[k][0])
        # wr_s = open("benders.txt", "a")
        # wr_s.write(wr)
        # wr_s.close()

        ub_cand = 0
        for k in range(len(gbv.decomp_units)):
            sub_result = sub_opt_results[k]

            sub_obj = sub_result[0]
            sub_dual = sub_result[1]

            ub_cand += sub_obj

            if theta_vals[k] < sub_obj:
                # optimality cut
                wr = "\nDual variables:"
                for i, j, t in sub_dual.keys():
                    wr += str((i, j, t)) + ": " + str(sub_dual[i, j, t]) + ",\t"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)

                mp.addConstr(mp.getVarByName(f"theta[{k}]") >= sub_obj + gp.quicksum(
                    sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                    for i, j in gbv.prod_key for t in gbv.period_list))

                wr = "\nOptimality cut:" + "\n\ttheta" + str([k]) + " >= " + str(sub_obj)
                for i, j in gbv.prod_key:
                    for t in gbv.period_list:
                        wr += " + (" + str(sub_dual[i, j, t]) + ")*" + "(x" + str([i, j, t]) + "-" + str(
                            x_vals[i, j, t]) + ")"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)

        if ub_cand < best_ub:
            best_ub = ub_cand

        mp.update()
        # mp.write("out_" + str(iter_num) + ".lp")

        if best_ub < math.inf and best_lb > -math.inf:
            gap = (best_ub - best_lb) / best_ub
        else:
            gap = math.inf

        wr = "\nCurrent Upper Bound: " + str(ub_cand) + "\nCurrent Lower Bound: " + str(lb_cand) + \
             "\nBest Upper Bound: " + str(best_ub) + "\nBest Lower Bound: " + str(best_lb) + "\n"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        if iter_num >= max_iter or gap < 0.05:
            break

if __name__ == "__main__":
    data_folder = "../data/small_test_real"  # pilot_test/fine_tune
    # create global information
    gbv = create_gbv(data_folder)


    '''
    opt_obj, opt_x_val = extensive_prob()
    wr = "\nOptimal objective: " + str(opt_obj) + "\nOptimal integer solution x:\n\t"
    for i, j, t in opt_x_val.keys():
        wr += str((i, j, t)) + " : " + str(opt_x_val[i, j, t]) + ",\t"
    wr_s = open("benders.txt", "a")
    wr_s.write(wr)
    wr_s.close()
    '''

    max_iter = 1000
    iter_num = 0
    best_lb = -math.inf
    best_ub = math.inf

    ##### regularization method: level_set / regularized BD
    reg_method = "level_set"
    #reg_method = None
    level_fractile = 0.5
    level_obj_norm = 2

    best_cand_ub = math.inf
    # initialization of best solution x
    best_incum = {(i, j, t): 0 for i, j in gbv.prod_key for t in gbv.period_list}

    # bender's cut coefficient
    #cut_coeff = {}

    ##### solution method: callback / for loop
    sol_method = "gb_callback"
    #sol_method = "master_sub_loop"

    ##### access to dual variables: gurobio dual / solution to dual problem
    # dual_opt = "gurobi_dual"
    dual_opt = "direct_solve"

    mp, mp_handle = master_prob()

    if sol_method == "gb_callback":
        mp._xVars = mp_handle[0]
        mp._thetaVars = mp_handle[1]

        mp.Params.LazyConstraints = 1
        mp.Params.Threads = 1
        mp.Params.OutputFlag = 1
        # mp.Params.TimeLimit = 100

        start = time.time()
        mp.optimize(mycallback)
        end = time.time()

        wr = "\nEnd in " + str(end - start) + " seconds!"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

    elif sol_method == "master_sub_loop":
        start = time.time()
        master_sub_loop()
        end = time.time()

        wr = "\nEnd in " + str(end - start) + " seconds!"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

