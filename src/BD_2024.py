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

from params import ADMMparams
from readin import *
from GlobalVariable import *
from variables import global_var

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
        if not(j in gbv.bom_key.keys()):
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

def extensive_prob(relax_option=False):
    # set up the extensive formulation
    # set up the extensive formulation
    global gbv
    ext_prob = gp.Model("extensive_form")
    ext_prob.setParam("Threads", 1)

    # set up model parameters (M: plant, T: time, L: transit,
    u = ext_prob.addVars(gbv.item_list, gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    s = ext_prob.addVars(gbv.transit_list, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    z = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="z")  # z_{ijt} for j,t
    v = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="v")  # v_{ijt} for j,t
    yUI = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUO = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yo")  # y^{O}_{ijt} for j,t
    xC = ext_prob.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t

    # initial condition setup
    for i in gbv.item_list:
        u[i, 0] = 0.0  # initial unmet demand set to 0
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[l], min(gbv.period_list)):
            s[l + (t,)] = 0.0    # initial transportation set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if (i, j) in gbv.init_inv.keys():
                v[i, j, 0] = gbv.init_inv[i, j]     # initial inventory set to given values
            else:
                v[i, j, 0] = 0.0
    for i, j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i, j], min(gbv.period_list)):
            xC[i, j, t] = 0.0   # initial production set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if not ((i, j) in gbv.prod_key):
                for t in gbv.period_list:
                    xC[i, j, t] = 0.0   # production for non-existent item-plant pair set to 0

    rC = ext_prob.addVars(gbv.alt_list, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # add constraints for the extensive formulation
    ext_prob.addConstrs((u[i, t] - u[i, t - 1] + gp.quicksum(z[i, j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[i, t] for i in gbv.item_list for t in gbv.period_list), name='unmet_demand')
    ext_prob.addConstrs((v[i, j, t] - v[i, j, t - 1] - yUI[i, j, t] + yUO[i, j, t] == 0 \
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list), name='inventory')
    
    ext_prob.addConstrs((yUI[i, j, t] == gp.quicksum(s[l + (t - gbv.transit_time[l],)] for l in gbv.transit_list if (l[0] == i) and (l[2] == j)) +
                     (xC[i, j, t - gbv.lead_time[i, j]] if (i,j) in gbv.prod_key else 0.0) + gbv.external_purchase[i, j, t] + \
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                    name='input_item')
    
    ext_prob.addConstrs((yUO[i, j, t] == gp.quicksum(s[l + (t,)] for l in gbv.transit_list if (l[0] == i) and (l[1] == j)) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xC.get((bk[0], j, t),0.0) for bk in gbv.bom_key[j] if bk[1] == i) + z[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    ext_prob.addConstrs((yUO[i, j, t] == gp.quicksum(s[l + (t,)] for l in gbv.transit_list if (l[0] == i) and (l[1] == j)) + z[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")
    
    ext_prob.addConstrs((gp.quicksum(gbv.unit_cap[i_iter, j_iter, ct_iter] * xC[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
                                 if (j_iter == j) and (ct == ct_iter)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.period_list), name='capacity')
    ext_prob.addConstrs((rC[jta] <= v[i, jta[0], jta[1] - 1] for i in gbv.item_list for jta in gbv.alt_list if jta[2][0] == i),
                    name='r_ub')
    ext_prob.addConstrs((yUO[i, j, t] <= v[i, j, t - 1] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')

    if not(relax_option):
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
        gp.quicksum(gbv.holding_cost[i] * v[i, j, t] for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list) +
        gp.quicksum(gbv.penalty_cost[i] * u[i, t] for i in gbv.item_list for t in gbv.period_list) +
        gp.quicksum(gbv.transit_cost[l] * s[l + (t,)] for l in gbv.transit_list for t in gbv.period_list)
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

    mp = gp.Model("master_prob")

    x = mp.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="x")
    w = mp.addVars(gbv.prod_key, gbv.period_list, vtype=GRB.INTEGER, lb=0.0, name="w")

    theta = mp.addVars(list(range(len(gbv.decomp_units))), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")

    mp.addConstrs(
        (x[i, j, t] == w[i, j, t] * gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='batch')
    mp.addConstrs(
        (w[i, j, t] <= gbv.max_prod[i, j] / gbv.lot_size[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='w_ub')
    mp.addConstrs((gp.quicksum(gbv.unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
                                 if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in gbv.prod_key) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()), name='capacity')
    
    mp.setObjective(gp.quicksum(theta[i] for i in range(len(gbv.decomp_units))))
    
    return mp, [x, theta]

def sub_prob(dummy_unit_index, x_vals, cut_opt):
    global gbv

    unit_ind_list = gbv.decomp_units[dummy_unit_index]
    print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, gbv.period_list, lb = 0.0, name="u")  # u_{it} for t, unmet demand

    transit_list_i = [i_trans for i_trans in gbv.transit_list if i_trans[0] in unit_ind_list]
    si = sp.addVars(transit_list_i, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    
    zi = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb = 0.0, name="z")  # z^+_{ijt} for j,t
    vi = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb = 0.0, name = "v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb = 0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = sp.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb = 0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    xCi = sp.addVars(gbv.prod_key, gbv.period_list, lb = 0.0, name="x")  # x_{ijt} for i,j,t (x copy)

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
    rCi = sp.addVars(alt_i, lb = 0.0, name="r")  # r_{ajt} for a=(i,i')

    if cut_opt == "feas_cut":
        err_p = sp.addVars(gbv.prod_key, gbv.period_list, lb = 0.0, name="error_p")  
        err_m = sp.addVars(gbv.prod_key, gbv.period_list, lb = 0.0, name="error_m")  

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi[i, j, t] for j in gbv.plant_list) \
                     == gbv.real_demand[i, t] for i in unit_ind_list for t in gbv.period_list), name='unmet_demand')
    sp.addConstrs((vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi[i, j, t] == 0 for i in unit_ind_list \
                     for j in gbv.plant_list for t in gbv.period_list), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[l + (t - gbv.transit_time[l],)] for l in transit_list_i if l[0] == i and l[2] == j) +
                        xCi[i, j, t - gbv.lead_time[i, j]] + gbv.external_purchase[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))\
                        for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list),
                    name='input_item')

    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == i and l[1] == j) +
                     gp.quicksum(
                         gbv.bom_dict[j][bk] * xCi[bk[0], j, t] for bk in gbv.bom_key[j] if bk[1] == i) + zi[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                     for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) != -1),
                    name="output_item")
    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(si[l + (t,)] for l in transit_list_i if l[0] == i and l[1] == j) + zi[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rCi[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i))
                     for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list if gbv.bom_key.get(j, -1) == -1),
                    name="output_item")

    sp.addConstrs((gp.quicksum(
        gbv.unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in
        gbv.unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in gbv.prod_key)) <= gbv.max_cap[ct, j][t]
                     for ct, j in gbv.max_cap.keys() for t in gbv.max_cap[ct, j].keys()), name='capacity')
    sp.addConstrs((rCi[jta] <= vi[jta[2][0], jta[0], jta[1] - 1] for jta in alt_i), name='r_ub')

    sp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list 
                     for j in gbv.plant_list for t in gbv.period_list),
                    name='yo_ub')
    sp.addConstrs(
        (xCi[i, j, t] <= gbv.max_prod[i, j] for i, j in gbv.prod_key for t in gbv.period_list),
        name='x_ub')
    
    # fix the local variables to their global counterparts' values
    sub_fix_x = {}
    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            if cut_opt == "opt_cut":
                sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] == x_vals[i, j, t])
            elif cut_opt == "feas_cut":
                sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] + err_p[i, j, t] - err_m[i, j, t] == x_vals[i, j, t])

    # set up the subproblem specific objective
    if cut_opt == "opt_cut":
        obj = gp.quicksum(gbv.holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list) + \
            gp.quicksum(gbv.penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in gbv.period_list) + \
                        gp.quicksum(gbv.transit_cost[l] * si[l + (t,)] for l in transit_list_i for t in gbv.period_list)
    elif cut_opt == "feas_cut":
        obj = gp.quicksum(err_p[i, j, t] + err_m[i, j, t] for i, j in gbv.prod_key for t in gbv.period_list)

    sp.setObjective(obj, GRB.MINIMIZE)

    sp.update()

    sp.optimize()
    
    dual_coeff = {}

    # obtain the objective value and return
    local_obj = sp.ObjVal

    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi
    
    return local_obj, dual_coeff

def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:   #MIPSOL
        global max_iter, iter_num, best_lb, best_ub

        #pool = Pool(min(len(gbv.decomp_units), 30))

        iter_num += 1

        CB_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        CB_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

        wr = "\nIteration: " + str(iter_num) + "\nCallback objective: " + str(CB_obj) + "\nCallback bound: " + str(CB_bound)
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        x_vals = model.cbGetSolution(model._xVars)
        theta_vals = model.cbGetSolution(model._thetaVars)

        wr = "\nMaster x_val:"
        for i, j in gbv.prod_key: 
            for t in gbv.period_list:
                wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
        wr += "\nMaster theta_val:"
        for k in range(len(gbv.decomp_units)):
            wr += str(k) + " : " + str(theta_vals[k]) + "\t,"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        # lower bound
        lb_cand = sum(theta_vals[k] for k in range(len(gbv.decomp_units)))
        if lb_cand > best_lb:
            best_lb = lb_cand

        # subproblem solution given a production level
        # feasibility cut
        #sub_feas_results = pool.map(partial(sub_prob, x_vals=x_vals, cut_opt="feas_cut"), range(len(gbv.decomp_units)))
        sub_feas_results = []
        for k in range(len(gbv.decomp_units)):
            sub_feas_results.append(sub_prob(k, x_vals, "feas_cut"))

        feas_flag = True
        for k in range(len(gbv.decomp_units)):
            sub_result = sub_feas_results[k]

            sub_obj = sub_result[0]
            sub_dual = sub_result[1]
            
            if sub_obj > 0:
                feas_flag = False
                # feasibility cut
                model.cbLazy(0 >= sub_obj + gp.quicksum(sub_dual[i, j, t] * (model._xVars[i, j, t]  - x_vals[i, j, t])
                                                  for i, j in gbv.prod_key for t in gbv.period_list))
                
                wr = "\nFeasiblity cut:" + "\n\t0 >= " + str(sub_obj)
                for i, j in gbv.prod_key:
                    for t in gbv.period_list:
                        wr += " + (" + str(sub_dual[i, j, t]) + ")*" + "(x" + str([i, j, t]) + "-" + str(x_vals[i, j, t]) + ")"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)
        
        # optimality cut
        if feas_flag:
            # sub_opt_results = pool.map(partial(sub_prob, x_vals=x_vals, cut_opt="opt_cut"), range(len(gbv.decomp_units)))
            sub_opt_results = []
            for k in range(len(gbv.decomp_units)):
                sub_opt_results.append(sub_prob(k, x_vals, "opt_cut"))
 
            ub_cand = 0
            for k in range(len(gbv.decomp_units)):
                sub_result = sub_opt_results[k]

                sub_obj = sub_result[0]
                sub_dual = sub_result[1]

                ub_cand += sub_obj

                if theta_vals[k] < sub_obj:
                    # optimality cut
                    model.cbLazy(model._thetaVars[k] >= sub_obj + gp.quicksum(sub_dual[i, j, t] * (model._xVars[i, j, t] - x_vals[i, j, t])
                                                  for i, j in gbv.prod_key for t in gbv.period_list))
                    wr = "\nOptimality cut:" + "\n\ttheta" + str([k]) + " >= " + str(sub_obj)
                    for i, j in gbv.prod_key:
                        for t in gbv.period_list:
                            wr += " + (" + str(sub_dual[i, j, t]) + ")*" + "(x" + str([i, j, t]) + "-" + str(x_vals[i, j, t]) + ")"
                    wr_s = open("benders.txt", "a")
                    wr_s.write(wr)
            # upper bound
            if ub_cand < best_ub:
                best_ub = ub_cand
        
        #model.write("out_" + str(iter_num) + ".lp")
        
        if best_ub < math.inf:
            gap = (best_ub - best_lb) / best_ub
        else:
            gap = math.inf
        
        wr = "\nBest Upper Bound: " + str(best_ub) + "\nBest Lower Bound: " + str(best_lb) + "\n"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        if iter_num >= max_iter or gap < 0.05:
            model.terminate()

def compute_sub_dual(dummy_unit_index, x_vals, cut_opt):
    global gbv

    unit_ind_list = gbv.decomp_units[dummy_unit_index]
    print(multiprocessing.current_process(), "\tSolving SubDualproblem " + str(unit_ind_list) + "!\n")

    sd = gp.Model("item_{}".format(unit_ind_list))
    sd.setParam("OutputFlag", 0)
    sd.setParam("DualReductions", 0)

    # set dual variables
    dual_u = sd.addVars(unit_ind_list, gbv.period_list, lb=-float('inf'), ub=float('inf'), name="dual_unmet_demand")
    dual_v = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=float('inf'), name="dual_inventory")
    dual_I = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=float('inf'), name="dual_input_item")
    dual_O = sd.addVars(unit_ind_list, gbv.plant_list, gbv.period_list, lb=-float('inf'), ub=float('inf'), name="dual_output_item")
    
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
    if cut_opt == "opt_cut":
        sd.addConstrs(gbv.holding_cost[i] - dual_v[i, j, t] + dual_v[i, j, t + 1] + dual_OUB[i, j, t + 1] + 
                    gp.quicksum(dual_rUB[jta] for jta in alt_i if jta[0] == j and jta[1] == t + 1 and jta[2][0] == i) >= 0
                    for i in unit_ind_list for j in gbv.plant_list for t in range(min(gbv.period_list), max(gbv.period_list)))
        sd.addConstrs(gbv.holding_cost[i] - dual_v[i, j, max(gbv.period_list)] >= 0 for i in unit_ind_list for j in gbv.plant_list)

        sd.addConstrs(gbv.penalty_cost[i] - dual_u[i, t] + dual_u[i, t + 1] >= 0 for i in unit_ind_list for t in range(min(gbv.period_list), max(gbv.period_list)))
        sd.addConstrs(gbv.penalty_cost[i] - dual_u[i, max(gbv.period_list)] >= 0 for i in unit_ind_list)

        sd.addConstrs(gbv.transit_cost[l] + dual_I[l[0], l[2], t + gbv.transit_time[l]] + dual_O[l[0], l[1], t + gbv.transit_time[l]] >= 0 
                    for l in transit_list_i for t in range(min(gbv.period_list), max(gbv.period_list) - gbv.transit_time[l] + 1) 
                    if gbv.transit_time[l] <= max(gbv.period_list) - min(gbv.period_list))
    elif cut_opt == "feas_cut":
        sd.addConstrs(-dual_v[i, j, t] + dual_v[i, j, t + 1] + dual_OUB[i, j, t + 1] + 
                    gp.quicksum(dual_rUB[jta] for jta in alt_i if jta[0] == j and jta[1] == t + 1 and jta[2][0] == i) >= 0
                    for i in unit_ind_list for j in gbv.plant_list for t in range(min(gbv.period_list), max(gbv.period_list)))
        sd.addConstrs(-dual_v[i, j, max(gbv.period_list)] >= 0 for i in unit_ind_list for j in gbv.plant_list)

        sd.addConstrs(-dual_u[i, t] + dual_u[i, t + 1] >= 0 for i in unit_ind_list for t in range(min(gbv.period_list), max(gbv.period_list)))
        sd.addConstrs(-dual_u[i, max(gbv.period_list)] >= 0 for i in unit_ind_list)

        sd.addConstrs(dual_I[l[0], l[2], t + gbv.transit_time[l]] + dual_O[l[0], l[1], t + gbv.transit_time[l]] >= 0 
                    for l in transit_list_i for t in range(min(gbv.period_list), max(gbv.period_list) - gbv.transit_time[l] + 1) 
                    if gbv.transit_time[l] <= max(gbv.period_list) - min(gbv.period_list))
        
    sd.addConstrs(dual_u[i, t] - dual_O[i, j, t] <= 0 for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list)

    sd.addConstrs(dual_v[i, j, t] - dual_I[i, j, t] >= 0 for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list)

    sd.addConstrs(dual_v[i, j, t] + dual_OUB[i, j, t] + dual_O[i, j, t] <= 0 for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list)

    sd.addConstrs(-dual_rUB[jta] + gbv.alt_dict[jta] * (dual_I[jta[2][1], jta[0], jta[1]] + dual_O[jta[2][0], jta[0], jta[1]]) >= 0 for jta in alt_i)
    
    sd.addConstrs(-dual_xUB[i, j, t] - pi[i, j, t] + dual_I[i, j, t + gbv.lead_time[i, j]]
                  - gp.quicksum(gbv.unit_cap.get((i, j, ct_j_t[0]), 0.0) * dual_C[ct_j_t] for ct_j_t in ct_j_t_list if ct_j_t[1] == j and ct_j_t[2] == t)
                  + gp.quicksum(gbv.bom_dict[j].get((i, ii), 0.0) * dual_O[ii, j, t] for ii in unit_ind_list) 
                  >= 0 for i, j in gbv.prod_key for t in gbv.period_list if j in gbv.bom_dict.keys())
    sd.addConstrs(-dual_xUB[i, j, t] - pi[i, j, t] + dual_I[i, j, t + gbv.lead_time[i, j]]
                  - gp.quicksum(gbv.unit_cap.get((i, j, ct_j_t[0]), 0.0) * dual_C[ct_j_t] for ct_j_t in ct_j_t_list if ct_j_t[1] == j and ct_j_t[2] == t) 
                  >= 0 for i, j in gbv.prod_key for t in gbv.period_list if not j in gbv.bom_dict.keys())
    
    if cut_opt == "feas_cut":
        sd.addConstrs(pi[i, j, t] <= 1 for i, j in gbv.prod_key for t in gbv.period_list)
        sd.addConstrs(pi[i, j, t] >= -1 for i, j in gbv.prod_key for t in gbv.period_list)
    
    obj = gp.quicksum(gbv.real_demand[i, t] * dual_u[i, t] for i in unit_ind_list for t in gbv.period_list) \
            + gp.quicksum(gbv.init_inv.get((i, j), 0.0) * (dual_v[i, j, 1] + dual_OUB[i, j, 1]) for i in unit_ind_list for j in gbv.plant_list) \
            + gp.quicksum(gbv.max_cap[ct, j][t] * dual_C[ct, j, t] for ct, j, t in ct_j_t_list) \
            + gp.quicksum(gbv.max_prod[i, j] * dual_xUB[i, j, t] for i, j in gbv.prod_key for t in gbv.period_list) \
            + gp.quicksum(x_vals[i, j, t] * pi[i, j, t] for i, j in gbv.prod_key for t in gbv.period_list) \
            + gp.quicksum(gbv.init_inv.get((jta[2][0], jta[0]), 0) * dual_rUB[jta] for jta in alt_i if jta[1] == min(gbv.period_list)) \
            + gp.quicksum(gbv.external_purchase[i, j, t] * dual_I[i, j, t] for i in unit_ind_list for j in gbv.plant_list for t in gbv.period_list)
    
    sd.setObjective(obj, GRB.MAXIMIZE)

    sd.update()

    sd.optimize()
    
    #print("SubDualprob " + str(unit_ind_list) + " : " + "Status " + str(sd.Status) + ", Objective " + str(sd.ObjVal))

    cut_coeff = {}
    for i, j in gbv.prod_key:
        for t in gbv.period_list:
            if abs(pi[i, j, t].X) > 1e-5:
                cut_coeff[i, j, t] = pi[i, j, t].X
            else:
                cut_coeff[i, j, t] = 0
    
    return cut_coeff

def master_sub_loop(dual_opt):
    global mp
    global max_iter, iter_num, best_lb, best_ub

    pool = Pool(min(len(gbv.decomp_units), 30))
    
    while True:
        iter_num += 1

        wr = "\nIteration: " + str(iter_num)
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()
        
        mp.optimize()
        
        if iter_num % 1 == 0:
            x_vals = {(i, j, t): mp.getVarByName(f"x[{i},{j},{t}]").X for i, j in gbv.prod_key for t in gbv.period_list}
            theta_vals = {k: mp.getVarByName(f"theta[{k}]").X for k in range(len(gbv.decomp_units))}
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

        # lower bound
        lb_cand = sum(theta_vals[k] for k in range(len(gbv.decomp_units)))
        if lb_cand > best_lb:
            best_lb = lb_cand

        # subproblem solution given a production level
        # feasibility cut
        sub_feas_results = pool.map(partial(sub_prob, x_vals=x_vals, cut_opt="feas_cut"), range(len(gbv.decomp_units)))
        #sub_feas_results = []
        #for k in range(len(gbv.decomp_units)):
        #    sub_feas_results.append(sub_prob(k, x_vals, "feas_cut"))

        feas_flag = True
        for k in range(len(gbv.decomp_units)):
            sub_result = sub_feas_results[k]

            sub_obj = sub_result[0]
            
            if sub_obj > 0:
                feas_flag = False
                # feasibility cut
                if dual_opt == "gurobi_dual":
                    sub_dual = sub_result[1]
                elif dual_opt == "direct_solve":
                    sub_dual = compute_sub_dual(k, x_vals, "feas_cut")
                
                wr = "\nDual variables:"
                for i, j, t in sub_dual.keys():
                    wr += str((i, j, t)) + ": " + str(sub_dual[i, j, t]) + ",\t"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)

                mp.addConstr(0 >= sub_obj + gp.quicksum(sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]")  - x_vals[i, j, t])
                                                    for i, j in gbv.prod_key for t in gbv.period_list))
                
                wr = "\nFeasiblity cut:" + "\n\t0 >= " + str(sub_obj)
                for i, j in gbv.prod_key:
                    for t in gbv.period_list:
                        wr += " + (" + str(sub_dual[i, j, t]) + ")*" + "(x" + str([i, j, t]) + "-" + str(x_vals[i, j, t]) + ")"
                wr_s = open("benders.txt", "a")
                wr_s.write(wr)
        
        # optimality cut
        if feas_flag:
            sub_opt_results = pool.map(partial(sub_prob, x_vals=x_vals, cut_opt="opt_cut"), range(len(gbv.decomp_units)))
            #sub_opt_results = []
            #for k in range(len(gbv.decomp_units)):
            #    sub_opt_results.append(sub_prob(k, x_vals, "opt_cut"))
            
            #sub_obj_list = []
            #for k in range(len(gbv.decomp_units)):
            #    sub_obj_list.append(sub_opt_results[k][0])
            #print("Primal objective: " + str(sub_obj_list))

            ub_cand = 0
            for k in range(len(gbv.decomp_units)):
                sub_result = sub_opt_results[k]

                sub_obj = sub_result[0]
                #sub_dual = sub_result[1]

                ub_cand += sub_obj

                if theta_vals[k] < sub_obj:
                    # optimality cut
                    if dual_opt == "gurobi_dual":
                        sub_dual = sub_result[1]
                    elif dual_opt == "direct_solve":
                        sub_dual = compute_sub_dual(k, x_vals, "opt_cut")
                    
                    wr = "\nDual variables:"
                    for i, j, t in sub_dual.keys():
                        wr += str((i, j, t)) + ": " + str(sub_dual[i, j, t]) + ",\t"
                    wr_s = open("benders.txt", "a")
                    wr_s.write(wr)

                    mp.addConstr(mp.getVarByName(f"theta[{k}]") >= sub_obj + gp.quicksum(sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                                    for i, j in gbv.prod_key for t in gbv.period_list))
                    wr = "\nOptimality cut:" + "\n\ttheta" + str([k]) + " >= " + str(sub_obj)
                    for i, j in gbv.prod_key:
                        for t in gbv.period_list:
                            wr += " + (" + str(sub_dual[i, j, t]) + ")*" + "(x" + str([i, j, t]) + "-" + str(x_vals[i, j, t]) + ")"
                    wr_s = open("benders.txt", "a")
                    wr_s.write(wr)
            # upper bound
            if ub_cand < best_ub:
                best_ub = ub_cand
        
        mp.update()
        #mp.write("out_" + str(iter_num) + ".lp")
        
        if best_ub < math.inf:
            gap = (best_ub - best_lb) / best_ub
        else:
            gap = math.inf
        
        wr = "\nBest Upper Bound: " + str(best_ub) + "\nBest Lower Bound: " + str(best_lb) + "\n"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        if iter_num >= max_iter or gap < 0.05:
            break

if __name__ == "__main__":
    data_folder = "../data/fine_tune"    # fine_tune
    # create global information
    gbv = create_gbv(data_folder)

    opt_obj, opt_x_val = extensive_prob()
    wr = "\nOptimal objective: " + str(opt_obj) + "\nOptimal integer solution x:\n\t"
    for i, j, t in opt_x_val.keys():
        wr += str((i, j, t)) + " : " + str(opt_x_val[i, j, t]) + ",\t"
    wr_s = open("benders.txt", "a")
    wr_s.write(wr)
    wr_s.close()

    exit()


    max_iter = 100
    iter_num = 0
    best_lb = 0
    best_ub = math.inf

    mp, mp_handle = master_prob()
    
    #sol_method = "gb_callback"
    sol_method = "master_sub_loop"
    
    if sol_method == "gb_callback":
        mp._xVars = mp_handle[0]
        mp._thetaVars = mp_handle[1]

        mp.Params.LazyConstraints = 1
        mp.Params.Threads = 1 
        mp.Params.OutputFlag = 1
        # mp.Params.TimeLimit = 100
        
        mp.optimize(mycallback)
    elif sol_method == "master_sub_loop":
        master_sub_loop(dual_opt="direct_solve")    #dual_opt = "gurobi_dual" or "direct_solve"
    
