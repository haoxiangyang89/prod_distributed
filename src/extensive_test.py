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

from params import ADMMparams
from readin import *
from GlobalVariable import *
from variables import global_var

from random import random
from time import sleep
import collections

def extensive_prob(solve_option = True, relax_option = False):
    # set up the extensive formulation
    # set up the extensive formulation
    global gbv
    ext_prob = gp.Model("extensive_form")
    ext_prob.setParam("Threads", 1)

    # set up model parameters (M: plant, T: time, L: transit,
    u = ext_prob.addVars(gbv.item_list, gbv.period_list, lb=0.0, name="u")  # u_{it} for t, unmet demand
    s = ext_prob.addVars(gbv.transit_list, gbv.period_list, lb=0.0, name="s")  # s_{ilt} for l,t
    z = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, name="z")  # z_{ijt} for j,t
    v = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="v")  # v_{ijt} for j,t
    yUI = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUO = ext_prob.addVars(gbv.item_list, gbv.plant_list, gbv.period_list, name="yo")  # y^{O}_{ijt} for j,t
    xC = ext_prob.addVars(gbv.prod_key, gbv.period_list, lb=0.0, name="x")  # x_{ijt} for i,j,t

    # initial condition setup
    for i in gbv.item_list:
        u[i, 0] = 0.0  # initial unmet demand set to 0
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[l], min(gbv.period_list)):
            s[l + (t,)] = 0.0    # initial transportation set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if (i,j) in gbv.init_inv.keys():
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
                     (xC[i, j, t - gbv.lead_time[i, j]] if (i,j) in gbv.prod_key else 0.0) + gbv.external_purchase[i, j, t] \
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
                    name='input_item')
    ext_prob.addConstrs((yUO[i, j, t] == gp.quicksum(s[l + (t,)] for l in gbv.transit_list if (l[0] == i) and (l[1] == j)) +
                     gp.quicksum(gbv.bom_dict[j][bk] * xC.get((bk[0], j, t),0.0) for bk in gbv.bom_key[j] if bk[1] == i) + z[i, j, t] +
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][0] == i)) -
                     gp.quicksum(gbv.alt_dict[jta] * rC[jta] for jta in gbv.alt_list if
                                 (jta[0] == j) and (jta[1] == t) and (jta[2][1] == i))
                     for i in gbv.item_list for j in gbv.plant_list for t in gbv.period_list),
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
        gp.quicksum(gbv.transit_cost[l] * s[l + (t,)] for l in gbv.transit_list for t in gbv.period_list), GRB.MINIMIZE)

    ext_prob.update()

    if solve_option:
        ext_prob.optimize()
        # collect the solution and objective value
        return ext_prob.objVal
    else:
        return ext_prob, [u, s, z, v, yUI, yUO, xC, rC]


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
    gbv.cap_period_list, gbv.cap_key, gbv.max_cap, gbv.cap_list = input_capacity(os.path.join(path_file, "dm_df_max_capacity.csv"))
    gbv.transit_list, gbv.transit_time, gbv.transit_cost = input_transit(os.path.join(path_file, "dm_df_transit.csv"))
    gbv.L = len(gbv.transit_list)
    gbv.init_inv = input_init_inv(os.path.join(path_file, "dm_df_inv.csv"))

    gbv.period_list = input_periods(os.path.join(path_file, "dm_df_periods.csv"))
    gbv.T = len(gbv.period_list)
    gbv.external_purchase = input_po(os.path.join(path_file, "dm_df_po.csv"))
    gbv.real_demand, gbv.forecast_demand = input_demand(os.path.join(path_file, "dm_df_demand.csv"))

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
                if not((i,j,t) in gbv.external_purchase.keys()):
                    gbv.external_purchase[i,j,t] = 0.0
        for t in gbv.period_list:
            if not((i,t) in gbv.real_demand.keys()):
                gbv.real_demand[i,t] = 0.0

    return gbv

if __name__ == '__main__':
    data_folder = "./data/small_test_fine_tune"
    gbv = create_gbv(data_folder)
    ext_prob, varList = extensive_prob(False,True)
    ext_prob.optimize()
