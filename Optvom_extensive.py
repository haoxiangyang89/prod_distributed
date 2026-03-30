import optvom as op
import os
import sys
from argparse import ArgumentParser
import numpy as np
from numpy import linalg as LA
import time
import math
import copy
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

        unit_cap_i_j = []
        for ct, j in gbv.max_cap.keys():
            tem_unit_cap_i_j = []
            flag = False
            for i_iter, j_iter, ct_iter in gbv.unit_cap.keys():
                if (j_iter == j) and (ct == ct_iter):
                    tem_unit_cap_i_j.append((i_iter, j_iter))
                    if i_iter == i:
                        flag = True
            if flag:
                unit_cap_i_j.extend(tem_unit_cap_i_j)
        x_prod_list.extend(unit_cap_i_j)

        x_prod_list = [*set(x_prod_list)]
        '''
        x_prod_list = gbv.prod_key

        # obtain the variable names: x
        x_name = ["x[{},{},{}]".format(x_prod[0], x_prod[1], t) for x_prod in x_prod_list for t in gbv.period_list]

        #r_list = [item for item in gbv.alt_list if i in item[2]]
        r_list = gbv.alt_list
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


def extensive_prob(solve_option=True, relax_option=True):
    # set up the extensive formulation
    # set up the extensive formulation
    global gbv
    ext_prob = op.AbstractModel(name="extensive_form")

    # set up model parameters (M: plant, T: time, L: transit,
    ext_prob.idx_item = op.Index(name="I", range=[(0, len(gbv.item_list))])
    ext_prob.idx_plant = op.Index(name="J", range=[(0, len(gbv.plant_list))])
    ext_prob.idx_period = op.Index(name="T", range=[(0, len(gbv.period_list))])
    ext_prob.idx_transit = op.Index(name="L", range=[(0, len(gbv.transit_list))])
    ext_prob.idx_alt = op.Index(name="A", range=[(0, len(gbv.alt_list))])
    ext_prob.idx_cap_key = op.Index(name="Cap_Key", range=[(0, len(gbv.cap_key))])
    ext_prob.idx_cap_period = op.Index(name="Cap_Period", range=[(0, len(gbv.cap_period_list))])

    ext_prob.u = op.Variable(ext_prob.idx_item, ext_prob.idx_period, lb=0, name="u")  # u_{it} for t, unmet demand
    ext_prob.s = op.Variable(ext_prob.idx_transit, ext_prob.idx_period, lb=0, name="s")  # s_{ilt} for l,t
    ext_prob.z = op.Variable(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, lb=0, name="z")  # z_{ijt} for j,t
    ext_prob.v = op.Variable(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, lb=0, name="v")  # v_{ijt} for j,t
    ext_prob.yUI = op.Variable(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, lb=0, name="yi")  # y^{I}_{ijt} for j,t
    ext_prob.yUO = op.Variable(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, lb=0, name="yo")  # y^{O}_{ijt} for j,t
    ext_prob.xC = op.Variable(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, lb=0, name="x")  # x_{ijt} for i,j,t
    ext_prob.rC = op.Variable(ext_prob.idx_alt, lb=0, name="r")  # r_{ajt} for a=(i,i')

    # initial condition setup
    # production for non-existent item-plant pair set to 0
    non_exist_prod_pair_constr = {}
    for i in range(len(gbv.item_list)):
        for j in range(len(gbv.plant_list)):
            if (gbv.item_list[i], gbv.plant_list[j]) not in gbv.prod_key:
                non_exist_prod_pair_constr[i, j] = op.Constraint(ext_prob.idx_period['t'],
                                          lhs=ext_prob.xC[i, j, 't'],
                                          sense='=',
                                          rhs=0)
    ext_prob.non_exist_prod_pair = non_exist_prod_pair_constr
    '''
    for i in len(gbv.item_list):
        ext_prob.u[str(i), "0"] = 0.0  # initial unmet demand set to 0
    for l in gbv.transit_list:
        for t in range(min(gbv.period_list) - gbv.transit_time[l], min(gbv.period_list)):
            ext_prob.s[str(gbv.transit_list.index(l)), str(t)] = 0.0  # initial transportation set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if (i, j) in gbv.init_inv.keys():
                ext_prob.v[str(gbv.item_list.index(i)), str(gbv.plant_list.index(j)), "0"] = gbv.init_inv[i, j]  # initial inventory set to given values
            else:
                ext_prob.v[str(gbv.item_list.index(i)), str(gbv.plant_list.index(j)), "0"] = 0.0
    for i, j in gbv.prod_key:
        for t in range(min(gbv.period_list) - gbv.lead_time[i, j], min(gbv.period_list)):
            ext_prob.xC[str(gbv.item_list.index(i)), str(gbv.plant_list.index(j)), str(t)] = 0.0  # initial production set to 0
    for i in gbv.item_list:
        for j in gbv.plant_list:
            if not ((i, j) in gbv.prod_key):
                for t in gbv.period_list:
                    ext_prob.xC[i, j, t] = 0.0  # production for non-existent item-plant pair set to 0
    '''
    
    # add constraints for the extensive formulation
    # unmet demand
    real_demand_dict = {(i, t): gbv.real_demand.get((gbv.item_list[i], t), 0) for i in range(len(gbv.item_list)) for t in range(len(gbv.period_list))}
    ext_prob.para_real_demand = op.Param(ext_prob.idx_item, ext_prob.idx_period, name="real_demand", init=real_demand_dict)
    ext_prob.unmet_demand_init = op.Constraint(ext_prob.idx_item['i'],
                                          lhs=ext_prob.u['i', 0],
                                          sense='=',
                                          rhs=ext_prob.para_real_demand['i', 0] - op.sum_by(e=ext_prob.z['i', 'j', 0], by=[ext_prob.idx_plant['j']]))             
    ext_prob.unmet_demand = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_period['t'],
                                          lhs=ext_prob.u['i', 't'],
                                          sense='=',
                                          rhs=ext_prob.u['i', ['t', -1]] + ext_prob.para_real_demand['i', 't'] - op.sum_by(e=ext_prob.z['i', 'j', 't'], by=[ext_prob.idx_plant['j']]),
                                          filter=['T_t > 0'])
    
    # inventory
    init_inventory_dict = {(i, j): gbv.init_inv.get((gbv.item_list[i], gbv.plant_list[j]), 0) for i in range(len(gbv.item_list)) for j in range(len(gbv.plant_list))}
    ext_prob.para_init_inventory = op.Param(ext_prob.idx_item, ext_prob.idx_plant, name="initial_inventory", init=init_inventory_dict)
    ext_prob.inventory_init = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'],
                                          lhs=ext_prob.v['i', 'j', 0],
                                          sense='=',
                                          rhs=ext_prob.para_init_inventory['i', 'j'] + ext_prob.yUI['i', 'j', 0] - ext_prob.yUO['i', 'j', 0])      
    ext_prob.inventory = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'], ext_prob.idx_period['t'],
                                          lhs=ext_prob.v['i', 'j', 't'],
                                          sense='=',
                                          rhs=ext_prob.v['i', 'j', ['t', -1]] + ext_prob.yUI['i', 'j', 't'] - ext_prob.yUO['i', 'j', 't'],
                                          filter=['T_t > 0'])
    
    # inbound
    inbound_constr = {}
    para_transit_in = {}
    external_purchase_dict = {(i, j, t): gbv.external_purchase.get((gbv.item_list[i], gbv.plant_list[j], t), 0) 
                              for i in range(len(gbv.item_list)) for j in range(len(gbv.plant_list)) for t in range(len(gbv.period_list))}
    ext_prob.para_external_purchase = op.Param(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, name="external_purchase", init=external_purchase_dict)
    for i in range(len(gbv.item_list)):
        for j in range(len(gbv.plant_list)):
            for t in range(len(gbv.period_list)):
                transit_lt = []
                for l in range(len(gbv.transit_list)):
                    for t_prime in range(len(gbv.period_list)):
                        if (gbv.transit_list[l][0] == gbv.item_list[i]) and (gbv.transit_list[l][2] == gbv.plant_list[j]) and (t_prime == t - gbv.transit_time[gbv.transit_list[l]]):
                            transit_lt[(l, t_prime)] = 1
                        else:
                            transit_lt[(l, t_prime)] = 0
                para_transit_in[i, j, t] = op.Param(ext_prob.idx_transit, ext_prob.idx_period, name="transit_in" + str((i, j, t)), init=transit_lt)
                
                x_flag = 0
                if (gbv.item_list[i], gbv.plant_list[j]) in gbv.prod_key:
                    if t - gbv.lead_time[gbv.item_list[i], gbv.plant_list[j]] >= 0:
                        x_flag = 1
                        
                if x_flag == 1:
                    inbound_constr[i, j, t] = op.Constraint(lhs=ext_prob.yUI[i, j, t],
                                                         sense='=',
                                                         rhs=op.sum_by(cond=para_transit_in[i, j, t]['l', 't_prime'], e=ext_prob.s['l', 't_prime'], by=[ext_prob.idx_transit['l'], ext_prob.idx_period['t_prime']]) +
                                                         ext_prob.xC[i, j, t - gbv.lead_time[gbv.item_list[i], gbv.plant_list[j]]] + 
                                                         ext_prob.para_external_purchase[i, j, t]
                                                         )
                else:
                    inbound_constr[i, j, t] = op.Constraint(lhs=ext_prob.yUI[i, j, t],
                                                         sense='=',
                                                         rhs=op.sum_by(cond=para_transit_in[i, j, t]['l', 't_prime'], e=ext_prob.s['l', 't_prime'], by=[ext_prob.idx_transit['l'], ext_prob.idx_period['t_prime']]) +
                                                         ext_prob.para_external_purchase[i, j, t]
                                                         )
    ext_prob.inbound = inbound_constr

    # outbound
    outbound_constr = {}
    para_transit_out = {}
    para_bom = {}
    para_alt_out = {}
    para_alt_in = {}
    for i in range(len(gbv.item_list)):
        for j in range(len(gbv.plant_list)):
            transit_l = []
            for l in range(len(gbv.transit_list)):
                if (gbv.transit_list[l][0] == gbv.item_list[i]) and (gbv.transit_list[l][1] == gbv.plant_list[j]):
                    transit_l[l] = 1
                else:
                    transit_l[l] = 0
            para_transit_out[i, j] = op.Param(ext_prob.idx_transit, name="transit_out" + str((i, j)), init=transit_l)
            
            if gbv.plant_list[j] not in gbv.bom_key.keys():
                bom_e = {i_prime: 0 for i_prime in range(len(gbv.item_list))}
            else:
                bom_e = {}
                for bk in gbv.bom_key[gbv.plant_list[j]]:
                    if bk[1] == gbv.item_list[i]:
                        bom_e[gbv.item_list.index(bk[0])] = gbv.bom_dict[gbv.plant_list[j]][bk]
                for i_prime in range(len(gbv.item_list)):
                    if i_prime not in bom_e.keys():
                        bom_e[i_prime] = 0
            para_bom[i, j] = op.Param(ext_prob.idx_item, name="bom" + str((i, j)), init=bom_e)
            
            for t in range(len(gbv.period_list)):
                alt_out = []
                alt_in = []
                for jta in range(len(gbv.alt_list)):
                    if (gbv.alt_list[jta][0] == gbv.plant_list[j]) and (gbv.alt_list[jta][1] == t) and (gbv.alt_list[jta][2][0] == gbv.item_list[i]):
                        alt_out[jta] = gbv.alt_dict[gbv.alt_list[jta]]
                        alt_in[jta] = 0
                    elif (gbv.alt_list[jta][0] == gbv.plant_list[j]) and (gbv.alt_list[jta][1] == t) and (gbv.alt_list[jta][2][1] == gbv.item_list[i]):
                        alt_out[jta] = 0
                        alt_in[jta] = gbv.alt_dict[gbv.alt_list[jta]]
                    else:
                        alt_out[jta] = 0
                        alt_in[jta] = 0   
                para_alt_out[i, j, t] = op.Param(ext_prob.idx_alt, name="alt_out" + str((i, j, t)), init=alt_out)
                para_alt_in[i, j, t] = op.Param(ext_prob.idx_alt, name="alt_in" + str((i, j, t)), init=alt_in)

                outbound_constr[i, j, t] = op.Constraint(lhs=ext_prob.yUO[i, j, t],
                                                         sense='=',
                                                         rhs=op.sum_by(cond=para_transit_out[i, j]['l'], e=ext_prob.s['l', t], by=[ext_prob.idx_transit['l']]) +
                                                         op.sum_by(cond=para_bom[i, j]['i_prime'], e=ext_prob.xC['i_prime', j, t], by=[ext_prob.idx_item['i_prime']]) +
                                                         ext_prob.z[i, j, t] + 
                                                         op.sum_by(cond=para_alt_out[i, j, t]['a'], e=ext_prob.rC['a'], by=[ext_prob.idx_alt['a']]) - 
                                                         op.sum_by(cond=para_alt_in[i, j, t]['a'], e=ext_prob.rC['a'], by=[ext_prob.idx_alt['a']]))
    ext_prob.outbound = outbound_constr

    # capacity
    max_cap_dict = {(ct_j, t): gbv.max_cap[gbv.cap_key[ct_j]].get(gbv.cap_period_list[t], 0) 
                    for ct_j in range(len(gbv.cap_key)) for t in range(len(gbv.cap_period_list))}
    ext_prob.para_max_cap = op.Param(ext_prob.idx_cap_key, ext_prob.idx_cap_period, name="max_capacity", init=max_cap_dict)
    capacity_constr = {}
    para_unit_cap = {}
    for ct_j in range(len(gbv.cap_key)):
        for t in range(len(gbv.cap_period_list)):
            unit_cap_i = {i: gbv.unit_cap.get((gbv.item_list[i], gbv.cap_key[ct_j][1], gbv.cap_key[ct_j][0]), 0) for i in range(len(gbv.item_list))}
            para_unit_cap[ct_j, t] = op.Param(ext_prob.idx_item, name="unit_cap" + str((ct_j, t)), init=unit_cap_i)

            capacity_constr[ct_j, t] = op.Constraint(lhs=op.sum_by(cond=para_unit_cap[ct_j, t]['i'], e=ext_prob.xC['i', gbv.plant_list.index(gbv.cap_key[ct_j][1]), gbv.cap_period_list[t]], by=[ext_prob.idx_item['i']]),                                                       
                                                         sense='<=',
                                                         rhs=ext_prob.para_max_cap[ct_j, t])
    ext_prob.capacity = capacity_constr
    
    # variable upper bound
    # r upper bound
    r_ub_constr = {}
    for i in range(len(gbv.item_list)):
        for j in range(len(gbv.plant_list)):
            for t in range(len(gbv.period_list)):
                alt_ijt = []
                for a in range(len(gbv.gbv.alt_list)):
                    if (gbv.alt_list[a][0] == gbv.plant_list[j]) and (gbv.alt_list[a][1] == t) and (gbv.alt_list[a][2][0] == gbv.item_list[i]):
                        alt_ijt.append(a)
                
                if len(alt_ijt) > 0:
                    if t > 0:
                        r_ub_constr[i, j, t] = op.Constraint(ext_prob.idx_alt['a'],
                                                             lhs=ext_prob.rC['a'],
                                                             sense='<=',
                                                             rhs=ext_prob.v[i, j, t - 1],
                                                             filter={"a": alt_ijt})
                    elif t == 0:
                        r_ub_constr[i, j, 0] = op.Constraint(ext_prob.idx_alt['a'],
                                                             lhs=ext_prob.rC['a'],
                                                             sense='<=',
                                                             rhs=ext_prob.para_init_inventory[i, j],
                                                             filter={"a": alt_ijt})
    ext_prob.r_ub = r_ub_constr
    # y^o upper bound
    ext_prob.yo_ub_init = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'],
                                          lhs=ext_prob.yUO['i', 'j', 0],
                                          sense='=',
                                          rhs=ext_prob.para_init_inventory['i', 'j'])      
    ext_prob.yo_ub = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'], ext_prob.idx_period['t'],
                                          lhs=ext_prob.yUO['i', 'j', 't'],
                                          sense='<=',
                                          rhs=ext_prob.v['i', 'j', ['t', -1]])


    if not (relax_option):
        # if we require an integer number of batches
        w = op.Variable(ext_prob.idx_item, ext_prob.idx_plant, ext_prob.idx_period, lb=0, type=op.INTEGER, name="w")  # w_{ijt} for i,j,t
        # batch
        lot_size_dict = {(i, j): gbv.lot_size.get((gbv.item_list[i], gbv.plant_list[j]), 0) for i in range(len(gbv.item_list)) for j in range(len(gbv.plant_list))}
        ext_prob.para_lot_size = op.Param(ext_prob.idx_item, ext_prob.idx_plant, name="lot_size", init=lot_size_dict)
        ext_prob.batch = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'], ext_prob.idx_period['t'],
                                          lhs=ext_prob.xC['i', 'j', 't'],
                                          sense='=',
                                          rhs=ext_prob.para_lot_size['i', 'j'] * ext_prob.w['i', 'j', 't'])  
        # max production upper bound
        max_prod_dict = {(i, j): gbv.max_prod.get((gbv.item_list[i], gbv.plant_list[j]), 0) for i in range(len(gbv.item_list)) for j in range(len(gbv.plant_list))}
        ext_prob.para_max_prod = op.Param(ext_prob.idx_item, ext_prob.idx_plant, name="max_production", init=max_prod_dict)
        ext_prob.w_ub = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'], ext_prob.idx_period['t'],
                                          lhs=ext_prob.w['i', 'j', 't'],
                                          sense='<=',
                                          rhs=ext_prob.para_max_prod['i', 'j'] / ext_prob.para_lot_size['i', 'j'])  
    else:
        # if we can relax the integer constraint
        # max production upper bound
        max_prod_dict = {(i, j): gbv.max_prod.get((gbv.item_list[i], gbv.plant_list[j]), 0) for i in range(len(gbv.item_list)) for j in range(len(gbv.plant_list))}
        ext_prob.para_max_prod = op.Param(ext_prob.idx_item, ext_prob.idx_plant, name="max_production", init=max_prod_dict)
        ext_prob.x_ub = op.Constraint(ext_prob.idx_item['i'], ext_prob.idx_plant['j'], ext_prob.idx_period['t'],
                                          lhs=ext_prob.xC['i', 'j', 't'],
                                          sense='<=',
                                          rhs=ext_prob.para_max_prod['i', 'j'])  

    # set up the subproblem specific objective
    holding_cost_dict = {i: gbv.holding_cost[gbv.item_list[i]] for i in range(len(gbv.item_list))}
    penalty_cost_dict = {i: gbv.penalty_cost[gbv.item_list[i]] for i in range(len(gbv.item_list))}
    transit_cost_dict = {l: gbv.transit_cost[gbv.transit_list[l]] for l in range(len(gbv.transit_list))}
    ext_prob.para_holding_cost = op.Param(ext_prob.idx_item, name="holding_cost", init=holding_cost_dict)
    ext_prob.para_penalty_cost = op.Param(ext_prob.idx_item, name="penalty_cost", init=penalty_cost_dict)
    ext_prob.para_transit_cost = op.Param(ext_prob.idx_transit, name="transit_cost", init=transit_cost_dict)

    ext_prob.obj = op.Objective(expr=op.sum_by(cond=ext_prob.para_holding_cost['i'], e=ext_prob.v['i', 'j', 't'], 
                                               by=[ext_prob.idx_item['i'], ext_prob.idx_plant['j'], ext_prob.idx_period['t']]) + 
                                    op.sum_by(cond=ext_prob.para_penalty_cost['i'], e=ext_prob.u['i', 't'], 
                                               by=[ext_prob.idx_item['i'], ext_prob.idx_period['t']]) + 
                                    op.sum_by(cond=ext_prob.para_transit_cost['l'], e=ext_prob.s['l', 't'], 
                                               by=[ext_prob.idx_transit['l'], ext_prob.idx_period['t']]),
                                    sense=op.MIN)

    if solve_option:
        instance = ext_prob.build_instance(instance_name='production_planning')
        solver = op.SolverFactory(solver=op.SolverList.OPTVERSE_CPP)
        instance.solve(solver=solver)
    else:
        return ext_prob, [ext_prob.u, ext_prob.s, ext_prob.z, ext_prob.v, ext_prob.yUI, ext_prob.yUO, ext_prob.xC, ext_prob.rC]


if __name__ == '__main__':
    data_folder = "data/fine_tune"

    # create shared information and manager
    gbv = create_gbv(data_folder)

    extensive_prob()
    
