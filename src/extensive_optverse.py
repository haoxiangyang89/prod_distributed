# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:58:01 2022

@author: haoxiang
"""

'''
Solve the extensive formulation for the multi-period production problem
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
import optvom as op
import pandas as pd

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


def data_proc(gbv):
    opt_data = dict()
    I_norm = len(gbv.item_list)
    J_norm = len(gbv.plant_list)
    T_norm = len(gbv.period_list)
    Tr_norm = len(gbv.transit_list)
    Ct_norm = len(gbv.cap_list)

    # ep.real_demand = op.Param(ep.I, ep.T, name="demand")
    itList = [(i, t) for (i, t) in gbv.real_demand.keys()]
    df_demand = pd.DataFrame(data={'I': [gbv.item_list.index(it[0]) for it in itList],
                                   'T': [gbv.period_list.index(it[1]) for it in itList],
                                   'V': [gbv.real_demand[it[0], it[1]] for it in itList]})
    opt_data['demand'] = op.df_to_coo(df=df_demand,
                                      index_columns=['I', 'T'],
                                      data_column='V',
                                      shape=(I_norm, T_norm))

    # ep.external_purchase = op.Param(ep.I, ep.J, ep.T, name = "external_purchase")
    ijtList = [(i, j, t) for (i, j, t) in gbv.external_purchase.keys()]
    df_external = pd.DataFrame(data={'I': [gbv.item_list.index(it[0]) for it in ijtList],
                                     'J': [gbv.plant_list.index(it[1]) for it in ijtList],
                                     'T': [gbv.period_list.index(it[2]) for it in ijtList],
                                     'V': [gbv.external_purchase[it[0], it[1], it[2]] for it in ijtList]})
    opt_data['external_purchase'] = op.df_to_coo(df=df_external,
                                                 index_columns=['I', 'J', 'T'],
                                                 data_column='V',
                                                 shape=(I_norm, J_norm, T_norm))

    # ep.transit_time = op.Param(ep.Tr, name = "transit_time")
    df_transitTC = pd.DataFrame(data={'Tr': [tr for tr in range(Tr_norm)],
                                      'V': [gbv.transit_time[gbv.transit_list[tr]] for tr in range(Tr_norm)],
                                      'TC': [gbv.transit_cost[gbv.transit_list[tr]] for tr in range(Tr_norm)]})

    opt_data['transit_time'] = op.df_to_coo(df=df_transitTC,
                                            index_columns=['Tr', ],
                                            data_column='V',
                                            shape=(Tr_norm,))

    # ep.s_shift_helper = op.Param(ep.Tr, ep.T, ep.T, name="s_shift_helper")
    df_t = pd.DataFrame(data={'T': gbv.period_list})
    df_tr = df_transitTC[['Tr', 'V']]
    df_t['aux_key'], df_tr['aux_key'] = 0, 0
    df_s_shift = pd.merge(left=df_tr, right=df_t, on=['aux_key'])
    df_s_shift.pop('aux_key')
    df_s_shift['T_2'] = df_s_shift['T'] - df_s_shift['V']
    df_s_shift = df_s_shift.query("T_2 >= 0")

    opt_data['s_shift_helper'] = op.df_to_coo(df=df_s_shift,
                                              index_columns=['Tr', 'T', 'T_2'],
                                              data=1,
                                              shape=(Tr_norm, T_norm, T_norm))

    # ep.transit_cost = op.Param(ep.Tr, name = "transit_cost")
    opt_data['transit_cost'] = op.df_to_coo(df=df_transitTC,
                                            index_columns=['Tr', ],
                                            data_column='TC',
                                            shape=(Tr_norm,))

    # ep.transit_data = op.Param(ep.I, ep.J, ep.Tr, name = "transit_data")
    df_transit = pd.DataFrame(data={'I': [gbv.item_list.index(gbv.transit_list[tr][0]) for tr in range(Tr_norm)],
                                    'J': [gbv.plant_list.index(gbv.transit_list[tr][2]) for tr in range(Tr_norm)],
                                    'Tr': [tr for tr in range(Tr_norm)],
                                    'V': [1 for tr in range(Tr_norm)]})
    opt_data['transit_data'] = op.df_to_coo(df=df_transit,
                                            index_columns=['I', 'J', 'Tr'],
                                            data_column='V',
                                            shape=(I_norm, J_norm, Tr_norm))

    # ep.r_transit_data = op.Param(ep.I, ep.J, ep.Tr, name = "r_transit_data")
    df_rtransit = pd.DataFrame(data={'I': [gbv.item_list.index(gbv.transit_list[tr][0]) for tr in range(Tr_norm)],
                                     'J': [gbv.plant_list.index(gbv.transit_list[tr][1]) for tr in range(Tr_norm)],
                                     'Tr': [tr for tr in range(Tr_norm)],
                                     'V': [1 for tr in range(Tr_norm)]})
    opt_data['r_transit_data'] = op.df_to_coo(df=df_rtransit,
                                              index_columns=['I', 'J', 'Tr'],
                                              data_column='V',
                                              shape=(I_norm, J_norm, Tr_norm))

    # ep.lead_time = op.Param(ep.I, ep.J, name = "lead_time")
    df_prod = pd.DataFrame(data={'I': [gbv.item_list.index(it[0]) for it in gbv.prod_key],
                                 'J': [gbv.item_list.index(it[1]) for it in gbv.prod_key],
                                 'LT': [gbv.lead_time[it[0], it[1]] for it in gbv.prod_key],
                                 'pKey': [1 for it in gbv.prod_key],
                                 'MaxProd': [gbv.max_prod[it[0], it[1]] for it in gbv.prod_key],
                                 'LS': [gbv.lot_size[it[0], it[1]] for it in gbv.prod_key]})
    opt_data['lead_time'] = op.df_to_coo(df=df_prod,
                                         index_columns=['I', 'J'],
                                         data_column='LT',
                                         shape=(I_norm, J_norm))

    # ep.xC_shift_helper = op.Param(ep.I, ep.J, ep.T, ep.T, name="xC_shift_helper")
    df_lead_time = df_prod[['I', 'J', 'LT']]
    df_t['aux_key'], df_lead_time['aux_key'] = 0, 0
    df_xC_shift = pd.merge(left=df_lead_time, right=df_t, on=['aux_key'])
    df_xC_shift.pop('aux_key')
    df_xC_shift['T_2'] = df_xC_shift['T'] - df_xC_shift['LT']
    df_xC_shift = df_xC_shift.query("T_2 >= 0")

    opt_data['xC_shift_helper'] = op.df_to_coo(df=df_xC_shift,
                                               index_columns=['I', 'J', 'T', 'T_2'],
                                               data=1,
                                               shape=(I_norm, J_norm, T_norm, T_norm))

    # ep.prod_key = op.Param(ep.I, ep.J, name = "prod_key")
    opt_data['prod_key'] = op.df_to_coo(df=df_prod,
                                        index_columns=['I', 'J'],
                                        data_column='pKey',
                                        shape=(I_norm, J_norm))

    # ep.lot_size = op.Param(ep.I, ep.J, name = "lot_size")
    opt_data['lot_size'] = op.df_to_coo(df=df_prod,
                                        index_columns=['I', 'J'],
                                        data_column='LS',
                                        shape=(I_norm, J_norm))

    # ep.max_prod = op.Param(ep.I, ep.J, name = "max_prod")
    opt_data['max_prod'] = op.df_to_coo(df=df_prod,
                                        index_columns=['I', 'J'],
                                        data_column='MaxProd',
                                        shape=(I_norm, J_norm))

    # ep.bom_dict = op.Param(ep.I, ep.I, ep.J, name = "bom_dict")
    bom_keys = [(bk[0], bk[1], j) for j in gbv.plant_list for bk in gbv.bom_key[j]]
    df_bom = pd.DataFrame(data={'II': [gbv.item_list.index(it[0]) for it in bom_keys],
                                'I': [gbv.item_list.index(it[1]) for it in bom_keys],
                                'J': [gbv.item_list.index(it[2]) for it in bom_keys],
                                'V': [gbv.bom_dict[it[2]][(it[0], it[1])] for it in bom_keys]})
    opt_data['bom_dict'] = op.df_to_coo(df=df_bom,
                                        index_columns=['II', 'I', 'J'],
                                        data_column='V',
                                        shape=(I_norm, I_norm, J_norm))

    # ep.alt_dict = op.Param(ep.I, ep.I, ep.J, ep.T, name = "alt_dict")
    df_alt = pd.DataFrame(data={'I': [gbv.item_list.index(jta[2][0]) for jta in gbv.alt_list],
                                'II': [gbv.item_list.index(jta[2][1]) for jta in gbv.alt_list],
                                'J': [gbv.plant_list.index(jta[0]) for jta in gbv.alt_list],
                                'T': [gbv.period_list.index(jta[1]) for jta in gbv.alt_list],
                                'V': [gbv.alt_dict[jta] for jta in gbv.alt_list]})
    opt_data['alt_dict'] = op.df_to_coo(df=df_alt,
                                        index_columns=['I', 'II', 'J', 'T'],
                                        data_column='V',
                                        shape=(I_norm, I_norm, J_norm, T_norm))

    # ep.unit_cap = op.Param(ep.I, ep.J, ep.Ct, name = "unit_cap")
    itList = [item for item in gbv.unit_cap.keys()]
    df_unitCap = pd.DataFrame(data={'I': [gbv.item_list.index(it[0]) for it in itList],
                                    'J': [gbv.plant_list.index(it[1]) for it in itList],
                                    'Ct': [gbv.cap_list.index(it[2]) for it in itList],
                                    'V': [gbv.unit_cap[it] for it in itList]})
    opt_data['unit_cap'] = op.df_to_coo(df=df_unitCap,
                                        index_columns=['I', 'J', 'Ct'],
                                        data_column='V',
                                        shape=(I_norm, J_norm, Ct_norm))

    # ep.max_cap = op.Param(ep.J, ep.Ct, name = "max_cap")
    itList = [item for item in gbv.max_cap.keys()]
    df_maxCap = pd.DataFrame(data={'J': [gbv.plant_list.index(it[1]) for it in itList],
                                   'Ct': [gbv.cap_list.index(it[0]) for it in itList],
                                   'V': [gbv.max_cap[it] for it in itList]})
    opt_data['max_cap'] = op.df_to_coo(df=df_maxCap,
                                       index_columns=['J', 'Ct'],
                                       data_column='V',
                                       shape=(J_norm, Ct_norm))

    # ep.holding_cost = op.Param(ep.I, name = "holding_cost")
    df_cost = pd.DataFrame(data={'I': [i for i in range(I_norm)],
                                 'HC': [gbv.holding_cost[gbv.item_list[i]] for i in range(I_norm)],
                                 'PC': [gbv.penalty_cost[gbv.item_list[i]] for i in range(I_norm)]})
    opt_data['holding_cost'] = op.df_to_coo(df=df_cost,
                                            index_columns=['I', ],
                                            data_column='HC',
                                            shape=(I_norm,))

    # ep.penalty_cost = op.Param(ep.I, name = "penalty_cost")
    opt_data['penalty_cost'] = op.df_to_coo(df=df_cost,
                                            index_columns=['I', ],
                                            data_column='PC',
                                            shape=(I_norm,))

    return opt_data


def extensive_prob(opt_data, solve_option=True, relax_option=False):
    # set up the extensive formulation
    # set up the extensive formulation
    global gbv
    ep = op.AbstractModel(name="extensive_form", sense=op.MIN)
    # set up the index sets
    IList = [(0, len(gbv.item_list))]
    JList = [(0, len(gbv.plant_list))]
    TList = [(0, len(gbv.period_list))]
    TrList = [(0, len(gbv.transit_list))]
    CtList = [(0, len(gbv.cap_list))]
    ep.I = op.Index(name="i", range=IList)
    ep.J = op.Index(name="j", range=JList)
    ep.T = op.Index(name="t", range=TList)
    ep.Tr = op.Index(name="tr", range=TrList)
    ep.Ct = op.Index(name="ct", range=CtList)

    # set up model parameters
    ep.real_demand = op.Param(ep.I, ep.T, name="demand")
    ep.external_purchase = op.Param(ep.I, ep.J, ep.T, name="external_purchase")
    ep.transit_time = op.Param(ep.Tr, name="transit_time", type=op.INT)
    ep.transit_data = op.Param(ep.I, ep.J, ep.Tr, name="transit_data")
    ep.s_shift_helper = op.Param(ep.Tr, ep.T, ep.T, name="s_shift_helper")
    ep.r_transit_data = op.Param(ep.I, ep.J, ep.Tr, name="r_transit_data")
    ep.lead_time = op.Param(ep.I, ep.J, name="lead_time", type=op.INT)
    ep.prod_key = op.Param(ep.I, ep.J, name="prod_key")
    ep.xC_shift_helper = op.Param(ep.I, ep.J, ep.T, ep.T, name="xC_shift_helper")
    ep.bom_dict = op.Param(ep.I, ep.I, ep.J, name="bom_dict")
    ep.alt_dict = op.Param(ep.I, ep.I, ep.J, ep.T, name="alt_dict")
    ep.unit_cap = op.Param(ep.I, ep.J, ep.Ct, name="unit_cap")
    ep.max_cap = op.Param(ep.J, ep.Ct, name="max_cap")
    ep.lot_size = op.Param(ep.I, ep.J, name="lot_size")
    ep.max_prod = op.Param(ep.I, ep.J, name="max_prod")
    ep.holding_cost = op.Param(ep.I, name="holding_cost")
    ep.penalty_cost = op.Param(ep.I, name="penalty_cost")
    ep.transit_cost = op.Param(ep.Tr, name="transit_cost")

    # set up model variables
    ep.u = op.Variable(ep.I, ep.T, lb=0.0, name="u")  # u_{it} for t, unmet demand
    ep.s = op.Variable(ep.Tr, ep.T, lb=0.0, name="s")  # s_{ilt} for l,t
    ep.z = op.Variable(ep.I, ep.J, ep.T, name="z")  # z_{ijt} for j,t
    ep.v = op.Variable(ep.I, ep.J, ep.T, lb=0.0, name="v")  # v_{ijt} for j,t
    ep.yUI = op.Variable(ep.I, ep.J, ep.T, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    ep.yUO = op.Variable(ep.I, ep.J, ep.T, name="yo")  # y^{O}_{ijt} for j,t
    ep.xC = op.Variable(ep.I, ep.J, ep.T, lb=0.0, name="x")  # x_{ijt} for i,j,t
    ep.rC = op.Variable(ep.I, ep.I, ep.J, ep.T, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # add constraints for the extensive formulation
    ep.unmet_demand = op.Constraint(ep.I['i'], ep.T['t'],
                                    lhs=ep.u['i', 't'] - ep.u['i', ['t', - 1]] + op.sum_by(ep.z['i', 'j', 't'],
                                                                                           by=[ep.J['j'], ]),
                                    sense='=',
                                    rhs=ep.real_demand['i', 't'],
                                    name='unmet_demand')
    ep.inventory = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                                 lhs=ep.v['i', 'j', 't'] - ep.v['i', 'j', ['t', - 1]] - ep.yUI['i', 'j', 't'] + ep.yUO[
                                     'i', 'j', 't'],
                                 sense='=',
                                 rhs=0,
                                 name='inventory')

    # ep.s['tr', ['t', - ep.transit_time['tr']]] = ep.s_shift_helper['tr', 't', 't_2'] * ep.s['tr', 't_2']
    # ep.xC['i', 'j', ['t', - ep.lead_time['i', 'j']]] = ep.xC['i', 'j', 't_2'] * ep.xC_shift_helper['i', 'j', 't','t_2']
    ep.input_item = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                                  lhs=ep.yUI['i', 'j', 't'],
                                  sense='=',
                                  rhs=op.sum_by(e=ep.s_shift_helper['tr', 't', 't_2'] * ep.s['tr', 't_2'],
                                                cond=ep.transit_data['i', 'j', 'tr'], by=[ep.Tr["tr"], ]) + \
                                      ep.xC['i', 'j', 't_2'] * ep.xC_shift_helper['i', 'j', 't', 't_2'] * ep.prod_key[
                                          'i', 'j'] +
                                      ep.external_purchase['i', 'j', 't'],
                                  name='input_item')

    ep.output_item = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                                   lhs=ep.yUO['i', 'j', 't'],
                                   sense='=',
                                   rhs=op.sum_by(e=ep.s['tr', 't'], cond=ep.r_transit_data['i', 'j', 'tr'],
                                                 by=[ep.Tr["tr"], ]) + \
                                       op.sum_by(e=ep.xC['ii', 'j', 't'], cond=ep.bom_dict['ii', 'i', 'j'],
                                                 by=[ep.I['ii'], ]) + ep.z['i', 'j', 't'] + \
                                       op.sum_by(e=ep.rC['i', 'ii', 'j', 't'], cond=ep.alt_dict['i', 'ii', 'j', 't'],
                                                 by=[ep.I['ii'], ]) - \
                                       op.sum_by(e=ep.rC['ii', 'i', 'j', 't'], cond=ep.alt_dict['ii', 'i', 'j', 't'],
                                                 by=[ep.I['ii'], ]),
                                   name='output_item')
    ep.capacity = op.Constraint(ep.J['j'], ep.Ct['ct'],
                                lhs=op.sum_by(e=ep.xC['i', 'j', 't'], cond=ep.unit_cap['i', 'j', 'ct'],
                                              by=[ep.I['i'], ]),
                                sense='<=',
                                rhs=ep.max_cap['j', 'ct'],
                                name='capacity')
    ep.rUB = op.Constraint(ep.I['i'], ep.I['ii'], ep.J['j'], ep.T['t'],
                           lhs=ep.rC['i', 'ii', 'j', 't'],
                           sense='<=',
                           rhs=ep.v['i', 'j', ['t', - 1]],
                           name='rUB')
    ep.yoUB = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                            lhs=ep.yUO['i', 'j', 't'],
                            sense='<=',
                            rhs=ep.v['i', 'j', ['t', - 1]])

    if not (relax_option):
        # if we require an integer number of batches
        ep.wi = op.Variable(ep.I, ep.J, ep.T, type=op.INTEGER, lb=0.0, name="w")  # w_{ijt} for i,j,t
        ep.batch = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                                 lhs=ep.xC['i', 'j', 't'],
                                 sense='=',
                                 rhs=ep.wi['i', 'j', 't'] * ep.lot_size['i', 'j'],
                                 name='batch')
        ep.wUB = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                               lhs=ep.wi['i', 'j', 't'],
                               sense='<=',
                               rhs=ep.max_prod['i', 'j'] / ep.lot_size['i', 'j'],
                               name='wUB')
    else:
        # if we can relax the integer constraint
        ep.prodcap = op.Constraint(ep.I['i'], ep.J['j'], ep.T['t'],
                                   lhs=ep.xC['i', 'j', 't'],
                                   sense='<=',
                                   rhs=ep.max_prod['i', 'j'],
                                   name='prodcap')

    # set up the subproblem specific objective
    ep.obj = op.Objective(expr=
                          op.sum_by(cond=ep.holding_cost['i'], e=ep.v['i', 'j', 't'],
                                    by=[ep.I['i'], ep.J['j'], ep.T['t']]) +
                          op.sum_by(cond=ep.penalty_cost['i'], e=ep.u['i', 't'], by=[ep.I['i'], ep.T['t']]) +
                          op.sum_by(cond=ep.transit_cost['tr'], e=ep.s['tr', 't'], by=[ep.Tr['tr'], ep.T['t']]),
                          name='obj')

    if solve_option:
        instance = ep.build_instance(instance_name='extensive',
                                     data=opt_data)
        # instance.write('Test.lp', True)
        solver_optv = op.SolverFactory(solver=op.SolverList.OPTVERSE_CPP)
        sol = instance.solve(solver=solver_optv, save_solution=True)
        # collect the solution and objective value

        print(sol.objective_value)
        return sol.objective_value
    else:
        return ep


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
    gbv.cap_period_list, gbv.cap_key, gbv.max_cap, gbv.cap_list = input_capacity(
        os.path.join(path_file, "dm_df_max_capacity.csv"))
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

    return gbv


if __name__ == '__main__':
    data_folder = "./Datasets/pilot_test"
    gbv = create_gbv(data_folder)
    opt_data = data_proc(gbv)
    ext_prob_obj = extensive_prob(opt_data, True, True)
