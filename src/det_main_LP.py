# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 01:01:06 2022

@author: haoxiang
"""

'''
main structure of running decomposition instances
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

from src.params import ADMMparams
from src.readin import *
from src.GlobalVariable import *
from src.variables import global_var

def create_gbv(path_file):
    '''
       This function creates a global data structure to contain all the problem parameters.
    '''
    gbv = GlobalVariable()

    # load the data and form a data structure and create the global instance data frame
    gbv.item_list, gbv.holding_cost, gbv.penalty_cost = input_item(os.path.join(path_file, "dm_df_item.csv"))
    gbv.K = len(gbv.item_list)
    gbv.item_set, gbv.set_list = input_item_set(os.path.join(path_file, "dm_df_item_set.csv"))
    gbv.alt_list, gbv.alt_dict, gbv.alt_cost = input_item_alternate(os.path.join(path_file, "dm_df_alternate_item.csv"))
    gbv.unit_cap, gbv.unit_cap_type = input_unit_capacity(os.path.join(path_file, "dm_df_unit_capacity.csv"))
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

class gbvManager(BaseManager):
    pass
gbvManager.register('gbv',GlobalVariable)

# run the ADMM process
if __name__ == '__main__':
    # read in ADMM meta information
    args = parse_arguments()

    # process the hyperparameters
    hparams = ADMMparams(args.params_file)

    # record the primal/dual residual
    pri_re = []
    d_re = []

    # set up maximum iteration and tolerance terms
    # MaxIte = args.max_iter
    # ep_abs = args.eps_abs
    # ep_rel = args.eps_rel

    # initialize the iteration, primal residual, dual residual
    iter_no = 0
    primal_residual = hparams.init_pr
    dual_residual = hparams.init_dr

    # create a global parameter manager for problem data
    global gbv
    gbv = create_gbv(args.data_folder)

    # obtain the x problem function handle
    from src.admm_models import *
    x_no = args.number
    x_prob_model = eval(args.type)
    x_solve = eval("x_solve")
    z_solve = eval("z_solve")
    pi_solve = eval("pi_solve")

    # TODO: need to register the problem type to the manager!!!!!
    # gbv_manager = gbvManager()

    # ------------- initialize the primal and dual variables (needs to do a parallel version) ------------
    # initialize the primal local/global variable information
    global global_vars, global_ind, prob_list
    global_vars, global_ind, prob_list = x_item_init(x_prob_model)
    sqrt_dim = np.sqrt(sum(len(global_ind[i][gvar_ind]) for gvar_ind in range(len(global_vars)) for i in range(len(global_ind))))

    # start timing
    time_start = time.time()

    UB = math.inf
    LB = -math.inf
    rho = 10.0
    rel_tol = hparams.rel_tol

    # obtain the initial z solution
    local_results = []
    for i in gbv.item_list:
        prob_ind = gbv.item_list.index(i)
        local_sol = x_solve(prob_ind, rho, False)
        local_results.append(local_sol)
    z_solve(local_results, rho)
    primal_residual_record = []
    dual_residual_record = []

    while primal_residual > hparams.p_tol or dual_residual > hparams.d_tol:
       iter_no += 1

       # set up the termination criterion
       if iter_no > hparams.iter_tol:
           break
       else:
           if UB < math.inf and LB > -math.inf:
               rel_err = (UB - LB) / (UB + 1e-10)
               if rel_err < rel_tol:
                   break

       # solve x problem in parallel
       local_results = []
       for i in gbv.item_list:
           prob_ind = gbv.item_list.index(i)
           local_sol = x_solve(prob_ind, rho)
           local_results.append(local_sol)

       # solve z problem, obtain dual residual
       dual_residual, primal_tol_norm = z_solve(local_results, rho)

       # update dual variables, obtain primal residual
       primal_residual, dual_tol_norm = pi_solve(local_results, rho)

       # upper bound and lower bound calculation

       # calculate the primal/dual tolerance
       primal_tol = sqrt_dim * hparams.eps_abs + hparams.eps_rel * primal_tol_norm
       dual_tol = sqrt_dim * hparams.eps_abs + hparams.eps_rel * dual_tol_norm
       primal_residual_record.append(primal_residual)
       dual_residual_record.append(dual_residual)

       # update the rho penalty term
       if primal_residual > 10 * dual_residual and primal_residual > primal_tol:
           rho *= 2
       elif dual_residual > 10 * primal_residual and dual_residual > dual_tol:
           rho /= 2