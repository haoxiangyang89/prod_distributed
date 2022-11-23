# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 01:01:06 2022

@author: haoxiang
"""

'''
main structure of running decomposition instances
'''

from src.readin import *
from src.GlobalVariable import *
from src.admm_models import *

import os
from argparse import ArgumentParser
import numpy as np
from numpy import linalg as LA
import time
import math
import matplotlib.pyplot as plt
import random
import multiprocessing

from multiprocessing import Pool
from multiprocessing.managers import BaseManager

def create_gbv(path_file):
    '''
       This function creates a global data structure to contain all the problem parameters.
    '''
    gbv = GlobalVariable()

    # load the data and form a data structure and create the global instance data frame
    gbv.item_list, gbv.holding_cost = input_item(os.path.join(path_file, "dm_df_item.csv"))
    gbv.K = len(gbv.item_list)
    gbv.item_set, gbv.set_list = input_item_set(os.path.join(path_file, "dm_df_item_set.csv"))
    gbv.alt_list, gbv.alt_dict, gbv.alt_cost = input_item_alternate(os.path.join(path_file, "dm_df_alternate_item.csv"))
    gbv.unit_cap, gbv.unit_cap_type = input_unit_capacity(os.path.join(path_file, "dm_df_unit_capacity.csv"))
    gbv.item_plant = input_item_plant(os.path.join(path_file, "dm_df_item_plant.csv"))
    gbv.bom_key, gbv.bom_dict = input_bom(os.path.join(path_file, "dm_df_bom.csv"))
    gbv.prod_key, gbv.lot_size, gbv.lead_time, gbv.holding_cost, \
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
                        default='[]',
                        type=str,
                        help='Global variables')
    parser.add_argument("-m", "--max_iter",
                        default=2000,
                        dest='max_iter',
                        type=int,
                        help='Maximum number of iterations')
    parser.add_argument("-e_a", "--eps_abs",
                        default=0.01,
                        dest='eps_abs',
                        type=float,
                        help='Absolute tolerance level')
    parser.add_argument("-e_r", "--eps_rel",
                        dest="eps_rel",
                        default=0.0001,
                        type=float,
                        help='Relative tolerance level')

    args = parser.parse_args()
    return args

class gbvManager(BaseManager):
    pass
gbvManager.register('gbv',GlobalVariable)

# run the ADMM process
if __name__ == '__main__':
    # read in the x problem type and the shared variables
    args = parse_arguments()

    # record the primal/dual residual
    pri_re = []
    d_re = []

    # set up maximum iteration and tolerance terms
    MaxIte = args.max_iter
    ep_abs = args.eps_abs
    ep_rel = args.eps_rel

    # initialize primal/dual residual及primal/dual tolerance
    pr = 10
    dr = 10
    p_tol = 1e-5
    d_tol = 1e-5

    iter_no = 0

    # obtain the x problem function handle
    x_no = args.number
    x_prob_create = eval(args.type)
    gbv_manager = gbvManager()
    gbv_manager.gbv = create_gbv('./data/pilot_test')

    # need to register the problem type to the manager!!!!!
    gbv = gbv_manager.gbv

    # create all x problems
    processes = [multiprocessing.Process(target=x_prob_create, args=(i,)) for i in gbv_manager.gbv.item_list]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    # initialize the primal and dual variables

    # start timing
    time_start = time.time()

    while pr > p_tol or dr > d_tol:
       iter_no += 1
       if iter_no > MaxIte:
           break

       # solve x problem
