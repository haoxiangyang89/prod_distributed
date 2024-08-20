# extensive formulation with new data format
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
from multiprocessing import set_start_method
from multiprocessing import get_context

#from src.params import ADMMparams
#from readin import *
#from GlobalVariable import *
#from variables import global_var

from random import random
from time import sleep
import collections
from readin_array import *

from gurobipy import gurobipy as gp
from gurobipy import GRB

import pandas as pd
#from scipy.sparse import csr_array, coo_array
import scipy.sparse

import itertools
import copy
import csv
import pickle

def extensive():
    # data preparation

    mp = gp.Model("master_prob")
    
    x = mp.addVars(prod_key, period_df.I, vtype=GRB.INTEGER, lb=0.0, name="x")
        
    mp.addConstrs(
        (x[i, j, t] * lot_size[i, j] <= max_prod[i,j] for i, j in prod_key for t in range(T_norm)),
        name='batch')
    mp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] * lot_size[i_iter, j_iter] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')
    
    # set up model parameters (M: plant, T: time, L: transit,
    ui = mp.addVars(item_df.I, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = mp.addVars(df_transit.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi = mp.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = mp.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = mp.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = mp.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t

    # initial condition setup
    for i in item_df.I:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in item_df.I:
        for j in plant_df.I:
            vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
    for l in df_transit.Tr:
        for t in range(min(period_df.I) - transit_time[l], min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            x[i, j, t] = 0.0  # initial production set to 0

    rCi = mp.addVars(df_alt.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    mp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in item_df.I for t in period_df.I), name='unmet_demand')
    mp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi[i, j, t] == 0 for i in item_df.I \
         for j in plant_df.I for t in period_df.I), name='inventory')

    mp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[df_transit.Tr[l], t - transit_time[l]] for l in df_transit.index[(df_transit.I == i) & (df_transit.JJ == j)]) +
                   (x[i, j, t - lead_time[i, j]] * lot_size[i, j] if (i, j) in prod_key else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(df_alt.V[jta] * rCi[jta] for jta in df_alt.index[(df_alt.J == j) & (df_alt.Ti == t) & (df_alt.II == i)]) \
                   for i in item_df.I for j in plant_df.I for t in period_df.I),
                  name='input_item')

    mp.addConstrs((yUOi[i, j, t] == gp.quicksum(
        si[df_transit.Tr[l], t] for l in df_transit.index[(df_transit.I == i) & (df_transit.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * x[df_bom.I[bk], j, t] * lot_size[df_bom.I[bk], j] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)]) + zi[i, j, t] +
                   gp.quicksum(df_alt.V[jta] * rCi[jta] for jta in df_alt.index[(df_alt.J == j) & (df_alt.Ti == t) & (df_alt.I == i)])
                   for i in item_df.I for j in plant_df.I for t in period_df.I),
                  name="output_item")

    mp.addConstrs((rCi[jta] <= vi[df_alt.I[jta], df_alt.J[jta], df_alt.Ti[jta] - 1] for jta in df_alt.index), name='r_ub')

    mp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in item_df.I
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # set up the subproblem specific objective
    obj = gp.quicksum(holding_cost[i] * vi[i, j, t] for i in item_df.I for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in item_df.I for t in period_df.I) + \
          gp.quicksum(transit_cost[df_transit.Tr[l]] * si[df_transit.Tr[l], t] for l in df_transit.index for t in period_df.I)    
    
    mp.setObjective(obj, GRB.MINIMIZE)
    mp.update()
    mp.optimize()

    sol = {}
    for i,j in prod_key:
        for t in period_df.I:
            sol[i,j,t] = x[i,j,t].X

    return sol, mp.ObjVal

if __name__ == "__main__":
    # read in data
    data_folder = "./data/fine_tune"  # pilot_test/fine_tune
    global df_cost, df_alt, df_iniInv, df_external, df_unitCap, df_prod, df_transit, \
        df_transitTC, df_bom, df_maxCap, df_demand, item_df, plant_df, period_df, set_df, \
        I_norm, J_norm, T_norm, Ct_norm, Tr_norm
    df_cost, df_alt, df_iniInv, df_external, df_unitCap, df_prod, df_transit, \
        df_transitTC, df_bom, df_maxCap, df_demand, item_df, plant_df, period_df, set_df, \
        I_norm, J_norm, T_norm, Ct_norm, Tr_norm = readin_array(data_folder)

    global prod_key, lot_size, max_prod, unit_cap, max_cap
    prod_key = list(zip(df_prod.I, df_prod.J))
    lot_size = np.zeros((I_norm, J_norm))
    lot_size[df_prod.I, df_prod.J] = df_prod.LS
    max_prod = np.zeros((I_norm, J_norm))
    max_prod[df_prod.I, df_prod.J] = df_prod.MaxProd
    unit_cap = dict(zip(list(zip(df_unitCap.I, df_unitCap.J, df_unitCap.Ct)), df_unitCap.V))
    max_cap = dict(zip(list(zip(df_maxCap.Ct, df_maxCap.J, df_maxCap.Ti)), df_maxCap.V))

    global lead_time, real_demand, external_purchase, init_inv, holding_cost, penalty_cost, transit_cost, transit_time 
    lead_time = np.zeros((I_norm, J_norm))
    lead_time[df_prod.I, df_prod.J] = df_prod.LT
    real_demand = np.zeros((I_norm,T_norm))
    real_demand[df_demand.I, df_demand.Ti] = df_demand.V
    external_purchase = np.zeros((I_norm,J_norm,T_norm))
    external_purchase[df_external.I, df_external.J, df_external.Ti] = df_external.V
    init_inv = np.zeros((I_norm,J_norm))
    init_inv[df_iniInv.I, df_iniInv.J] = df_iniInv.V
    holding_cost = dict(zip(df_cost.I, df_cost.HC))
    penalty_cost = dict(zip(df_cost.I, df_cost.PC))
    transit_cost = dict(zip(df_transitTC.Tr, df_transitTC.TC))
    transit_time = dict(zip(df_transitTC.Tr, df_transitTC.V))
    
    # item clustering such that items with substitutution relationship are in the same cluster
    global alt_item_pair_list, alt_items_list, decomp_units, bom_list
    alt_item_pair_list = list(zip(df_alt.I, df_alt.II))
    alt_item_pair_list = [tuple(sorted(tuple_)) for tuple_ in alt_item_pair_list]
    alt_item_pair_list = list(dict.fromkeys(alt_item_pair_list))
    
    alt_items_list = list(itertools.chain(*alt_item_pair_list))

    decomp_units = [list(alt_pair) for alt_pair in alt_item_pair_list]
    for i in list(item_df.I):
        if i not in alt_items_list:
            decomp_units.append([i])
    
    bom_pair = df_bom[["I","II"]].drop_duplicates()
    bom_list = list(zip(bom_pair.I, bom_pair.II))

    sol_ext, obj_ext = extensive()
    with open('extensive_fine_tune.p', 'wb') as fp:
        pickle.dump(sol_ext, fp, protocol=pickle.HIGHEST_PROTOCOL)