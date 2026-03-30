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

def master_prob(sol_method, cut_opt, reg_method, level_obj_norm):
    mp = gp.Model("master_prob")
    
    if sol_method == "master_sub_loop":
        mp.Params.TimeLimit = 10 * 60

    prod_key = list(zip(df_prod.I, df_prod.J))
    lot_size = np.zeros((I_norm, J_norm))
    lot_size[df_prod.I, df_prod.J] = df_prod.LS
    max_prod = dict(zip(prod_key, df_prod.MaxProd))
    unit_cap = dict(zip(list(zip(df_unitCap.I, df_unitCap.J, df_unitCap.Ct)), df_unitCap.V))
    max_cap = dict(zip(list(zip(df_maxCap.Ct, df_maxCap.J, df_maxCap.Ti)), df_maxCap.V))

    # item clustering such that items with substitutution relationship are in the same cluster
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
    bom_value = np.zeros((I_norm, I_norm, J_norm))
    bom_value[df_bom.I, df_bom.II, df_bom.J] = df_bom.V
    
    x = mp.addVars(prod_key, period_df.I, vtype=GRB.INTEGER, lb=0.0, name="x")
    
    if cut_opt == "multi_cut":
        theta = mp.addVars(range(len(decomp_units)), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    elif cut_opt == "single_cut":
        theta = mp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    
    mp.addConstrs(
        (x[i, j, t] * lot_size[i, j] <= max_prod[i,j] for i, j in prod_key for t in range(T_norm)),
        name='batch')
    mp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] * lot_size[i_iter, j_iter] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    if sol_method == "gb_callback":
        if cut_opt == "multi_cut":
            mp.setObjective(gp.quicksum(theta[unit_idx] for unit_idx in range(len(decomp_units))), GRB.MINIMIZE)
        elif cut_opt == "single_cut":
            mp.setObjective(theta, GRB.MINIMIZE)
    elif sol_method == "master_sub_loop":
        if reg_method == "level_set" and level_obj_norm == 1:
            x_dev_p = mp.addVars(prod_key, period_df.I, lb=0.0, name="x_dev_p")
            x_dev_m = mp.addVars(prod_key, period_df.I, lb=0.0, name="x_dev_m")

    # set up the simple production bounds
    # mp.addConstrs((gp.quicksum(
    #         bom_value[i,ii,j] * x[i,j,tau] * lot_size[i,j] for j in plant_df.I for tau in range(t+1) if (i,j) in prod_key) -
    #         sum(init_inv[ii,j] for j in plant_df.I) + sum(external_purchase[ii, j ,tau] for j in plant_df.I for tau in range(t)) - 
    #         gp.quicksum(x[ii,j,tau] * lot_size[ii,j] for j in plant_df.I for tau in range(t) if (ii,j) in prod_key) <= 0 
    #             for i,ii in bom_list for t in period_df.I), name = "simple") 

    mp.update()

    return mp, [x, theta]

def sub_primal_prob(unit_ind_list, x_vals, penalty_mag=1e5, orig_dual=False):
    global df_alt, df_transit, df_bom, item_df, plant_df, period_df
    global prod_key, lot_size, max_prod, unit_cap, max_cap, lead_time, real_demand, external_purchase, \
        init_inv, unit_cap, max_cap, holding_cost, penalty_cost, transit_cost, transit_time

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_m = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_m")  # z^+_{ijt} for j,t
    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_m = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_m")  # y^{O,+}_{ijt} for j,t
    yUOi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    xCi = sp.addVars(prod_key, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - transit_time[l], min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = sp.addVars(alt_i.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] - zi_m[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in unit_ind_list for t in period_df.I), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi_p[i, j, t] - yUOi_m[i, j, t] == 0 for i in unit_ind_list \
         for j in plant_df.I for t in period_df.I), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t - transit_time[l]] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - lead_time[i, j]] if (i, j) in prod_key else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)]) + zi_p[i, j, t] - zi_m[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i,j] for i, j in prod_key for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # fix the local variables to their global counterparts' values
    sub_fix_x = {}
    for i, j in prod_key:
        for t in period_df.I:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] == x_vals[i, j, t] * lot_size[i, j])
    # set up the subproblem specific objective
    obj = gp.quicksum(
        holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
          gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
          gp.quicksum(
              penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)

    sp.update()

    sp.optimize()
    '''
    if sp.Status == 3:
        wr = "\nSubPrimalprob " + str(unit_ind_list) + " infeasible\n"
        sp.computeIIS()
        sp.write("model1.ilp")

        f = open("model1.ilp")
        while True:
            line = f.readline()
            line = line[:-1]
            if (line):
                wr += line + "\n"
            else:
                break
            f.close()

        wr_s = open("dual_status.txt", "a")
        wr_s.write(wr)
    if sp.Status == 5:
        wr = "\nSubPrimalprob " + str(unit_ind_list) + " unbounded"
        wr_s = open("dual_status.txt", "a")
        wr_s.write(wr)
    '''
    #print("\nSubPrimalprob " + str(unit_ind_list) + " : " + "Status " + str(sp.Status) + ", Objective " + str(sp.ObjVal))
    # print("Subprob " + str(unit_ind_list) + " Status " + str(sp.Status))
    # if dummy_unit_index == 0 and sp.Status == 3:
    #    sp.computeIIS()
    #    sp.write("model1.ilp")

    dual_coeff = np.zeros((I_norm, J_norm, T_norm))

    # obtain the objective value and return
    local_obj = sp.ObjVal

    for i, j in prod_key:
        for t in period_df.I:
            if abs(sub_fix_x[i, j, t].Pi) > 1e-4:
                if orig_dual:
                    dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi
                else:
                    dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi * lot_size[i, j]
            else:
                dual_coeff[i, j, t] = 0

    return local_obj, dual_coeff

def sub_primal_Lag(unit_ind_list, x_vals, pi_vals, penalty_mag=1e5):
    # Lagrangian relaxation problem to test the cut coefficients
    global df_alt, df_transit, df_bom, item_df, plant_df, period_df
    global prod_key, lot_size, max_prod, unit_cap, max_cap, lead_time, real_demand, external_purchase, \
        init_inv, unit_cap, max_cap, holding_cost, penalty_cost, transit_cost, transit_time

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_m = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_m")  # z^+_{ijt} for j,t
    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi_m = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_m")  # y^{O,+}_{ijt} for j,t
    yUOi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    xCi = sp.addVars(prod_key, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - transit_time[l], min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = sp.addVars(alt_i.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] - zi_m[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in unit_ind_list for t in period_df.I), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi_p[i, j, t] - yUOi_m[i, j, t] == 0 for i in unit_ind_list \
         for j in plant_df.I for t in period_df.I), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t - transit_time[l]] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - lead_time[i, j]] if (i, j) in prod_key else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)]) + zi_p[i, j, t] - zi_m[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i,j] for i, j in prod_key for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi_p[i, j, t] - yUOi_m[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # set up the subproblem specific objective
    obj = gp.quicksum(
        holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
          gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
          gp.quicksum(
              penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(pi_vals[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t]) for i, j in prod_key for t in period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)

    sp.update()

    sp.optimize()

    # obtain the objective value and return
    local_obj = sp.ObjVal
    xCi_grad_vals = np.zeros((I_norm, J_norm, T_norm))
    for i, j in prod_key:
        for t in period_df.I:
            xCi_grad_vals[i,j,t] = x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t].X

    return local_obj, xCi_grad_vals

def pi_iter(k, data, x_vals, theta_vals):
    # Attempt to find the flat constraints with better numerical properties

    worker_init(data)    
    penalty_mag = 20 * np.max(list(penalty_cost.values()))
    unit_ind_list = decomp_units[k]

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    xCi = sp.addVars(prod_key, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    x_lim = np.zeros((I_norm, J_norm, T_norm))
    for t in period_df.I:
        x_lim[:,:,t] = max_prod
    x_im_p = sp.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, ub=x_lim, name="x_p")  # x_imb_p_{ijt} for i,j,t (x copy)
    x_im_m = sp.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, ub=x_lim, name="x_m")  # x_imb_p_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - transit_time[l], min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xCi[i, j, t] = 0.0  # initial production set to 0

    rCi = sp.addVars(alt_i.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in unit_ind_list for t in period_df.I), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi[i, j, t] == 0 for i in unit_ind_list \
         for j in plant_df.I for t in period_df.I), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t - transit_time[l]] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - lead_time[i, j]] if (i, j) in prod_key else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)]) + zi_p[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i,j] for i, j in prod_key for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # fix the local variables to their global counterparts' values
    sub_fix_x = {}
    for i, j in prod_key:
        for t in period_df.I:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] + x_im_p[i,j,t] - x_im_m[i,j,t] == x_vals[i, j, t] * lot_size[i, j])
    # set up the subproblem specific objective
    obj = gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key for t in period_df.I)
    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    if sp.ObjVal > 1e-2:
        feas_gen = True
        feas_value = sp.ObjVal
        feas_cut_coeff = np.zeros((I_norm, J_norm, T_norm))
        for i, j in prod_key:
            for t in period_df.I:
                if abs(sub_fix_x[i, j, t].Pi * lot_size[i, j]) > 1e-4:
                    feas_cut_coeff[i, j, t] = sub_fix_x[i, j, t].Pi * lot_size[i, j]
                else:
                    feas_cut_coeff[i, j, t] = 0
    else:
        feas_gen = False
        feas_value = 0
        feas_cut_coeff = np.zeros((I_norm, J_norm, T_norm))

    obj = gp.quicksum(
        holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
          gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
          gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key for t in period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    sub_obj = sp.ObjVal
    dual_coeff = np.zeros((I_norm, J_norm, T_norm))
    for i, j in prod_key:
        for t in period_df.I:
            if abs(sub_fix_x[i, j, t].Pi) > 1e-4:
                dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi * lot_size[i, j]
            else:
                dual_coeff[i, j, t] = 0

    #print(k, " ", sub_obj, " ", theta_vals[k])
    if ((sub_obj - theta_vals[k]) > 1e-4) and (sub_obj > 1.00001 * theta_vals[k]):
        cut_gen = True
        rho = np.maximum(np.linalg.norm(dual_coeff) * 10/sub_obj,1)
        #rho = 100

        smcp = gp.Model()
        smcp.setParam("OutputFlag",0)
        thetav = smcp.addVar(ub=1.01*rho*sub_obj, name="thetav")
        pi = smcp.addVars(prod_key, period_df.I, lb=-float('inf'), name="beta")
        pi_abs = smcp.addVars(prod_key, period_df.I, lb=0.0, name="beta_abs")
        smcp.setObjective(rho * thetav - gp.quicksum(pi_abs[i,j,t] for i,j in prod_key for t in period_df.I), GRB.MAXIMIZE)
        smcp.addConstrs(pi_abs[i,j,t] >= pi[i,j,t] for i,j in prod_key for t in period_df.I)
        smcp.addConstrs(pi_abs[i,j,t] >= -pi[i,j,t] for i,j in prod_key for t in period_df.I)
        smcp.update()
        smcp.optimize()
        ub_smcp = smcp.ObjVal

        # initialize the pi_hat as 0
        pi_hat = np.zeros((I_norm, J_norm, T_norm))

        iter_num = 0

        for i, j in prod_key:
            for t in period_df.I:
                sp.remove(sub_fix_x[i, j, t])

        current_lb = 0
        best_dual = copy.deepcopy(dual_coeff)
        alpha = 0.5
        
        while (ub_smcp > 1.01 * current_lb) and (iter_num <= 500):
            iter_num += 1
            #print(iter_num," ", ub_smcp," ",current_lb)
            
            # generate the cut to characterize the Lagrangian function            
            obj = gp.quicksum(
                holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
                gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
                gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
                gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key for t in period_df.I) + \
                gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t] - x_im_p[i,j,t] + x_im_m[i,j,t]) for i, j in prod_key for t in period_df.I)
                # gp.quicksum(
                #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

            sp.setObjective(obj, GRB.MINIMIZE)
            sp.update()
            sp.optimize()

            pi_obj = sp.ObjVal
            lb_cand = rho * pi_obj - sum(abs(pi_hat[i,j,t]) for i,j in prod_key for t in period_df.I)
            if lb_cand > current_lb:
                current_lb = lb_cand
                best_dual = copy.deepcopy(pi_hat)
                best_dual[abs(best_dual) < 1e-4] = 0.0
            x_grad_vals = np.zeros((I_norm, J_norm, T_norm))
            for i, j in prod_key:
                for t in period_df.I:
                    x_grad_vals[i,j,t] = x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t].X  - x_im_p[i,j,t].X + x_im_m[i,j,t].X

            # adding the cuts for the Lagrangian function
            smcp.addConstr(thetav <= pi_obj + gp.quicksum(x_grad_vals[i,j,t] * (pi[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key for t in period_df.I))
            smcp.setObjective(rho * thetav - gp.quicksum(pi_abs[i,j,t] for i,j in prod_key for t in period_df.I), GRB.MAXIMIZE)
            smcp.update()
            smcp.optimize()
            ub_smcp = smcp.ObjVal

            if (ub_smcp > 1.01 * current_lb):
                # adding the cuts to obtain the next pi
                smcp.setObjective(gp.quicksum(
                        (pi[i, j, t] - best_dual[i, j, t]) ** 2 for i, j in prod_key for t in
                        period_df.I), GRB.MINIMIZE)
                level = alpha * ub_smcp + (1-alpha) * current_lb
                level_con = smcp.addConstr(rho * thetav - gp.quicksum(pi_abs[i,j,t] for i,j in prod_key for t in period_df.I) >= level)
                smcp.update()
                smcp.optimize()
                if smcp.Status == 12:
                    smcp.setParam("NumericFocus",3)
                    smcp.update()
                    smcp.optimize()
                    smcp.remove(level_con)
                else:
                    smcp.remove(level_con)
                try:
                    for i,j in prod_key:
                        for t in period_df.I:
                            if abs(pi[i,j,t].X) > 1e-4:
                                pi_hat[i,j,t] = pi[i,j,t].X
                            else:
                                pi_hat[i,j,t] = 0
                except:
                    pass

        pi_hat = best_dual
        obj = gp.quicksum(
            holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
            gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
            gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
            gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key for t in period_df.I) + \
            gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t] - x_im_p[i,j,t] + x_im_m[i,j,t]) for i, j in prod_key for t in period_df.I)
        # gp.quicksum(
        #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

        sp.setObjective(obj, GRB.MINIMIZE)
        sp.update()
        sp.optimize()

        pi_obj = sp.ObjVal
        for i, j in prod_key:
            for t in period_df.I:
                pi_hat[i,j,t] *= lot_size[i,j]
        print(multiprocessing.current_process(), " ", k, " ", pi_obj, " ", sub_obj, " ", iter_num)
    else:
        cut_gen = False
        pi_hat = np.zeros((I_norm, J_norm, T_norm))
        pi_obj = sub_obj
        print(multiprocessing.current_process(), " ", k, " ", sub_obj)

    return k, cut_gen, sub_obj, pi_obj, pi_hat, feas_gen, feas_cut_coeff, feas_value

def sub_dual_prob(unit_ind_list, x_vals, penalty_mag=1e5):
    global df_alt, df_transit, df_bom, item_df, plant_df, period_df
    global prod_key, lot_size, max_prod, unit_cap, max_cap, lead_time, real_demand, external_purchase, \
        init_inv, unit_cap, max_cap, holding_cost, penalty_cost, transit_cost, transit_time
    
    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sd = gp.Model("item_{}".format(unit_ind_list))
    sd.setParam("OutputFlag", 0)
    sd.setParam("DualReductions", 0)
    # sd.Params.TimeLimit = 5 * 60

    # set dual variables
    dual_u = sd.addVars(unit_ind_list, period_df.I, lb=-float('inf'), ub=float('inf'), name="dual_unmet_demand")
    dual_v = sd.addVars(unit_ind_list, plant_df.I, period_df.I, lb=-float('inf'), ub=float('inf'),
                        name="dual_inventory")
    dual_I = sd.addVars(unit_ind_list, plant_df.I, period_df.I, lb=-float('inf'), ub=float('inf'),
                        name="dual_input_item")
    dual_O = sd.addVars(unit_ind_list, plant_df.I, period_df.I, lb=-float('inf'), ub=float('inf'),
                        name="dual_output_item")

    ct_j_t_list = list(max_cap.keys())
    dual_C = sd.addVars(ct_j_t_list, lb=-float('inf'), ub=0.0, name="dual_capacity")

    dual_rUB = sd.addVars(alt_i.index, lb=-float('inf'), ub=0.0, name="dual_r_ub")

    dual_OUB = sd.addVars(unit_ind_list, plant_df.I, period_df.I, lb=-float('inf'), ub=0.0, name="dual_yo_ub")
    dual_xUB = sd.addVars(prod_key, period_df.I, lb=-float('inf'), ub=0.0, name="dual_x_ub")

    pi = sd.addVars(prod_key, period_df.I, lb=-float('inf'), ub=float('inf'), name="dual_x_hat")

    # zero-padding
    for i, j in prod_key:
        if i in unit_ind_list:
            for t in range(max(period_df.I) + 1, max(period_df.I) + int(lead_time[i, j]) + 1):
                dual_I[i, j, t] = 0.0
        else:
            for t in range(min(period_df.I), max(period_df.I) + int(lead_time[i, j]) + 1):
                dual_I[i, j, t] = 0.0
                dual_O[i, j, t] = 0.0

    # dual constraints
    # coeff_v >= 0
    sd.addConstrs(holding_cost[i] - dual_v[i, j, t] + dual_v[i, j, t + 1] + dual_OUB[i, j, t + 1] +
                  gp.quicksum(dual_rUB[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t + 1) & (alt_i.I == i)]) >= 0
                  for i in unit_ind_list for j in plant_df.I for t in
                  range(min(period_df.I), max(period_df.I)))
    sd.addConstrs(
        holding_cost[i] - dual_v[i, j, max(period_df.I)] >= 0 for i in unit_ind_list for j in plant_df.I)

    # coeff_u >= 0
    sd.addConstrs(penalty_cost[i] - dual_u[i, t] + dual_u[i, t + 1] >= 0 for i in unit_ind_list for t in
                  range(min(period_df.I), max(period_df.I)))
    sd.addConstrs(penalty_cost[i] - dual_u[i, max(period_df.I)] >= 0 for i in unit_ind_list)

    # coeff_s >= 0
    sd.addConstrs(transit_cost[transit_list_i.Tr[l]] + dual_I[transit_list_i.I[l], transit_list_i.JJ[l], t + transit_time[l]] + \
                  dual_O[transit_list_i.I[l], transit_list_i.J[l], t] >= 0
                  for l in transit_list_i.index for t in
                  range(min(period_df.I), max(period_df.I) - transit_time[l] + 1)
                  if transit_time[l] <= max(period_df.I) - min(period_df.I))
    sd.addConstrs(transit_cost[transit_list_i.Tr[l]] + dual_O[transit_list_i.I[l], transit_list_i.J[l], t] >= 0
                  for l in transit_list_i.index for t in
                  range(max(period_df.I) - transit_time[l] + 1, max(period_df.I) + 1))

    # coeff_z >= 0
    sd.addConstrs(
        dual_u[i, t] - dual_O[i, j, t] <= 0 for i in unit_ind_list for j in plant_df.I for t in period_df.I)
    sd.addConstrs(-dual_u[i, t] + dual_O[i, j, t] <= penalty_mag for i in unit_ind_list for j in plant_df.I for t in
                  period_df.I)

    # coeff_yUI >= 0
    sd.addConstrs(
        dual_v[i, j, t] - dual_I[i, j, t] >= 0 for i in unit_ind_list for j in plant_df.I for t in period_df.I)

    # coeff_yUO >= 0
    sd.addConstrs(
        dual_v[i, j, t] + dual_OUB[i, j, t] + dual_O[i, j, t] <= 0 for i in unit_ind_list for j in plant_df.I for t
        in period_df.I)
    sd.addConstrs(-dual_v[i, j, t] - dual_OUB[i, j, t] - dual_O[i, j, t] <= penalty_mag for i in unit_ind_list for j in
                  plant_df.I for t in period_df.I)

    # coeff_r >= 0
    sd.addConstrs(-dual_rUB[jta] + alt_i.V[jta] * (
                dual_I[alt_i.II[jta], alt_i.J[jta], alt_i.Ti[jta]] + dual_O[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta]]) >= 0 for jta in alt_i.index)

    # coeff_x >= 0
    sd.addConstrs(-dual_xUB[i, j, t] - pi[i, j, t] + dual_I[i, j, t + lead_time[i, j]]
                  - gp.quicksum(unit_cap.get((i, j, ct_j_t[0]), 0.0) * dual_C[ct_j_t] for ct_j_t in max_cap.keys() if
                                ct_j_t[1] == j and ct_j_t[2] == t)
                  + gp.quicksum(df_bom.V[bk] * dual_O[df_bom.II[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.I == i) & (df_bom.II.isin(unit_ind_list))])
                  >= 0 for i, j in prod_key for t in period_df.I)

    obj = gp.quicksum(real_demand[i, t] * dual_u[i, t] for i in unit_ind_list for t in period_df.I) \
          + gp.quicksum(
        init_inv[i, j] * (dual_v[i, j, min(period_df.I)] + dual_OUB[i, j, min(period_df.I)]) for i in unit_ind_list for j in
        plant_df.I) \
          + gp.quicksum(max_cap[ct, j, t] * dual_C[ct, j, t] for ct, j, t in max_cap.keys()) \
          + gp.quicksum(max_prod[i, j] * dual_xUB[i, j, t] for i, j in prod_key for t in period_df.I) \
          + gp.quicksum(x_vals[i, j, t] * lot_size[i, j] * pi[i, j, t] for i, j in prod_key for t in period_df.I) \
          + gp.quicksum(
        init_inv[alt_i.I[jta], alt_i.J[jta]] * dual_rUB[jta] for jta in alt_i.index[alt_i.Ti == min(period_df.I)]) \
          + gp.quicksum(
        external_purchase[i, j, t] * dual_I[i, j, t] for i in unit_ind_list for j in plant_df.I for t in
        period_df.I)

    sd.setObjective(obj, GRB.MAXIMIZE)

    sd.update()

    sd.optimize()
    '''
    if sd.Status == 3:
        wr = "\nSubDualprob " + str(unit_ind_list) + " infeasible\n"
        sd.computeIIS()
        sd.write("model1.ilp")

        f = open("model1.ilp")
        while True:
            line = f.readline()
            line = line[:-1]
            if (line):
                wr += line + "\n"
            else:
                break
            f.close()

        wr_s = open("dual_status.txt", "a")
        wr_s.write(wr)
    if sd.Status == 5:
        wr = "\nSubDualprob " + str(unit_ind_list) + " unbounded"
        wr_s = open("dual_status.txt", "a")
        wr_s.write(wr)
    '''
    #print("SubDualprob " + str(unit_ind_list) + " : " + "Status " + str(sd.Status) + ", Objective " + str(sd.ObjVal))
    #wr = "\nSubDualprob " + str(unit_ind_list) + " : " + "Status " + str(sd.Status) + ", Objective " + str(sd.ObjVal)
    #print(wr)
    # wr_s = open("benders.txt", "a")
    # wr_s.write(wr)
    # wr_s.close()

    cut_coeff = np.zeros((I_norm, J_norm, T_norm))
    for i, j in prod_key:
        for t in period_df.I:
            if abs(pi[i, j, t].X) > 1e-4:
                cut_coeff[i, j, t] = pi[i, j, t].X * lot_size[i, j]

    local_obj = sd.ObjVal

    return local_obj, cut_coeff

def mycallback(model, where):
    prod_key = list(zip(df_prod.I, df_prod.J))
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
        for i, j in prod_key:
            for t in period_df.I:
                wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
                # wr += str(int(x_vals[i, j, t])) + ","
        wr += "\nMaster theta_val:"
        for k in range(len(decomp_units)):
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
            with Pool(min(len(decomp_units), 4)) as pool:
                sub_opt_results = pool.map(partial(sub_prob, x_vals=x_vals_dict), range(len(gbv.decomp_units)))

            #sub_opt_results = []
            #for k in range(len(gbv.decomp_units)):
            #    sub_opt_results.append(sub_prob(k, x_vals))
        elif dual_opt == "direct_solve":
            x_vals_dict = {(i, j, t): x_vals[i, j, t] for i, j, t in x_vals.keys()}
            with Pool(min(len(decomp_units), 4)) as pool:
                sub_opt_results = pool.map(partial(compute_sub_dual, x_vals=x_vals_dict), range(len(decomp_units)))

            #sub_opt_results = []
            #for k in range(len(gbv.decomp_units)):
            #    sub_opt_results.append(compute_sub_dual(k, x_vals))

        #ub_cand = 0
        for k in range(len(decomp_units)):
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
                    for i, j in prod_key for t in period_df.I))

                #cut_coeff[iter_num, k] = {(i, j, t): (sub_dual[i, j, t], x_vals[i, j, t]) for i, j in gbv.prod_key for t
                #                          in gbv.period_list}
                #cut_coeff[iter_num, k][0] = sub_obj

                wr = "\nOptimality cut:" + "\n\ttheta" + str([k]) + " >= " + str(sub_obj)
                for i, j in prod_key:
                    for t in period_df.I:
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

def worker_init(data):
    global df_cost, df_alt, df_iniInv, df_external, df_unitCap, df_prod, df_transit, \
        df_transitTC, df_bom, df_maxCap, df_demand, item_df, plant_df, period_df, set_df, \
        I_norm, J_norm, T_norm, Ct_norm, Tr_norm
    
    df_cost = data["df_cost"]
    df_alt = data["df_alt"]
    df_iniInv = data["df_iniInv"]
    df_external = data["df_external"]
    df_unitCap = data["df_unitCap"]
    df_prod = data["df_prod"]
    df_transit = data["df_transit"]
    df_transitTC = data["df_transitTC"]
    df_bom = data["df_bom"]
    df_maxCap = data["df_maxCap"]
    df_demand = data["df_demand"]
    item_df = data["item_df"]
    plant_df = data["plant_df"]
    period_df = data["period_df"]
    set_df = data["set_df"]
    I_norm = data["I_norm"]
    J_norm = data["J_norm"]
    T_norm = data["T_norm"]
    Ct_norm = data["Ct_norm"]
    Tr_norm = data["Tr_norm"]
    
    # initialize all the parameters used
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

if __name__ == "__main__":
    # read in data
    data_folder = "./data/fine_tune"  # pilot_test/fine_tune
    set_start_method("spawn")
    df_cost, df_alt, df_iniInv, df_external, df_unitCap, df_prod, df_transit, \
        df_transitTC, df_bom, df_maxCap, df_demand, item_df, plant_df, period_df, set_df, \
        I_norm, J_norm, T_norm, Ct_norm, Tr_norm = readin_array(data_folder)
    gbv = {}
    gbv["df_cost"] = df_cost
    gbv["df_alt"] = df_alt
    gbv["df_iniInv"] = df_iniInv
    gbv["df_external"] = df_external
    gbv["df_unitCap"] = df_unitCap
    gbv["df_prod"] = df_prod
    gbv["df_transit"] = df_transit
    gbv["df_transitTC"] = df_transitTC
    gbv["df_bom"] = df_bom
    gbv["df_maxCap"] = df_maxCap
    gbv["df_demand"] = df_demand
    gbv["item_df"] = item_df
    gbv["plant_df"] = plant_df
    gbv["period_df"] = period_df
    gbv["set_df"] = set_df
    gbv["I_norm"] = I_norm
    gbv["J_norm"] = J_norm
    gbv["T_norm"] = T_norm
    gbv["Ct_norm"] = Ct_norm
    gbv["Tr_norm"] = Tr_norm
        
    # initialize the Benders procedure
    max_iter = 100
    iter_num = 0
    best_lb = -math.inf
    best_ub = math.inf

    ##### regularization method: level_set / regularized BD
    reg_method = None
    # reg_method = "level_set"
    level_fractile = 0.5
    level_obj_norm = 1

    best_cand_ub = math.inf
    # initialization of best solution x
    best_incum = {}

    ##### solution method: callback / for loop
    #sol_method = "gb_callback"
    sol_method = "master_sub_loop"

    ##### access to dual variables: gurobio dual / solution to dual problem
    #dual_opt = "gurobi_dual"
    dual_opt = "direct_solve"

    ##### Benders multi-cut vs single cut
    #cut_opt = "single_cut"
    cut_opt = "multi_cut"

    mp, mp_handle = master_prob(sol_method, cut_opt, reg_method, level_obj_norm)

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
        worker_init(gbv)

        # level set initialization
        if reg_method == "level_set":
            level = 1e10
            if cut_opt == "multi_cut":
                level_constr = mp.addConstr(
                    gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))) <= level)
            elif cut_opt == "single_cut":
                level_constr = mp.addConstr(mp.getVarByName(f"theta") <= level)
        mp.update()

        ubList = []
        lbList = []

        while True:
            iter_num += 1

            wr = "\nIteration: " + str(iter_num)
            wr_s = open("benders.txt", "a")
            wr_s.write(wr)
            wr_s.close()

            if reg_method == None:
                if cut_opt == "multi_cut":
                    mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
                elif cut_opt == "single_cut":
                    mp.setObjective(mp.getVarByName(f"theta"))
                mp.update()
                mp.optimize()
                
                x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
                if cut_opt == "multi_cut":
                    theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}
                    lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))
                elif cut_opt == "single_cut":
                    theta_vals = mp_handle[1].X
                    lb_cand = theta_vals
                    
                lbList.append(lb_cand)
                if lb_cand > best_lb:
                    best_lb = lb_cand
            elif reg_method == "level_set":
                # master
                level_constr.rhs = math.inf
                if cut_opt == "multi_cut":
                    mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
                elif cut_opt == "single_cut":
                    mp.setObjective(mp.getVarByName(f"theta"))
                mp.update()
                mp.optimize()
                
                # best incumbent initialization
                if iter_num == 1:
                    best_incum = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
                
                if cut_opt == "multi_cut":
                    theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}
                    lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))
                elif cut_opt == "single_cut":
                    theta_vals = mp_handle[1].X
                    lb_cand = theta_vals

                if lb_cand > best_lb:
                    best_lb = lb_cand
                lbList.append(lb_cand)

                # auxiliary problem, refinement to master
                # update level set auxiliary problem
                if best_ub < math.inf and best_lb > -math.inf:
                    level = best_lb + level_fractile * (best_ub - best_lb)
                    level_constr.rhs = level

                if level_obj_norm == 2:
                    obj = gp.quicksum(
                        (mp.getVarByName(f"x[{i},{j},{t}]") - best_incum[i, j, t]) ** 2 for i, j in prod_key for t in
                        period_df.I)
                elif level_obj_norm == 1:
                    if iter_num == 1:
                        x_dev_constr = mp.addConstrs(
                            mp.getVarByName(f"x[{i},{j},{t}]") + mp.getVarByName(f"x_dev_p[{i},{j},{t}]")
                            - mp.getVarByName(f"x_dev_m[{i},{j},{t}]") == best_incum[i, j, t] for i, j in prod_key for t
                            in period_df.I)
                    elif iter_num > 1:
                        for i, j in prod_key:
                            for t in period_df.I:
                                x_dev_constr[i, j, t].rhs = best_incum[i, j, t]

                    obj = gp.quicksum(
                        (mp.getVarByName(f"x_dev_p[{i},{j},{t}]") + mp.getVarByName(f"x_dev_m[{i},{j},{t}]")) for i, j in
                        prod_key for t in period_df.I)

                mp.setObjective(obj)
                mp.update()
                mp.optimize()

                x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
                if cut_opt == "multi_cut":
                    theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}
                elif cut_opt == "single_cut":
                    theta_vals = mp_handle[1].X
                
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
                for i, j in prod_key:
                    for t in period_df.I:
                        wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
                wr += "\nMaster theta_val:"
                if cut_opt == "multi_cut":
                    for k in range(len(decomp_units)):
                        wr += str(k) + " : " + str(theta_vals[k]) + ",\t"
                elif cut_opt == "single_cut":
                    wr += str(theta_vals)
                #wr_s = open("benders.txt", "a")
                #wr_s.write(wr)
                #wr_s.close()

            # subproblem solution given a production level

            # optimality cut
            with get_context("spawn").Pool(processes=6, maxtasksperchild=1) as pool:   # 30
            #with Pool(processes=6) as pool:
                sub_opt_results = pool.map(partial(pi_iter, data=gbv, x_vals=x_vals, theta_vals=theta_vals), range(len(decomp_units)), chunksize=5)

            ub_cand = 0
            for item in sub_opt_results:
                k = item[0]

                cut_gen = item[1]
                sub_obj = item[2]
                pi_obj = item[3]
                sub_dual = item[4]
                feas_gen = item[5]
                feas_dual = item[6]
                feas_value = item[7]

                ub_cand += sub_obj

                if feas_gen:
                    mp.addConstr(0 >= feas_value + gp.quicksum(
                        feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                        for i, j in prod_key for t in period_df.I))
                    mp.update()

                if cut_gen:
                    # optimality cut
                    for i, j in prod_key:
                        for t in period_df.I:
                            if abs(sub_dual[i, j, t]) < 1e-4:
                                sub_dual[i, j, t] = 0
                    mp.addConstr(mp.getVarByName(f"theta[{k}]") >= pi_obj + gp.quicksum(
                        sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                        for i, j in prod_key for t in period_df.I))
            
            ubList.append(ub_cand)

            if ub_cand < best_ub:
                best_ub = ub_cand

                # write best x so far to csv
                header = ['I', 'J', 'Ti', 'V']

                with open("best_batch.csv", 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)

                    # write the header
                    writer.writerow(header)

                    # write the data
                    for i, j, t in x_vals.keys():
                        data_output = [i, j, t, x_vals[i, j, t]]
                        writer.writerow(data_output)
                
                # reain best x into a dictionary
                #df_best_batch =  readin_csv("best_batch.csv")
                #best_batch = unit_cap = dict(zip(list(zip(df_best_batch.I, df_best_batch.J, df_best_batch.Ti)), df_best_batch.V))
                

            mp.update()
            # mp.write("out_" + str(iter_num) + ".lp")

            if best_ub < math.inf and best_lb > -math.inf:
                gap = (best_ub - best_lb) / best_ub
            else:
                gap = math.inf

            print("ub: ", ub_cand, "lb: ", lb_cand, "best_ub: ", best_ub, "best_lb: ", best_lb)
            wr = "\nCurrent Upper Bound: " + str(ub_cand) + "\nCurrent Lower Bound: " + str(lb_cand) + \
                "\nBest Upper Bound: " + str(best_ub) + "\nBest Lower Bound: " + str(best_lb) + "\n"
            wr_s = open("benders.txt", "a")
            wr_s.write(wr)
            wr_s.close()

            if iter_num >= max_iter or gap < 0.05:
                break
        end = time.time()

        wr = "\nEnd in " + str(end - start) + " seconds!"
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()