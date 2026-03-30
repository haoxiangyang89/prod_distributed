# Pool implementation, subproblems are rebuilt and solved in parallel in each iteration

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

import random
from time import sleep
import collections
from readin_array import *

import gurobipy as gp
from gurobipy import GRB

import pandas as pd
#from scipy.sparse import csr_array, coo_array
import scipy.sparse

import itertools
import copy
import csv

def master_prob(relax_option=False):
    mp = gp.Model("master_prob")
    
    mp.Params.TimeLimit = 10 * 60
    #mp.Params.MIPGap = 1e-3

    prod_key = list(zip(df_prod.I, df_prod.J))
    lot_size = dict(zip(prod_key, df_prod.LS))
    max_prod = dict(zip(prod_key, df_prod.MaxProd))
    unit_cap = dict(zip(list(zip(df_unitCap.I, df_unitCap.J, df_unitCap.Ct)), df_unitCap.V))
    max_cap = dict(zip(list(zip(df_maxCap.Ct, df_maxCap.J, df_maxCap.Ti)), df_maxCap.V))

    # item clustering such that items with substitutution relationship are in the same cluster
    alt_item_pair_list = list(zip(df_alt.I, df_alt.II))
    alt_item_pair_list = [tuple(sorted(tuple_)) for tuple_ in alt_item_pair_list]
    alt_item_pair_list = list(dict.fromkeys(alt_item_pair_list))
    
    decomp_dict = {i:[i] for i in item_df.I[(item_df.I.isin(df_alt.I)) | (item_df.I.isin(df_alt.II)) | (item_df.I.isin(df_prod.I)) | (item_df.I.isin(df_bom.I)) | (item_df.I.isin(df_bom.II))]}
    for alt_pair in alt_item_pair_list:
        joined_list = list(set(decomp_dict[alt_pair[0]] + decomp_dict[alt_pair[1]]))
        for item in joined_list:
            decomp_dict[item] = joined_list
    decomp_units = []
    item_selected = {i:0 for i in decomp_dict.keys()}
    for i in decomp_dict.keys():
        if item_selected[i] == 0:
            decomp_units.append(decomp_dict[i])
            for item in decomp_dict[i]:
                item_selected[item] = 1
    
    if relax_option:
        x = mp.addVars(prod_key, period_df.I, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    else:
        x = mp.addVars(prod_key, period_df.I, vtype=GRB.INTEGER, lb=0.0, name="x")
    
    if level_obj_norm == 1:
        x_dev = mp.addVars(prod_key, period_df.I, vtype=GRB.CONTINUOUS, lb=0.0, name="x_dev")
    elif level_obj_norm == 1.5:
        global pwl_keys, l1_keys

        all_keys = [(i, j, t) for i, j in prod_key for t in period_df.I]
        m = 5    # number of x variables to generate 2-pieces approximation for 2-norm regularization
        random.seed(22) 

        pwl_keys = random.sample(all_keys, m)
        l1_keys = [k for k in all_keys if k not in pwl_keys]

        y = mp.addVars(all_keys, vtype=GRB.CONTINUOUS, name="y")
        z = mp.addVars(pwl_keys, vtype=GRB.CONTINUOUS, name="z")

        # z = x (template)
        mp.addConstrs(z[k] == x[k] for k in pwl_keys)

    theta = mp.addVars(range(len(decomp_units)), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    
    mp.addConstrs(
        (x[i, j, t] * lot_size[i, j] <= max_prod[i,j] for i, j in prod_key for t in range(T_norm)),
        name='batch')
    mp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] * lot_size[i_iter, j_iter] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')
    
    # logical constraint
    if logic_constr_flag:
        producible_item = [*set(list(df_prod.I))]
        mp.addConstrs((gp.quicksum(x[ii, j, t] * lot_size[ii, j] * composition_list[ii, i] for ii in parent_list[i] for j in plant_df.I if (ii, j) in prod_key)
                       <= sum(init_inv.get((i, j), 0) for j in plant_df.I) + sum(external_purchase.get((i, j, tt), 0) for j in plant_df.I for tt in period_df.I if tt <= t)
                       + gp.quicksum(x[i, j, tt] * lot_size[i, j] for j in plant_df.I for tt in period_df.I if ((i, j) in prod_key) and (tt <= t - int(lead_time[i, j])))
                       for i in item_df.I for t in period_df.I if (len(parent_list[i]) > 0) and (bool(set(producible_item) & set(parent_list[i]))))
                       , name='logical_constr')


    mp.setObjective(gp.quicksum(theta[unit_idx] for unit_idx in range(len(decomp_units))), GRB.MINIMIZE)

    mp.update()

    return mp, [x, theta]

def pi_iter_sparse(k, x_vals, theta_vals, lag_cuts=True, feas_cuts=True, sparse_cuts=True, level_lambda=0.5, level_mu=0.8):
    # Attempt to find the flat constraints with better numerical properties
    #worker_init(data)

    penalty_mag = len(period_df.I) * np.max(list(penalty_cost.values()))
    unit_ind_list = decomp_units[k]

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("Method", 1)
    sp.setParam("DualReductions", 0)
    sp.setParam("Threads",1)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    
    if sparse_cuts:
        # find the related items and create x copies for only them
        item_used_list = []
        # items that are parents of i's
        for i in unit_ind_list:
            item_used_list.append(i)
            for bk in df_bom.index[df_bom.II == i]:
                item_used_list.append(df_bom.I[bk])
        # items that share same plant capacity with i's
        plant_list = df_prod.loc[df_prod["I"].isin(unit_ind_list), "J"].tolist()
        item_used_list.extend(df_prod.loc[df_prod["J"].isin(plant_list), "I"].tolist())
        item_used_list = list(set(item_used_list))

        df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
        prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
        
    else:
        prod_key_used = list(zip(df_prod.I, df_prod.J))
    xCi = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    x_im_p = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x_p")  # x_imb_p_{ijt} for i,j,t (x copy)
    x_im_m = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x_m")  # x_imb_p_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            if (i,j) in init_inv.keys():
                vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
            else:
                vi[i, j, min(period_df.I) - 1] = 0.0
            for t in period_df.I:
                if not((i, j, t) in external_purchase.keys()):
                    external_purchase[i, j, t] = 0.0
        for t in period_df.I:
            if not((i, t) in real_demand.keys()):
                real_demand[i, t] = 0.0
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - int(transit_time[l]), min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key_used:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xCi[i, j, t] = 0.0  # initial production set to 0
        for t in period_df.I:
            x_im_p[i, j, t].UB = max_prod[i, j]
            x_im_m[i, j, t].UB = max_prod[i, j]

    rCi = sp.addVars(alt_i.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in unit_ind_list for t in period_df.I), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi[i, j, t] == 0 for i in unit_ind_list \
         for j in plant_df.I for t in period_df.I), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t - int(transit_time[l])] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - int(lead_time[i, j])] if (i, j) in prod_key_used else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)] if (df_bom.I[bk], j) in prod_key_used) + zi_p[i, j, t] +
                   gp.quicksum(rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i,j] for i, j in prod_key_used for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key_used) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # fix the local variables to their global counterparts' values
    # solve subproblem to generate feasibility cuts
    sub_fix_x = {}
    for i, j in prod_key_used:
        for t in period_df.I:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] + x_im_p[i,j,t] - x_im_m[i,j,t] == x_vals[i, j, t] * lot_size[i, j])
    if feas_cuts:
        # set up the subproblem specific objective
        obj = gp.quicksum(x_im_p[i, j, t] + x_im_m[i, j, t] for i,j in prod_key_used for t in period_df.I)
        sp.setObjective(obj, GRB.MINIMIZE)
        sp.update()
        sp.optimize()

        if sp.ObjVal > 1e-2:
            feas_gen = True
            feas_value = sp.ObjVal
            feas_cut_coeff = {}
            for i, j in prod_key_used:
                for t in period_df.I:
                    if abs(sub_fix_x[i, j, t].Pi * lot_size[i, j]) > 1e-4:
                        feas_cut_coeff[i, j, t] = sub_fix_x[i, j, t].Pi * lot_size[i, j]
                    else:
                        feas_cut_coeff[i, j, t] = 0
        else:
            feas_gen = False
            feas_value = 0
            feas_cut_coeff = 0
    else:
        feas_gen = False
        feas_value = 0
        feas_cut_coeff = 0

    # obtain the LP objective value of the subproblem
    obj = gp.quicksum(
        holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
          gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
          gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key_used for t in period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    sub_obj = sp.ObjVal

    sub_HC = sum(holding_cost[i] * vi[i, j, t].X for i in unit_ind_list for j in plant_df.I for t in period_df.I)
    sub_PC = sum(penalty_cost[i] * ui[i, t].X for i in unit_ind_list for t in period_df.I)
    sub_TC = sum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t].X for l in transit_list_i.index for t in period_df.I)
    sub_dev_cost = sum(penalty_mag * (x_im_p[i, j, t].X + x_im_m[i, j, t].X) for i,j in prod_key_used for t in period_df.I)

    #print(k, " ", sub_obj, " ", theta_vals[k])
    if ((sub_obj - theta_vals[k]) > 1e-4) and (sub_obj > 1.00001 * theta_vals[k]):
        cut_gen = True

        dual_coeff = {}
        for i, j in prod_key_used:
            for t in period_df.I:
                if abs(sub_fix_x[i, j, t].Pi) > 1e-4:
                    dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi
                else:
                    dual_coeff[i, j, t] = 0

        iter_num = 0

        if lag_cuts:
            # initialize the pi_hat as 0
            pi_hat = {}
            best_dual = {}
            GList = []
            fList = []

            # set up the lower bound problem
            smcp = gp.Model()
            smcp.setParam("OutputFlag",0)
            smcp.setParam("Threads",1)
            pi = smcp.addVars(prod_key_used, period_df.I, lb=-float('inf'), name="beta")
            pi_abs = smcp.addVars(prod_key_used, period_df.I, lb=0.0, name="beta_abs")
            smcp.setObjective(gp.quicksum(pi_abs[i,j,t] for i,j in prod_key_used for t in period_df.I), GRB.MINIMIZE)
            smcp.addConstrs(pi_abs[i,j,t] >= pi[i,j,t] for i,j in prod_key_used for t in period_df.I)
            smcp.addConstrs(pi_abs[i,j,t] >= -pi[i,j,t] for i,j in prod_key_used for t in period_df.I)
            smcp.update()
            smcp.optimize()
            current_ub = np.inf
            f_star = smcp.ObjVal

            # set up the level problem
            lp = gp.Model()
            lp.setParam("OutputFlag",0)
            lp.setParam("Threads",1)
            pi_level = lp.addVars(prod_key_used, period_df.I, lb=-float('inf'), name="beta")
            pi_abs_level = lp.addVars(prod_key_used, period_df.I, lb=0.0, name="beta_abs")
            theta_lp = lp.addVar(name = "theta")
            lp.addConstrs(pi_abs_level[i,j,t] >= pi_level[i,j,t] for i,j in prod_key_used for t in period_df.I)
            lp.addConstrs(pi_abs_level[i,j,t] >= -pi_level[i,j,t] for i,j in prod_key_used for t in period_df.I)

            for i, j in prod_key_used:
                for t in period_df.I:
                    sp.remove(sub_fix_x[i, j, t])

            alpha = 0.0
            keep_iter = True
            
            while (keep_iter) and (iter_num <= 1000):
                iter_num += 1
                #print(iter_num," ", pi_obj," ",sub_obj)
                
                # generate the cut to characterize the Lagrangian function           
                obj = gp.quicksum(
                    holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
                    gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
                    gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
                    gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key_used for t in period_df.I) + \
                    gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t] - x_im_p[i,j,t] + x_im_m[i,j,t]) for i, j in prod_key_used for t in period_df.I)
                    # gp.quicksum(
                    #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

                sp.setObjective(obj, GRB.MINIMIZE)
                sp.update()
                sp.optimize()

                pi_obj = sp.ObjVal
                fList.append(np.sum(np.abs(list(pi_hat.values()))))
                if iter_num == 1:
                    best_dual = copy.deepcopy(pi_hat)
                elif sub_obj - pi_obj < np.min(GList):
                    best_dual = copy.deepcopy(pi_hat)
                GList.append(sub_obj - pi_obj)

                # if we have obtain a good solution, stop
                if sub_obj - pi_obj <= 0.01 * sub_obj:
                    keep_iter = False
                else:
                    x_grad_vals = {}
                    for i, j in prod_key_used:
                        for t in period_df.I:
                            x_grad_vals[i,j,t] = x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t].X  - x_im_p[i,j,t].X + x_im_m[i,j,t].X

                    # adding the cuts for the Lagrangian function
                    smcp.addConstr(sub_obj - pi_obj - gp.quicksum(x_grad_vals[i,j,t] * (pi[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key_used for t in period_df.I) <= 0)
                    smcp.update()
                    smcp.optimize()
                    f_star = smcp.ObjVal
                    lp.addConstr(theta_lp >= sub_obj - pi_obj - gp.quicksum(x_grad_vals[i,j,t] * (pi_level[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key_used for t in period_df.I))

                    # calculate f_*, h_i, Delta
                    hp = gp.Model()
                    hp.setParam("OutputFlag",0)
                    alpha_v = hp.addVar(lb = 0.0, ub = 1.0, name = "alpha")
                    h_theta = hp.addVar(name = "h_i")
                    hp.addConstrs(h_theta <= alpha_v * (fList[j] - f_star) + (1 - alpha_v) * (GList[j]) for j in range(len(GList)))

                    # calculate alpha
                    hp.addConstr(h_theta >= 0)
                    hp.setObjective(alpha_v, GRB.MAXIMIZE)
                    hp.update()
                    hp.optimize()
                    alpha_max = hp.ObjVal

                    hp.setObjective(alpha_v, GRB.MINIMIZE)
                    hp.update()
                    hp.optimize()
                    alpha_min = hp.ObjVal

                    if iter_num == 1:
                        alpha = 0.5 * (alpha_max + alpha_min)
                    else:
                        if ((alpha - alpha_min)/(alpha_max - alpha_min) <= 1 - 0.5 * level_mu) and ((alpha - alpha_min)/(alpha_max - alpha_min) >= 0.5 * level_mu):
                            alpha = 0.5 * (alpha_max + alpha_min)

                    # obtain the lb, ub, and level
                    lb_est = alpha * f_star
                    ub_est = np.min([alpha * fList[j] + (1 - alpha) * GList[j] for j in range(len(GList))])
                    level = level_lambda * ub_est + (1 - level_lambda) * lb_est
                    level_con = lp.addConstr(alpha * gp.quicksum(pi_abs_level[i,j,t] for i,j in prod_key_used for t in period_df.I) + (1 - alpha) * theta_lp <= level)
                    lp.setObjective(gp.quicksum((pi_level[i,j,t] - pi_hat[i,j,t]) ** 2 for i,j in prod_key_used for t in period_df.I))

                    # generate the next pi
                    lp.update()
                    lp.optimize()
                    for i,j in prod_key_used:
                        for t in period_df.I:
                            pi_hat[i,j,t] = pi_level[i,j,t].X
                    lp.remove(level_con)

            for i,j in prod_key_used:
                for t in period_df.I:
                    if abs(best_dual[i,j,t]) < 1e-6: 
                        pi_hat[i,j,t] = 0
                    else:
                        pi_hat[i,j,t] = best_dual[i,j,t]

        else:
            pi_hat = dual_coeff
        
        if lag_cuts:
            obj = gp.quicksum(
                holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
                gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
                gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
                gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key_used for t in period_df.I) + \
                gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t] - x_im_p[i,j,t] + x_im_m[i,j,t]) for i, j in prod_key_used for t in period_df.I)
            # gp.quicksum(
            #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

            sp.setObjective(obj, GRB.MINIMIZE)
            sp.update()
            sp.optimize()

        pi_obj = sp.ObjVal
        for i, j in prod_key_used:
            for t in period_df.I:
                pi_hat[i,j,t] *= lot_size[i,j]
        print(multiprocessing.current_process(), " ", k, " ", pi_obj, " ", sub_obj, " ", iter_num)
    else:
        cut_gen = False
        pi_hat = 0
        pi_obj = sub_obj
        print(multiprocessing.current_process(), " ", k, " ", sub_obj)

    return k, cut_gen, sub_obj, pi_obj, pi_hat, feas_gen, feas_cut_coeff, feas_value, sub_HC, sub_PC, sub_TC, sub_dev_cost

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
    lot_size = dict(zip(prod_key, df_prod.LS))
    max_prod = dict(zip(prod_key, df_prod.MaxProd))
    unit_cap = dict(zip(list(zip(df_unitCap.I, df_unitCap.J, df_unitCap.Ct)), df_unitCap.V))
    max_cap = dict(zip(list(zip(df_maxCap.Ct, df_maxCap.J, df_maxCap.Ti)), df_maxCap.V))

    global lead_time, real_demand, external_purchase, init_inv, holding_cost, penalty_cost, transit_cost, transit_time
    lead_time = dict(zip(prod_key, df_prod.LT))
    demand_key = list(zip(df_demand.I, df_demand.Ti))
    real_demand = dict(zip(demand_key, df_demand.V))
    external_purchase_key = list(zip(df_external.I, df_external.J, df_external.Ti))
    external_purchase = dict(zip(external_purchase_key, df_external.V))
    init_inv = dict(zip(list(zip(df_iniInv.I, df_iniInv.J)), df_iniInv.V))
    transit_time = dict(zip(df_transitTC.Tr, df_transitTC.V))

    global degree_list, consistuent_list, parent_list, composition_list, subsidiary_list, precursor_list
    degree_list, consistuent_list, parent_list, composition_list, subsidiary_list, precursor_list = analyze_bom()
    
    # readin cost
    holding_cost = dict(zip(df_cost.I, df_cost.HC))
    penalty_cost = dict(zip(df_cost.I, df_cost.PC))
    transit_cost = dict(zip(df_transitTC.Tr, df_transitTC.TC))
    '''
    for l in df_transit.index:
        # adjust transit cost in case transit cost is so small that items are moved between plants to avoid high inventory cost
        #if holding_cost[df_transit.I[l]] == 0:
        #    transit_cost[l] = 1.01 * 0.0001 * max(transit_time[l], 1)
        #else:
        transit_cost[l] = max(1.01 * holding_cost[df_transit.I[l]] * max(transit_time[l], 1), transit_cost[l])
    
    '''
    '''
    # adjust costs
    holding_cost = {}
    penalty_cost = {}
    transit_cost = {}
    for i in item_df.I:
        # adjust holding cost: raw material 0, degree 1 - 0.001, degree 2 - 0.002, ...
        holding_cost[i] = 0.001 * degree_list[i]

        # adjust penalty cost: holding cost * 1000
        penalty_cost[i] = 1000 * holding_cost[i]
    for l in df_transit.index:
        # adjust transit cost:
        #if holding_cost[df_transit.I[l]] == 0:
        #    transit_cost[l] = 1.01 * 0.0001 * max(transit_time[l], 1)
        #else:
        transit_cost[l] = 1.01 * holding_cost[df_transit.I[l]] * max(transit_time[l], 1)
    
    # adjust holding cost in case lots of item A convert into item B since holding cost of B is smaller than A (A and B are alternatives) 
    for jta in df_alt.index:
       if holding_cost[df_alt.I[jta]] > df_alt.V[jta] * holding_cost[df_alt.II[jta]]:
           holding_cost[df_alt.II[jta]] = holding_cost[df_alt.I[jta]] / df_alt.V[jta]
    '''

    # item clustering such that items with substitutution relationship are in the same cluster
    global alt_item_pair_list, decomp_units, bom_list
    alt_item_pair_list = list(zip(df_alt.I, df_alt.II))
    alt_item_pair_list = [tuple(sorted(tuple_)) for tuple_ in alt_item_pair_list]
    alt_item_pair_list = list(dict.fromkeys(alt_item_pair_list))
    
    # decomp_units = [list(alt_pair) for alt_pair in alt_item_pair_list]
    # for i in list(item_df.I):
    #     if i not in alt_items_list:
    #         decomp_units.append([i])

    bom_pair = df_bom[["I", "II"]].drop_duplicates()
    bom_list = list(zip(bom_pair.I, bom_pair.II))
    
    
    decomp_dict = {i: [i] for i in item_df.I[
        (item_df.I.isin(df_alt.I)) | (item_df.I.isin(df_alt.II)) | (item_df.I.isin(df_prod.I)) | (
            item_df.I.isin(df_bom.I)) | (item_df.I.isin(df_bom.II))]}
    for alt_pair in alt_item_pair_list:
        joined_list = list(set(decomp_dict[alt_pair[0]] + decomp_dict[alt_pair[1]]))
        for item in joined_list:
            decomp_dict[item] = joined_list
    decomp_units = []
    item_selected = {i: 0 for i in decomp_dict.keys()}
    for i in decomp_dict.keys():
        if item_selected[i] == 0:
            decomp_units.append(decomp_dict[i])
            for item in decomp_dict[i]:
                item_selected[item] = 1
    
    # fine_tune
    #decomp_units = [[33, 253, 272, 276], [28, 312, 315, 70, 71], [6, 82, 117, 122], [27, 151, 154, 157], [36, 285, 319, 321], [34, 322, 427, 430], [30, 165, 173, 174], [31, 328, 336, 341], [32, 344, 346, 352], [29, 175, 178, 180], [35, 355, 185, 193], [37, 146, 207, 236], [18, 382, 275, 284], [432, 177, 204, 287], [19, 435, 389, 289, 292], [13, 298, 299, 301], [26, 307, 310, 311], [11, 314, 316, 331], [8, 383, 403, 414], [9, 369, 373, 374], [10, 388, 418, 419], [12, 420, 422, 423], [7, 424, 425, 428], [14, 429, 443, 444], [15, 445, 446, 447], [17, 448, 449, 450], [44, 1, 3, 20], [16, 38, 40, 67], [50, 130, 144, 184], [56, 195, 203, 206], [84, 237, 242, 244], [160, 248, 277, 278, 281], [155, 283, 286, 290], [45, 291, 295, 296], [57, 297, 300, 317], [66, 323, 385, 395], [68, 416, 0, 4], [76, 5, 21, 22], [81, 25, 39, 42], [85, 43, 46, 58], [91, 60, 64, 69], [123, 72, 74, 75], [124, 88, 89, 92], [125, 93, 94, 95], [161, 96, 98, 99], [163, 100, 101, 102], [164, 104, 105, 108], [166, 109, 111, 112], [179, 113, 115], [183, 116, 118], [211, 120, 126], [212, 128, 129], [213, 134, 137], [214, 139, 143], [238, 148, 150], [240, 153, 159], [249, 168, 182], [250, 187, 188], [258, 189, 190], [262, 192, 196], [265, 198, 199], [267, 202, 209], [268, 210, 215], [269, 216, 217], [270, 218, 219], [273, 220, 221], [279, 222, 223], [399, 224, 225], [400, 226, 229], [401, 233, 266], [404, 280, 288], [410, 293, 302], [41, 304, 308], [51, 309, 313], [133, 318, 320], [167, 325, 327], [239, 329, 330], [241, 333, 334], [243, 335, 337], [245, 339, 342], [247, 343, 345], [251, 347, 350], [252, 351, 353], [254, 354, 356], [256, 361, 362], [261, 365, 368], [263, 370, 371], [274, 372, 375], [282, 376, 377], [305, 378, 379], [332, 386, 390], [402, 391, 392], [426, 393, 394], [433, 396, 397], [434, 398, 405], [439, 407, 408], [440, 409, 411], [441, 412, 415], [442, 417, 436], [61, 437, 438], [162], [172], [197], [246], [255], [257], [264], [271], [366], [367], [451], [23], [59], [62], [73], [80], [83], [106], [121], [127], [132], [135], [141], [145], [149], [156], [158], [169], [170], [181], [186], [194], [200], [201], [205], [208], [228], [230], [232], [234], [259], [260], [294], [303], [306, 326], [324], [338], [340], [348], [349], [357], [358], [359], [360], [363], [364], [380], [381], [384], [387], [406], [413], [421], [431], [2], [24], [47], [48], [49], [52], [53], [54], [55], [63], [65], [77], [78], [79], [86], [87], [90], [97], [103], [107], [110], [114], [119], [131], [136], [138], [140], [142], [147], [152], [171], [176], [191], [227], [231], [235]]
    # D1
    #decomp_units = [[435, 710, 740, 756], [33, 470, 834, 845, 851], [2029, 694, 733, 768], [471, 661, 729, 882], [6, 718, 764, 767], [3048, 726, 761, 798], [281, 27, 901, 960, 993], [5, 3103, 843, 976, 988], [258, 762, 941, 945], [75, 2119, 667, 674, 707], [852, 949, 1484, 1485, 958, 986], [759, 714, 1556, 717, 735], [805, 47, 743, 837, 967], [2180, 745, 774, 794], [2934, 998, 1021, 1026], [2431, 782, 848, 948], [29, 359, 703, 1038, 1042], [2406, 1046, 1047, 1056], [2395, 1058, 1077, 1080], [408, 1091, 1953, 1122, 2278, 1170], [410, 1187, 1197, 1200], [875, 1219, 1225, 1250], [3050, 1268, 1314, 1317], [26, 239, 1338, 1339, 1365], [1244, 1399, 1401, 2651, 1414], [1810, 1435, 1448, 1453], [1822, 1456, 1468, 1471], [2882, 1476, 1480, 1496], [2883, 1502, 1517, 1537], [3042, 1542, 1551, 1554], [1240, 1562, 1565, 1570], [1816, 1592, 1594, 2427, 1608], [2135, 1617, 1657, 1676], [1859, 1697, 2515, 1702, 1704], [2972, 108, 672, 847, 964], [2702, 103, 1708, 1711, 1745], [1280, 804, 1224, 1748], [613, 1769, 1784, 1785], [39, 1791, 1792, 1793], [43, 1800, 1809, 1823], [107, 2965, 1134, 1216, 1720], [602, 1838, 1843, 1849], [2182, 1852, 1854, 1866], [286, 1867, 1870, 1871], [511, 1880, 1895, 1902], [522, 1911, 1913, 1941], [589, 1951, 1978, 1985], [35, 1999, 2000, 2007], [503, 2016, 2020, 2027], [506, 2034, 2043, 2066], [88, 2459, 734, 839, 908], [117, 2096, 2108, 2122], [284, 2125, 2127, 2139], [3125, 2177, 2184, 2190], [116, 871, 2192, 2210], [67, 1830, 991, 1498, 2207, 1698], [1804, 1296, 2213, 2232], [2394, 2234, 2250, 2256], [1944, 2259, 2267, 2270], [2454, 1298, 1504, 1626], [1942, 2273, 2286, 2290], [81, 2291, 2297, 2304], [2398, 2309, 2312, 2318], [62, 2323, 2335, 2339], [580, 2351, 2369, 2377], [1028, 970, 1017, 2381], [1635, 2384, 2390, 2392], [1659, 2393, 2409, 2412], [2385, 2413, 2455, 2462], [2429, 2472, 2473, 2480], [2443, 2483, 2490, 2492], [2848, 2494, 2503, 2508], [3091, 2511, 2514, 2524], [3096, 2527, 2555, 2570], [3098, 2572, 2587, 2600], [3100, 2602, 2614, 2642], [914, 53, 2648, 2649, 2659], [66, 2661, 2665, 2670], [109, 2685, 2692, 2700], [1806, 2723, 2728, 2733], [83, 2740, 2748, 2752], [94, 2753, 2778, 2779], [1024, 2803, 2804, 2805], [2432, 2812, 2815, 2819], [2194, 2836, 2843, 2862], [2721, 2892, 2900, 2908], [2874, 1138, 1521, 2909], [89, 2460, 2925, 2937, 2941], [803, 46, 1065, 1429, 1171, 1419], [2093, 1223, 1778, 2049], [2710, 1510, 2484, 2960], [2962, 2980, 2994, 3000], [3049, 3005, 3017, 3025], [712, 2817, 3089, 3026, 3038], [957, 3061, 3065, 3077], [2089, 3097, 3108, 3111], [2110, 3118, 3120, 3145], [104, 2666, 3147, 3155], [825, 3169, 3173, 3174], [1529, 3197, 3204, 3211], [2138, 1101, 1103, 1548], [2732, 3214, 3231, 3232], [3198, 3236, 3237, 3239], [3208, 3250, 3251, 3259], [1789, 122, 1130, 126, 127], [65, 1691, 1007, 1053, 1106], [365, 31, 1174, 1477, 2159], [2124, 1247, 1741, 2578], [55, 2669, 131, 2229, 139], [1142, 1233, 1803, 145, 428], [95, 153, 154, 155], [2510, 157, 2885, 166, 167], [102, 169, 171, 3243, 174], [1864, 134, 138, 144], [2638, 176, 2608, 188, 194, 2906], [101, 195, 1620, 198, 212], [2635, 1391, 215, 216, 217], [49, 219, 220, 221], [57, 222, 224, 225], [60, 229, 231, 235], [855, 236, 244, 246], [1523, 249, 250, 253], [61, 752, 1675, 1820], [99, 255, 257, 259], [1553, 2170, 2652, 2951], [204, 23, 1811, 1847, 1876], [100, 260, 261, 263], [363, 30, 1615, 268, 278], [59, 287, 289, 290], [379, 292, 293, 295], [454, 296, 298, 300, 2375], [1285, 302, 303, 304], [1520, 312, 314, 316], [1699, 323, 324, 330], [233, 332, 333, 334], [398, 337, 341, 345], [1695, 347, 348, 352], [2722, 354, 1960, 355, 357], [2976, 360, 364, 369], [2978, 373, 381, 382], [1279, 383, 384, 385], [1343, 387, 388, 389], [2884, 390, 2171, 395, 396], [24, 210, 2877, 402, 412], [120, 3258, 1213, 2923, 3202], [1252, 413, 414, 416], [1167, 418, 422, 423], [121, 3261, 1692, 2275, 2376], [78, 426, 434, 440], [1294, 1006, 441, 443], [1305, 446, 448, 464], [7, 465, 468, 469], [41, 473, 475, 476], [172, 477, 479, 482], [1195, 483, 484, 489], [1293, 491, 492, 498], [2114, 499, 501, 502], [1638, 504, 505, 2810, 512], [0, 2933, 936, 1344, 2969], [1677, 514, 518, 523], [32, 524, 534, 539], [115, 2542, 540, 541], [1290, 555, 557, 558], [2493, 560, 561, 781, 562], [1, 11, 564, 566, 571], [521, 2280, 573, 576], [58, 1447, 2935, 548, 2533, 577], [1055, 578, 581, 583], [1126, 584, 587, 593], [2084, 594, 603, 607], [3144, 610, 618, 621], [2247, 622, 623, 625], [2382, 626, 627, 630], [2776, 631, 633, 636], [3107, 2373, 637, 640, 641], [124, 644, 1527, 648, 653], [189, 658, 659, 660], [1355, 2140, 666, 668, 669], [1372, 671, 673, 678], [3059, 1137, 679, 680, 685], [3138, 1516, 1598, 691, 692], [2179, 77, 697, 849, 693], [394, 185, 1685, 696, 699], [1153, 702, 705, 706], [1390, 708, 709, 711], [2208, 713, 1623, 715, 716], [2598, 723, 728, 731], [2725, 741, 744, 746], [3255, 751, 755, 757], [8, 2977, 3010, 763], [76, 2173, 765, 766, 770], [91, 1085, 2735, 772], [92, 319, 779, 786], [163, 788, 790, 797], [184, 2579, 800, 802, 811], [266, 191, 812, 813, 814], [1109, 817, 818, 824], [1425, 831, 835, 838], [1705, 2426, 842, 846, 853], [2534, 858, 862, 863], [44, 864, 865, 867], [308, 877, 881, 885], [327, 887, 889, 892], [456, 647, 894, 895], [1154, 896, 900, 905, 2553], [1193, 907, 910, 911], [1588, 913, 916, 918], [1821, 919, 921, 922], [2332, 930, 932, 935], [9, 2, 12, 857, 944, 952], [85, 956, 959, 961], [137, 963, 968, 971], [421, 972, 978, 983], [886, 984, 987, 992], [1069, 995, 997, 1000], [1403, 1001, 1002, 1005], [1494, 1010, 1013, 1015], [1557, 2379, 1020, 1023, 1025], [1790, 1033, 1035, 1036], [2255, 1041, 1044, 1049], [2449, 1054, 1059, 1063], [2461, 2361, 2979, 1064], [2500, 1068, 1070, 1071], [2823, 1072, 1073, 1076], [2886, 1079, 1081, 1083], [2957, 1087, 1088, 1090], [1123, 1095, 1096, 1097], [1165, 1100, 1108, 1111], [1178, 1115, 1118, 1129], [2548, 1253, 1131, 1133, 1135], [1281, 1136, 1143, 1147], [1825, 1149, 1155, 1156], [2048, 1159, 1162, 1168], [2517, 1172, 1183, 1188], [2668, 1189, 1190, 1191], [2689, 1198, 1201, 1203], [2878, 2794, 2820, 2814, 1204, 1205, 1207], [2813, 1208, 1214, 1215], [3166, 150, 1217, 1218, 1222], [1851, 175, 1230, 1232, 1236], [1092, 1239, 1254, 1256], [1121, 1257, 1258, 1260], [1547, 1262, 1270, 1273], [2188, 1274, 1276, 1277], [1932, 68, 1278, 1286, 2078, 1288], [271, 990, 1291, 1297], [624, 306, 1300, 1303, 1306], [344, 1307, 1308, 1309], [346, 1311, 1313, 1595, 1315], [567, 807, 276, 1316], [1514, 1318, 1322, 1325], [1734, 1330, 1332, 1333], [1836, 1335, 1336, 1341], [2404, 1342, 1348, 1351], [2605, 1353, 2045, 1354, 1359], [2736, 2738, 747, 1363, 1366], [3241, 1371, 1373, 1375], [444, 1376, 1377, 1378], [616, 1379, 1380, 1382], [783, 1392, 1393, 1396], [1062, 1397, 1398, 1400], [1972, 1350, 1404, 1405, 1406], [1535, 1407, 1408, 1416], [1887, 1417, 1418, 1420], [2214, 1422, 1424, 1426], [2254, 1427, 1430, 1433], [2499, 2715, 1434, 1436, 1437], [3207, 1438, 1450, 1455], [10, 3, 4, 860, 1458, 1462], [1848, 170, 1463, 1464, 1465], [280, 1469, 1473, 1474], [1179, 1478, 1482, 1483], [1180, 1488, 1490, 1493], [1265, 1495, 1497, 1501], [1452, 1507, 1508, 1513], [1491, 1522, 1525, 1526], [1744, 1528, 1531, 1533], [1807, 1538, 1541, 1543], [1986, 1558, 1559, 1563], [2046, 1564, 1568, 1569], [2126, 1571, 1572, 1989, 1573], [2420, 1574, 1578, 1579], [2423, 1586, 1587, 1590], [2636, 1593, 1596, 1600], [2645, 1601, 1602, 1603], [2899, 338, 777, 1604], [3066, 1606, 1610, 1611], [336, 158, 1613, 1616, 1618], [629, 1624, 1625, 1627], [1052, 1629, 1630, 1632], [1119, 1634, 1640, 1641], [3225, 1259, 1645, 1646, 1651], [1326, 2123, 1653, 1655, 1661], [1481, 1662, 1663, 1664], [1660, 1667, 1670, 1671], [3132, 2061, 1672, 1678, 1680], [2352, 1683, 1686, 1687], [2522, 1689, 1693, 1696], [2713, 1701, 1709, 1710], [3080, 1712, 1713, 1714], [3221, 1716, 1718, 1719], [190, 1725, 1727, 1728], [200, 1730, 1731, 1733], [496, 1737, 1739, 1742], [1946, 1243, 574, 1753, 1754, 1755], [688, 1758, 1759, 1760], [859, 1762, 1764, 1765], [934, 1766, 1768, 1774], [1057, 1780, 1781, 1786], [1206, 1794, 1797, 2149, 1799], [1487, 1801, 2716, 1812, 1813], [2333, 1814, 1815, 1818], [2372, 1819, 1826, 1828], [2674, 1829, 1834, 1839], [2984, 1840, 1842, 1845], [2996, 1857, 1860, 1862], [3037, 1865, 1873, 1878], [3057, 1879, 1881, 1884], [3078, 1885, 1889, 1891], [806, 135, 1893, 1894, 1896], [1267, 141, 1899, 1901, 1903], [207, 1906, 1909, 1912], [214, 1917, 1920, 1929], [349, 1931, 1933, 1934], [437, 1936, 1937, 1943], [563, 1945, 1947, 1949], [649, 2025, 1952, 1954, 1957], [1163, 1959, 1962, 1969], [1251, 1973, 1974, 1977], [1349, 1981, 1984, 1987], [1352, 1990, 1993, 1994], [1369, 1995, 1996, 1997], [1479, 1998, 2001, 2004], [1582, 2006, 2010, 2011], [1589, 2014, 2015, 2017], [1607, 2018, 2021, 2023], [1732, 1519, 2028, 2030], [1783, 2031, 2036, 2038], [1916, 2040, 2042, 2050], [1992, 2051, 2052, 2053], [2080, 2054, 2060, 2065], [2544, 2072, 2073, 2075], [2546, 2086, 2087, 2090], [2599, 2092, 2094, 2095], [2717, 2099, 2100, 2104], [2801, 2111, 2113, 2118], [2850, 2128, 2134, 2142], [2860, 2151, 2152, 2157], [2945, 2158, 2161, 2162], [3068, 2164, 2165, 2166], [3075, 2168, 2169, 2172], [3105, 2175, 2178, 2181], [3114, 2185, 2186, 2195], [3230, 2196, 2197, 2198], [160, 1743, 2199, 2203, 2204], [165, 2205, 2209, 2211], [180, 1381, 2212, 2216, 2217], [203, 2220, 2225, 2227], [218, 2228, 2235, 2238], [294, 2239, 2240, 2241, 2314, 2307], [460, 2242, 2243, 2249], [462, 2257, 2260, 2261], [532, 2263, 2264, 2266], [657, 2268, 2269, 2272], [719, 2277, 2289, 2293], [2221, 926, 2294, 2295, 2299], [943, 2300, 2301, 2303], [965, 2306, 2308, 2311], [1237, 2315, 2319, 2320], [1441, 2325, 2326, 2328], [1637, 3175, 2334, 2338, 2342], [1715, 2343, 2344, 2346], [1850, 2513, 2347, 2348, 2350], [1914, 2353, 2355, 2356], [1966, 2362, 2364, 2365], [2026, 2367, 2370, 2387], [2274, 2402, 2403, 2405], [2285, 2408, 2410, 2414], [2554, 2415, 2416, 2422], [2607, 2425, 2428, 2437], [2765, 2440, 2441, 2445], [2927, 2448, 2458, 2465], [2942, 2468, 2470, 2471], [3029, 2476, 2477, 2481], [3130, 2482, 2488, 2489], [3172, 2491, 2495, 2498], [161, 2251, 2504, 2507, 2528], [179, 2531, 2536, 2537], [182, 2538, 2540, 2541], [247, 2550, 2551, 2558], [307, 2560, 2562, 2566], [409, 2568, 2575, 2581], [415, 2583, 2584, 2585], [526, 2586, 2589, 2591], [544, 2594, 2597, 2606], [646, 2610, 2612, 2615], [664, 2616, 2617, 2619], [670, 2623, 2624, 2626], [676, 2627, 2631, 2634], [677, 2641, 2647, 2654], [690, 2656, 2657, 2658], [698, 2662, 2664, 2671], [700, 2676, 2678, 2679], [841, 722, 2681, 2684, 2693], [748, 2696, 2698, 2699], [753, 2701, 2703, 2707], [760, 2712, 2714], [769, 2720, 2724], [784, 2729, 2731], [792, 2734, 2739], [819, 2206, 2741, 2743], [844, 2745, 2746], [872, 2750, 2755], [893, 2756, 2758], [915, 749, 2759], [927, 2760, 2761], [928, 2762, 2766], [962, 2767, 2771], [985, 2772, 2773], [1012, 2774, 2775], [1022, 2777, 2781], [1030, 2785, 2786], [1075, 2792, 2796], [1141, 2798, 2800], [1176, 2807, 2818], [1248, 2821, 2822], [1266, 2828, 2832], [1346, 2834, 2838], [1472, 2849, 2851], [1489, 2857, 2859], [1575, 2863, 2865], [1584, 2866, 2870], [1614, 2871, 2873], [1658, 2518, 2875, 2879], [1707, 2880, 2881], [1723, 2887, 2888], [1729, 2889, 2890], [1761, 2894, 2895], [1773, 2896, 2898], [1782, 2902, 2910], [1827, 2911, 2913], [1833, 2915, 2918], [1861, 2928, 2931], [1924, 2938, 2940], [1930, 2519, 2943, 2949], [2611, 2012, 2952, 2953], [2077, 2963, 2964], [2156, 2966, 2967], [2200, 2968, 2971], [2292, 2974, 2975], [2305, 2982, 2985], [2345, 2991, 2992], [2378, 2995, 3001], [2391, 3200, 3002, 3008], [2418, 3012, 3013], [2452, 3015, 3018], [2525, 3027, 3028], [2632, 3031, 3033], [2690, 3035, 3040], [2718, 3041, 3043], [2770, 3054, 3058], [2914, 3060, 3063], [3034, 3070, 3074], [3046, 3084, 3086], [3064, 3088, 3092], [3139, 3095, 3101], [3160, 3102, 3104], [193, 3106, 3110], [242, 3112, 3115], [264, 3123, 3124], [272, 3128, 3129], [321, 3131, 3133], [329, 3134, 3136], [372, 3149, 3152], [374, 3153, 3162], [411, 3168, 3170], [500, 3171, 3177], [510, 3179, 3180], [517, 3185, 3186], [590, 3187, 3188], [650, 3189, 3195], [675, 3201, 3203], [695, 3205, 3206], [1539, 701, 3210, 3216], [725, 3217, 3218], [868, 3224, 3226], [878, 3228, 3229], [902, 3234, 3235], [906, 3240, 3244], [940, 3247, 3249], [1536, 1384, 981, 2747, 2749, 2846, 1310, 3253, 3256], [1004, 3260], [1029], [1144], [1151], [1199], [1324], [1340], [1356], [1411], [1431], [1440], [1475], [1486], [1499], [1540], [1580], [1636], [1643], [1650], [1669], [1717], [1750], [1775], [1872], [1890], [1927], [1955], [2035], [2081], [2085], [2129], [2143], [2183], [2287], [2417], [2697, 2478], [3176, 2485], [2561], [2673], [2744], [2754, 3036], [2829], [2830], [2853], [2869], [2907], [3047], [3127], [3165], [3178], [3223], [3227], [3248], [609, 397, 2191, 123, 1500], [140], [164], [168, 1546, 2944, 1061], [177], [196], [240], [241], [283], [328], [371], [436], [439], [442], [458], [461], [467], [481], [493, 3190], [588], [595], [635], [665], [724], [754], [758], [775], [789], [808], [809], [820], [840], [856], [869], [883], [888], [897], [904], [946], [955], [966], [974], [977], [980], [994], [1003], [1037], [1066], [1067], [1074], [1089], [1093], [1120], [1140], [1181], [1184], [1238], [1242], [1269], [1282], [1283], [1320], [1327], [1385], [1413], [1423], [1449], [1451], [1457], [1509], [1532], [1674, 1599], [1633], [1642], [1649], [1665], [1673], [1679], [1690], [1735], [2033, 1746], [1788], [1844], [1868], [1883], [1886], [1923], [1950], [1991], [2019], [2058], [2062], [2144], [2154], [2163], [2187], [2189], [2202], [2226], [2233], [2246], [2316], [2363], [2399], [2421], [2430], [2433], [2435], [2451], [2463], [2505], [2520], [2545], [2564], [2620], [2628], [2637], [2640], [2646], [3233, 2650], [2683, 2022], [2727], [2742], [2784], [2797], [2802], [2826], [2837], [2845], [2897], [2904], [2912], [2920], [2922], [2939], [2946], [3003], [3014], [3022], [3076], [3083], [3146], [3151], [3157], [3159], [3163], [3164], [3167], [3182], [3192], [3196], [3219], [3238], [147], [209], [227], [234, 2354], [251], [254], [270], [273, 2310], [315], [318], [320], [339], [350], [356], [361], [362], [1312, 367], [430], [447], [450], [485], [490], [536], [538], [551], [559], [565, 1446], [582], [591], [605], [606], [608], [642], [652], [655], [656], [663], [1824, 682], [1492, 727], [736], [742], [785], [832], [850], [854], [870], [931], [937], [938], [982], [989], [996], [1009], [1016], [1034], [1039], [1043], [1045], [1078], [1094], [1110], [1124], [1127], [1128], [1146], [1152, 3215], [1185], [1192], [1196], [1210], [1228], [1246], [1249], [1263], [1272], [1284], [1289], [1292], [1323], [1364], [1367], [1374], [1394], [1395], [1409], [1439], [1444], [1560, 2358, 1772, 1454], [1459], [1466], [1503], [1511], [1512], [1515], [1544], [1555], [1567], [1577], [1597], [1621], [1622], [1644], [1668], [1688], [1700], [1722], [1736], [1776], [1787], [1796], [1808, 2283], [1835], [1846], [1853], [1858], [1863], [1877], [1888], [1892], [1908], [1918], [1971], [1976], [2003], [2024], [2039], [2044], [2055], [2056], [2068], [2069], [2076], [2082], [2101], [2112], [2116], [2117], [2121], [2130], [2132], [2145], [2148], [2153], [2155], [2160], [2174], [2176], [2218], [2222], [2223], [2230], [2248], [2253], [2282], [2284], [2313], [2340], [2359], [2366], [2368], [2371], [2569, 2388], [2389], [2401], [2419], [2438], [2450], [2457], [2487], [2509], [2543], [2557], [2576], [2580], [2588], [2590], [2629], [2630], [2680], [2682], [2688], [2706], [2709], [2751], [2795], [2806], [2825], [2831], [2844], [2847], [2855], [2858], [2864], [2868], [2893], [2903], [2905], [2924], [2926], [2954], [3023], [3032], [3039], [3052], [3053], [3069], [3085], [3090], [3093], [3094], [3109], [3113], [3116], [3126], [3143], [3158], [3191], [3194], [2424, 1226, 13], [2496, 21], [106, 2071], [125], [129], [130], [133], [142], [2780, 143], [148], [149], [162], [173], [181], [187], [197], [1552, 1234, 199], [201], [211], [228], [237], [243], [245], [248], [269], [279], [282], [285], [288], [291], [309], [317], [322], [331], [342], [400], [417], [420], [431], [433], [445], [452], [453], [459], [474], [487], [508], [513], [528], [531], [533], [535], [545], [546], [550], [556], [572], [579], [585], [586], [592], [597], [598], [601], [611], [615], [617], [638], [1150, 639], [2998, 15], [16, 1345], [17, 2556], [19, 2981], [2331, 20], [128, 178], [1082, 132], [146, 1652], [1112, 151], [152, 2147], [515, 156], [192, 1740], [1505, 830], [1194, 2453], [1777, 2321], [2867, 2861]]
    # D4
    #decomp_units = [[42, 477, 303, 385, 454], [2054, 239, 438, 469], [12, 290, 360, 1270, 498], [13, 2273, 517, 551, 573], [1304, 226, 317, 326], [76, 1318, 331, 352, 546], [3067, 2059, 223, 249, 286], [10, 310, 373, 481], [11, 437, 499, 523], [928, 452, 580, 610], [2736, 617, 646, 651], [257, 289, 461, 470, 514], [116, 521, 2830, 585, 655], [859, 308, 682, 711], [2738, 722, 725, 730], [2201, 741, 743, 744], [1114, 745, 749, 750], [2464, 475, 528, 751], [1153, 449, 767, 772], [1564, 779, 788, 792], [1464, 800, 817, 829], [2037, 841, 852, 853], [1841, 854, 857, 858], [2033, 898, 946, 953], [990, 957, 980, 983], [1397, 999, 1000, 1002], [1303, 1013, 1016, 1041], [2199, 1054, 1055, 1058], [2202, 388, 1059, 1062], [636, 1066, 1068, 1070], [133, 1072, 2226, 1085, 1097], [566, 1103, 1105, 1123], [588, 1125, 1183, 1156, 2861, 2303, 1185], [1141, 1193, 1209, 1210], [2723, 538, 1091, 797, 941], [16, 2, 1018, 1212, 1213], [67, 1214, 1221, 1242], [1811, 1250, 1254, 1261], [1961, 1281, 1282, 1285], [894, 1288, 1295, 1296], [2198, 616, 781, 784], [1958, 1333, 1340, 1343], [505, 1359, 1382, 1383], [2436, 727, 1384, 1401], [71, 1402, 1403, 1406], [2865, 124, 1407, 1413, 1417], [507, 1419, 1430, 1451], [595, 548, 1463, 1468], [861, 1469, 1472, 1473], [1046, 1476, 1478, 1485], [1577, 1488, 1494, 1497], [1590, 1507, 1512, 1514], [1592, 1520, 1529, 2529, 1558, 1540, 1549], [1599, 1550, 1555, 1562], [111, 1563, 1573, 1586], [115, 1608, 1624, 1631], [512, 1635, 1640, 1654], [756, 1671, 1684, 1686], [1646, 1687, 1717, 1732], [1679, 1740, 1743, 1744], [1823, 1786, 1747, 1756, 1762], [2422, 1771, 1775, 1777], [2432, 1782, 1788, 1795], [2463, 1796, 1807, 1816], [2466, 1837, 1844, 1845], [2525, 1850, 1859, 1863], [3118, 1877, 1890, 1922], [3123, 1926, 1928, 1929], [1040, 1966, 2009, 2018], [2210, 2028, 2038, 2043], [2731, 2056, 2057, 2061], [2891, 1167, 1536, 2078], [1135, 2079, 2081, 2088], [2139, 2099, 2123, 3134, 2134], [2727, 794, 2140, 2141], [2980, 2146, 2151, 2157], [3068, 2158, 2169, 2187], [423, 2204, 2205, 2211], [506, 2212, 2231, 2234], [963, 2251, 2283, 2286], [2117, 2308, 2310, 2315], [3269, 2320, 2329, 2335], [3271, 2348, 2350, 2353], [1706, 87, 1184, 1248, 1450], [122, 2103, 2362, 2380], [460, 2387, 2389, 2405], [1545, 2412, 2428, 2430], [2758, 2440, 2456, 2461], [3241, 2471, 2483, 2506], [89, 2508, 2511, 2522], [400, 2524, 2531, 2532], [1576, 2535, 2546, 2561], [1793, 2562, 2594, 2596], [1878, 2600, 2605, 2608], [1934, 2613, 2615, 2636], [2747, 2653, 2655, 2664], [3091, 2666, 2669, 2670], [123, 985, 2764, 1354, 1656], [2829, 1665, 1794, 1898], [1447, 2677, 2694, 2698], [1815, 367, 1971, 2560], [49, 2717, 2726, 2730], [118, 2761, 2776, 2779], [381, 2780, 2781, 2791], [1674, 2793, 2796, 2818], [2543, 2824, 2825, 2832], [2788, 685, 1005, 2334], [33, 2838, 2840, 2841], [48, 2864, 2868, 2873], [102, 2880, 2881, 2885], [108, 2886, 2914, 3042, 2915], [112, 2924, 2925, 2927], [113, 2945, 2957, 2962], [229, 2965, 2967, 2975], [541, 2995, 2996, 3006], [593, 3011, 3015, 3019], [851, 3024, 3041, 3043], [1029, 901, 2052, 3072], [2189, 3075, 3076, 3117], [2445, 3121, 3129, 3148], [2491, 3163, 3169, 3171], [2493, 3172, 3174, 3175], [1394, 3180, 3194, 3200], [2418, 667, 675, 678], [248, 1600, 1603, 1607], [455, 1609, 1910, 1972], [3114, 1001, 2724, 3203], [703, 3206, 3219, 3226], [1575, 1695, 3231, 3249], [550, 347, 3252, 3258], [1813, 1291, 3265, 3266], [277, 3240, 2318, 3267, 3268], [2590, 698, 709, 3281], [99, 599, 1491, 1617], [2115, 1872, 3290, 141], [58, 147, 153, 155], [60, 163, 167, 174], [798, 176, 181, 184], [804, 197, 199, 209], [1802, 641, 1130, 211], [59, 214, 219, 220], [526, 224, 2944, 225, 232], [537, 247, 251, 252], [802, 261, 263, 268], [1578, 283, 291, 292], [2326, 2021, 293, 297], [3084, 300, 316, 322], [45, 329, 332, 335], [75, 337, 341, 353], [518, 355, 356, 357], [520, 366, 369, 382], [1195, 384, 386, 387], [1414, 391, 395, 397], [2159, 411, 414, 416], [2161, 417, 418, 420], [2192, 421, 435, 447], [298, 412, 457, 458], [228, 462, 468, 473], [51, 474, 476, 478], [1118, 1392, 480, 482], [3177, 483, 486, 489], [114, 491, 493, 503], [1676, 515, 519, 531], [1720, 536, 539, 545, 556], [2229, 552, 563, 564], [2419, 567, 570, 578], [2646, 590, 594, 597], [3132, 601, 603, 607], [227, 609, 611, 614], [2332, 161, 172, 191], [343, 622, 623, 625], [3147, 2599, 626, 628], [3152, 630, 656, 650, 652], [72, 2786, 255, 296], [1714, 654, 664, 669], [2085, 670, 679, 683], [101, 684, 689, 693], [117, 701, 704, 708], [2186, 712, 714, 717], [2538, 718, 721, 734], [3104, 735, 738, 748], [20, 5, 1789, 691, 1166, 746], [1692, 763, 766, 768], [2284, 774, 1773, 775, 777], [1, 9, 1048, 1693, 330], [14, 3057, 757, 3273, 783], [2193, 103, 801, 807, 816], [4, 15, 940, 1366, 2987], [299, 824, 825, 826], [463, 834, 843, 848], [1309, 849, 855, 860], [1647, 867, 870, 872], [1767, 1301, 879, 884], [1984, 1993, 888, 892, 896], [380, 899, 904, 905], [688, 911, 918, 919], [2817, 1395, 1388, 924, 929, 934], [1559, 938, 948, 956], [53, 2740, 966, 971], [73, 972, 973, 978], [125, 988, 994, 1851, 995], [127, 996, 1009, 1011], [256, 1012, 1015, 1020], [260, 1021, 1025, 1027], [676, 1157, 1028, 1030, 1038], [776, 1050, 1052, 1065], [1063, 1069, 1074, 1075], [1215, 485, 671, 1076], [3002, 1079, 1086, 1098], [3005, 1100, 1101, 1102], [3007, 1107, 1109, 1112], [3012, 1119, 1126, 1128], [3016, 1139, 1140, 1148], [3073, 1161, 1163, 1164], [3115, 279, 364, 365], [166, 1165, 1179, 1196], [193, 1211, 1216, 1220], [838, 1222, 1223, 1224], [1082, 1227, 1228, 1229], [1574, 1239, 1240, 1246], [2016, 1251, 1256, 1258], [2266, 1272, 1274, 1290], [2685, 1294, 1299, 1314], [2834, 1024, 1315, 1317], [2890, 1319, 1322, 2782, 1324], [18, 21, 7, 1805, 862, 1325], [36, 1326, 1327, 1328], [2304, 997, 1705, 1237, 565, 1814, 2741, 2909, 159, 1329, 1330, 1331], [266, 1332, 1349, 1357], [269, 1361, 1377], [345, 1381, 1391], [961, 1393, 1396], [3272, 2049, 986, 2257, 1398, 1399], [2160, 1373, 1420, 1421], [2259, 362, 960], [2679, 1422, 1423], [77, 1424, 1431], [105, 2302, 1434], [754, 212, 1669, 399, 1439, 1441], [805, 1443, 1445], [864, 1154, 1037, 1448], [889, 1453, 1456], [1418, 1457, 1467], [1510, 1839, 1475, 1477], [2374, 1480, 1481], [2462, 1482, 1483], [2569, 1486, 1490], [2813, 1495, 1501], [2833, 1505, 1506], [2844, 1508, 1509], [3056, 1089, 1262], [516, 1516, 1521], [2243, 1525, 1527], [3054, 1442, 1579], [17, 705, 2991], [602, 1532, 1539], [1302, 1541, 1544], [1803, 246, 285], [2073, 1547, 1548], [2441, 1554, 1581], [2903, 1585, 1587], [2905, 1588, 1589], [350, 1591, 1594], [668, 1596, 1598], [0, 3, 1604, 3144], [666, 1876, 1605, 1610], [1143, 1614, 1615], [1364, 1618, 1619], [1500, 1621, 1627], [2481, 1629, 1638], [2550, 1639, 1644], [3208, 1645, 1649], [8, 19, 3238, 865], [937, 1652, 1658], [2695, 1659, 1662], [2901, 1843, 935, 1663], [2993, 1664, 1666], [3274, 1670, 1677], [2120, 1194, 1651, 1653, 890, 1533, 30, 1023, 1678, 1683], [98, 1915, 1917], [287, 1689, 1690], [451, 1697, 1699], [1092, 1700, 1703], [1268, 1708, 1718], [2007, 1721, 1725], [2106, 1728, 1730], [2150, 1737, 1741], [3009, 1746, 1749], [699, 1751, 1752], [799, 1754, 1757], [1283, 1760, 1761], [1345, 1763, 1766], [2645, 1367, 1768, 1774], [1992, 1370, 1776, 1781], [1842, 1783, 1799], [2427, 1801, 1808], [336, 142, 1809, 1817], [657, 1820, 1822], [921, 661, 1825, 1830], [706, 1832, 1849], [930, 1854, 1857], [1493, 1858, 1861], [1675, 1867, 1870], [2104, 2069, 1873, 1874], [2102, 1879, 1887], [2307, 1888, 1889], [2624, 1892, 1894], [2762, 752, 821], [2894, 1895, 1896], [3139, 1899, 1901], [3164, 1902, 1908], [3179, 1909, 1911], [6, 1003, 1912], [348, 1913, 1916], [428, 1918, 1923], [471, 1925, 1930], [2216, 1178, 716, 1933, 1936], [914, 1941, 1942], [915, 1176, 791, 1945], [998, 1946, 1948], [1546, 1950, 1953], [2454, 1957, 1962], [2579, 1964, 1968], [2809, 1974, 1979], [2898, 2839, 1981, 1987], [2849, 3283, 1988, 1990], [2923, 1994, 1995], [3248, 2004, 2005], [264, 2006, 2008], [349, 2010, 2011], [434, 2013, 2014], [764, 2020, 2023], [835, 2025, 2027], [900, 2031, 2032, 2552], [1095, 2034, 3230, 2039], [1170, 2040, 2042], [1231, 2044, 2045], [1602, 2053, 2064], [1750, 2065, 2066], [1853, 2067, 2071], [2035, 2075, 2076], [2144, 2077, 2090], [2235, 2094, 2095], [2404, 2098, 2108], [2487, 2344, 2113, 2341, 2114], [2500, 2116, 2121], [2558, 2125, 2131], [2593, 2137, 2170], [2929, 2177, 2180], [3017, 3030, 2182, 2185], [3213, 2188, 2190], [201, 2195, 2200], [379, 2209, 2214], [396, 27, 1701, 2217], [1504, 415, 2219, 2224], [584, 280, 2225], [639, 2228, 2230], [649, 2232, 2237], [869, 2238, 2239], [931, 2242, 2245], [1084, 2247, 2252], [1152, 2254, 2260], [1219, 2263, 2265], [1267, 2267, 2268], [1433, 2270, 2271], [2712, 1959, 2272, 2275], [2001, 2278, 2280], [2299, 2289, 2291], [2769, 2296, 2297], [3071, 2298, 2300], [165, 1286, 2306, 2313], [210, 2316, 2324], [319, 2325, 2333], [346, 2336, 2338], [525, 2342, 2349], [723, 2356, 2358], [736, 850, 3023, 728, 2359, 2367], [760, 2371, 2379], [1042, 2381, 2390], [1081, 2363, 2391, 2392], [1108, 2398, 2399], [1369, 2400, 2401], [1371, 2403, 2408], [1389, 2409, 2425], [1455, 2426, 2429], [1831, 2433, 2434], [1965, 2437, 2446], [1973, 2449, 2452], [2107, 2920, 2458, 2460], [2181, 2470, 2479], [2277, 2482, 2496], [2282, 2516, 2519], [2312, 2475, 2520, 2523], [2572, 2526, 2537], [2739, 2540, 2541], [2750, 2544, 2555], [2783, 2557, 2565], [2794, 2567, 2573], [2822, 2574, 2575], [3178, 3035, 2578, 2589], [3088, 2592, 2603], [3126, 2606, 2696, 2607], [3131, 2609, 2612], [3201, 2621, 2623], [26, 959, 2628, 2629], [189, 2631, 2634], [231, 2637, 2638], [426, 2641, 2658], [2949, 3149, 2255, 2354, 637, 2659, 2660], [836, 2661, 2665], [845, 2676, 2680], [974, 1943, 2684, 2687], [1110, 2690, 2691], [1134, 2693, 2697], [1181, 2701, 2704], [1192, 2705, 2710], [1217, 2713, 2716], [1287, 2719, 2721], [1347, 2722, 2732], [1355, 1350, 2734, 2744], [1622, 2745, 2746], [1636, 2749, 2756], [1672, 2319, 2759, 2765], [1691, 2767, 2770], [1733, 2771, 2778], [1828, 2785, 2797], [1852, 2800, 2816], [1855, 2819, 2820], [2101, 2826, 2827], [2175, 2828, 2837], [2207, 2843, 2845], [2215, 2848, 2856], [2240, 2857, 2858], [2382, 2870, 2874], [2513, 3212, 2875, 2887], [2527, 2888, 2892], [2733, 2893, 2895], [2876, 2896, 2897], [2922, 2906, 2910], [3047, 2911, 2916], [3052, 2917, 2930], [3053, 2933, 2934], [3106, 2939, 2941], [3130, 2947, 2950], [3170, 2958, 2961], [3256, 2970, 2971], [24, 2619, 2973, 2974], [137, 2978, 2979], [216, 2981, 2982], [241, 2983, 2984], [258, 2985, 2986], [444, 2997, 2998], [557, 3027, 3029], [576, 1963, 3032, 3045], [577, 3049, 3051], [631, 3058, 3060], [660, 3062, 3074], [686, 3079, 3080], [873, 3082, 3086], [1124, 3089, 3092], [1150, 3093, 3096], [1162, 3102, 3110], [1266, 3111, 3112], [1273, 3113, 3124], [1276, 3127, 3128], [1305, 3140, 3141], [1436, 3151, 3156], [1503, 3158, 3160], [1543, 2518, 2735, 3162, 3165], [1595, 2964, 3185, 3188], [1711, 3190, 3197], [1792, 3205, 3215], [1862, 3220, 3224], [2152, 3229, 3232], [2311, 3235, 3237], [2585, 3247, 3253], [2647, 3254, 3260], [2777, 3055, 3263, 3270], [2803, 581, 1932], [2852, 3277, 3282], [2853, 3284, 3289], [3003], [3038], [3143], [3146], [3198], [204], [207], [208], [325], [466], [479], [561], [787, 587], [591], [674], [681], [700], [969, 707], [731, 1755], [758], [810], [830], [875], [908], [989], [1045], [1067], [1104], [1201], [1205], [1208], [1234, 2514], [1279], [1293], [1368], [1432], [1489], [1528], [1567], [1806], [1836], [1868], [2080], [2149], [2176], [2196], [2222], [2321], [2402], [2415], [2447], [2453], [2505, 2718], [2517], [2533], [2554], [2743], [2883], [2899], [3065], [3066], [3078], [3098], [3196], [3264], [29, 2918], [136], [2281, 173], [274], [311], [320], [324], [374], [425], [443], [533], [543], [605], [606], [648], [659], [726], [742, 2047], [759], [765], [814], [916, 753], [965], [970], [1019], [1077], [1144], [1186], [1253], [1265], [1374], [1408], [1496], [2051, 1502], [1515], [1553], [1583], [1593], [1606], [1611], [1696], [1710], [1739, 1530], [1742], [1790], [1833], [1897], [1927], [1931], [2024], [2087], [2122], [3004, 2174], [2183], [2246], [2276], [2288], [2328], [2501], [2563], [2602], [2752], [2850], [2872], [2882], [2943], [2959], [2972], [3034], [3081], [3099], [3159], [3210], [3211], [3216], [3245], [135], [2416, 1035, 148, 2421], [178], [195], [198], [218], [272], [288], [2753, 294], [359], [371], [413], [419], [992, 1410, 422, 2867, 2773], [427], [453], [456], [464], [467], [540], [574], [589], [621], [629], [662], [677], [696], [702], [729], [773], [780], [809], [822], [823], [880], [882], [897], [910], [976], [1060], [1132], [1133], [1136], [1174], [1203], [1204], [1218], [1257], [1308], [1338], [1346], [1356], [1385, 1846], [1425], [1427], [1428], [1435], [1446], [1460], [1471], [1551], [1626], [1650], [1722], [1734], [1810], [1847], [1886], [1893], [1900], [1907], [1960, 2000], [2048], [2091], [2165], [2203], [2244], [3120, 2314], [2317], [2340], [2346], [2375], [2388], [3244, 2396], [2455], [2472], [2485], [2488, 3166], [2548], [2556], [2601], [2620], [2627], [2630], [2650], [2675], [2699], [2707], [2835], [2866], [2907], [2937], [2940], [2952], [3021], [3059], [3083], [3101], [3125], [3136], [3138], [3154], [3184], [3187], [3192], [3199], [3225], [3227], [3275], [144], [185], [2393, 237], [243], [259], [262], [295], [305, 2410, 2022], [315], [333], [338], [354], [376], [383], [398], [410], [430], [446], [459], [492], [496], [501], [502], [508], [532], [554], [559], [562], [568], [1466, 582], [596], [604], [618], [634], [653, 1182], [673], [692, 1829], [710], [720], [732], [739], [747], [761, 3261], [769], [1560, 770, 1557], [812], [820], [1570, 828], [866], [874], [891], [936], [944], [949], [951], [967], [982], [1004], [1007], [1010], [1031], [1036], [1039], [1049], [1080, 1997], [1083], [1113], [1116], [1122], [1155], [1159], [1169], [1171], [1177], [1206], [1241], [1245], [1247], [1269], [1278], [1307], [1313], [1334], [1342], [1363], [1386], [1438], [1454], [1458], [1459], [1498], [1499], [1518], [1519], [1523], [1524], [1534], [1552], [1584], [1970, 1620], [1628], [2656, 1637], [1641], [1657], [1660], [1667], [1685], [1719], [1726], [1738], [1778], [1784], [1785], [1791], [1827], [1860], [1871], [1937], [1939], [1952], [1998], [2012], [2017], [2046], [2050], [2062], [2109], [2111], [2112], [2127], [2133], [2142], [2153], [2172, 2357], [2173], [2191], [2194], [2206], [2218], [2221], [2249], [2250], [2258], [2262], [2285], [2290], [2309], [2337], [2343], [2383], [2397], [2420], [2451], [2467], [2473], [2474], [2476], [2512], [2542], [2566], [2580], [2582], [2639], [2662], [2671], [2673], [2748], [2754], [2766], [2768], [2772], [2790], [2807], [2814], [2846], [2860], [2879], [2902], [2912], [2935], [2953], [2954], [2966], [3031], [3033], [3036], [3039], [3046], [3087], [3109], [3119], [3157], [3176], [3189], [3193], [3222], [3228], [3234], [3259], [3287], [179, 1173, 1078, 23, 22], [2904, 139, 2551], [143], [146], [152], [156], [157], [160], [164], [168], [169], [170], [175], [180], [182], [183], [186], [187], [188], [190], [192], [194], [217], [222], [25, 433], [138, 3276], [1634, 140], [529, 150], [162, 1365], [171, 2261], [2681, 177, 1145], [196, 1111], [1989, 215], [273, 3246], [328, 1920], [344, 403], [1940, 390], [665, 510], [1709, 806], [2509, 831], [881, 964], [1190, 887], [2368, 926], [2179, 1044], [2842, 1172], [1648, 1316], [1341, 3142], [2227, 1511], [2345, 1642], [1819, 2799], [1866, 2932], [2041, 3100, 2015], [2795, 2086], [3251, 2093], [2683, 2279], [2490, 2948], [2528, 3018], [2536, 2539], [2884, 2877]]
    # D0
    #decomp_units = [[3176, 2200, 381, 2798], [4702, 2857, 3146, 4157], [2201, 2612, 2910, 5239], [4485, 2019, 2730, 3524], [5259, 2058, 3723, 4052], [5325, 507, 1409, 2706], [3834, 3510, 1265, 2870], [4116, 2108, 2393, 2401], [3862, 3677, 4308, 5403], [4666, 3114, 107, 282], [3986, 2583, 3231, 3389], [4441, 2446, 5426, 5464], [4491, 5518, 806, 1446], [4799, 4047, 4305, 4477], [4117, 4736, 4988, 5139], [4657, 4191, 4382, 416], [2843, 3046, 3671, 5142], [4417, 5230, 5348, 5471], [4569, 5102, 122, 158], [3629, 262, 368, 755], [4205, 892, 943, 962], [4129, 3050, 385, 667], [4772, 716, 2146, 3048], [5273, 810, 2866, 3088], [3631, 1986, 966, 978], [4671, 599, 1174, 1175], [4675, 1194, 1400, 1503], [4974, 2610, 2753, 3324], [3559, 3760, 5012, 1639], [2533, 2816, 4463, 4828], [2733, 4786, 650, 1759], [3164, 2105, 4278, 4721], [5369, 2633, 2655, 4385], [3184, 2718, 4740, 123], [3750, 451, 779, 1213], [4307, 2265, 801, 2780], [2842, 3391, 1430, 2431], [1693, 5068, 1755, 1802], [4359, 3512, 2070, 3252], [4652, 1620, 1829, 1954], [4557, 4291, 816, 1379], [1567, 1968, 2055, 2107], [3903, 2156, 2168, 4135, 2179], [4479, 2987, 1044, 1834], [4972, 1637, 1885, 1992], [4416, 2923, 4156, 4165], [3407, 1681, 2002, 2182], [5329, 3902, 4641, 3890], [4964, 274, 2218, 2231], [5261, 289, 2336, 2380], [3408, 2555, 2556, 2558], [4271, 2568, 2585, 2593], [4405, 2618, 2629, 2642], [4492, 3221, 1410, 2644], [4104, 374, 1354, 5007], [932, 5144, 649, 718], [5357, 1040, 1303, 2141], [379, 2679, 2685, 2743], [3419, 4544, 2764, 2766, 2818], [4185, 2077, 2183, 330], [4358, 504, 1919, 2184], [5128, 2834, 2893, 2948], [4588, 5289, 2219, 2949], [4294, 1671, 1676, 3039], [4125, 5248, 2985, 3019], [4853, 3026, 3037, 3047], [4182, 1031, 3096, 3193], [3163, 3717, 58, 1933], [3241, 3913, 370, 1334], [4139, 818, 2845, 3201], [4576, 2038, 1696, 2181], [3882, 3202, 3208, 3322], [901, 3392, 3522, 3533], [4761, 5436, 763, 1095], [4306, 1266, 3297, 4471], [225, 1535, 1707, 5137], [4149, 3679, 3683, 3714], [4420, 3724, 3744, 3756], [4656, 3514, 84, 1830], [1715, 1867, 3771, 3791], [2347, 3904, 3981, 4018], [3897, 4043, 4079, 4094], [3157, 2150, 4095, 4111], [4015, 4113, 4235, 4252], [1566, 2873, 864, 2907], [137, 3805, 4498, 1442, 4302], [1714, 2886, 4311, 4312], [4184, 4331, 4418, 4468], [4206, 3475, 373, 512], [4568, 1609, 4258, 4515], [4593, 4539, 4691, 4755], [5352, 1083, 2582, 4795], [2257, 4796, 4916, 4925], [4665, 3811, 3844, 4582, 4980, 5016, 5032], [3394, 3347, 1329, 2814], [4150, 1770, 2994, 3511], [2332, 1536, 2567, 2143], [3938, 2672, 5019, 5048], [4177, 682, 3549, 5058], [4186, 3417, 5097, 5115], [4567, 5190, 5141, 5176], [1009, 1861, 2577, 1183], [3985, 5177, 5181, 5202], [3922, 3431, 3731, 4167], [982, 5089, 3427], [5193, 344, 3200], [3162, 277, 5278], [4075, 1253, 3638], [3317, 4089, 3570], [952, 5280, 5321], [3960, 3232, 5377], [4134, 5418, 5438], [4422, 5440, 5463], [4522, 5469, 5470], [3185, 5346, 2412], [2536, 5495, 5505], [5162, 2030, 3674], [4524, 5516, 13], [2544, 2229, 2307], [2696, 2943, 524], [244, 3545, 915], [3605, 2739, 3642], [2360, 2270, 2859], [4152, 50, 60], [4661, 51, 1159], [1798, 4874, 3595, 1996, 2494, 62, 64], [4074, 762, 82], [3406, 85, 157], [4877, 979, 57], [4781, 2119, 3900], [1988, 177, 200], [4295, 3438, 266], [4362, 4238, 4859], [4855, 4343, 279], [5330, 4553, 1429], [2023, 2251, 285], [2835, 2148, 294], [3204, 322, 324], [3957, 326, 333], [4005, 336, 365], [4565, 366, 371], [4982, 372, 384], [3191, 394, 395], [4800, 3703, 397, 412], [3779, 5514, 420], [3914, 1667, 429], [4172, 351, 3837], [4189, 434, 445], [4442, 452, 468], [4443, 3259, 469], [4444, 499, 500], [4917, 3421, 546], [1148, 5515, 584], [2205, 609, 636], [3948, 462, 638], [3959, 662, 663], [3989, 664, 700], [4374, 712, 749], [4507, 170, 751], [4562, 752, 807], [4919, 826, 835], [4976, 849, 853], [4984, 1460, 857], [5159, 866, 890], [5320, 898, 910], [136, 919, 1003], [219, 838, 1079], [1807, 1087, 1097], [2537, 1099, 1109], [2545, 680, 4138], [3242, 1114, 1115], [3287, 1121, 1130], [3316, 1140, 1195], [3820, 1207, 1209], [4363, 1239, 1247], [4474, 1259, 1280], [4678, 1302, 1336], [4936, 1340, 1343], [4989, 355, 931], [5158, 1218, 1347], [5319, 1349, 1360], [302, 685, 2071], [456, 4412, 1364], [950, 47, 1365], [1750, 1117, 1272], [2334, 1368, 1378], [3240, 1388, 1405], [3399, 1277, 1411], [4377, 2591, 1827], [4489, 1415, 1426], [4573, 1434, 1450], [5360, 1470, 1487], [5537, 2252, 1529], [272, 1537, 1540], [1048, 1542, 1559], [1804, 1598, 1599], [3289, 1625, 1626], [3314, 1461, 1632], [3870, 1638, 1650], [3941, 1675, 1677], [3945, 1685, 1688], [3991, 1698, 1706], [4143, 1710, 1720], [4353, 1728, 1786], [4677, 1791, 1824], [4701, 1848, 1872], [4910, 1904, 1915], [4991, 1925, 1940], [5272, 1971, 1975], [5358, 1979, 2001], [953, 1634, 2007], [1805, 2014, 2016], [2203, 2026, 2036], [2204, 2045, 2046], [3238, 2047, 2049], [3285, 2050, 2063], [3558, 2082, 2084], [3561, 4086, 2093], [3562, 2099, 2144], [3623, 2154, 2155], [3814, 2174, 2177], [3869, 2180, 2186], [3919, 2213, 2227], [4351, 2240, 2246], [4423, 2284, 2285], [4433, 2295, 2298], [4435, 2299, 2312], [4487, 2317, 2324], [4577, 2327, 2341], [4903, 2343, 2354], [4906, 2376, 2386], [4990, 2395, 2397], [5157, 2404, 2423], [5313, 2428, 2433], [5323, 2449, 2482], [5328, 2484, 2487], [5356, 2543, 2551], [132, 2570, 2574], [214, 2575, 2580], [2123, 2626, 2635], [3395, 2640, 2651], [3557, 2654, 2689], [3624, 2709, 2720], [3962, 2728, 2729], [4371, 2741, 2742], [4406, 2744, 2752], [4572, 2773, 2779], [4594, 2784, 2794], [4779, 1422, 2801], [5205, 2808, 2815], [5268, 2871, 2874], [5269, 2884, 2889], [5322, 2899, 2909], [5324, 2911, 2916], [3803, 1875, 2917], [3963, 2925, 2932], [4770, 2937, 2938], [5270, 2970, 2983], [218, 2750, 2990], [455, 2991, 2993], [2122, 3010, 3013], [2534, 3028, 3049], [2540, 3059, 3064], [3815, 3095, 3100], [3921, 3108, 3118], [3946, 3144, 3161], [3966, 3180, 3198], [3990, 3209, 3214], [4000, 3215, 3219], [4131, 3223, 3225], [4355, 1279, 3245], [4373, 3255, 3266], [4421, 3292, 3294], [4493, 3296, 3330], [4579, 3335, 3342], [4583, 3343, 3390], [4782, 3429, 3432], [5361, 3453, 3485], [5362, 3495, 3497], [135, 3498, 3502], [216, 3503, 3529], [458, 3536, 3537], [1125, 3598, 3634], [1149, 3640, 3669], [1658, 3690, 3699], [1959, 3702, 3713], [2504, 3719, 3728], [2933, 3729, 3733], [3137, 3738, 3754], [3872, 3769, 3775], [3874, 3877, 3927], [3876, 3969, 3972], [3920, 4759, 3975, 4026], [3949, 4042, 4050], [4354, 4053, 4061], [4388, 4146, 4158], [4425, 4168, 4198], [4486, 4199, 4207], [4520, 4215, 4221], [4523, 4223, 4236], [4714, 4245, 4246], [4857, 4248, 4264], [4887, 4267, 4313], [4922, 4322, 4325], [4927, 4330, 4340], [4975, 4384, 4387], [5036, 4396, 4402], [5536, 4036, 4426], [1806, 4447, 4455], [3318, 4457, 4459], [4142, 4473, 4505], [4364, 4518, 4532], [4663, 4538, 4599], [4683, 4629, 4648], [4684, 4694, 4705], [5331, 4712, 4719], [5399, 4722, 4726], [5538, 4741, 4742, 4790], [3780, 106, 205], [4372, 1258, 2592], [3964, 4244, 4821], [3175, 5459, 4832], [1146, 4846, 4847], [2022, 1774, 754], [3942, 4872, 4943], [4494, 4960, 4994], [1747, 80, 2476, 5017], [3943, 5018, 5026], [4188, 1483, 4098], [4578, 5345, 5028], [4893, 5030, 5040], [3843, 5043, 5060], [3288, 3783, 5078, 5098], [2538, 3947, 5105, 5106], [4194, 5124, 5136], [2121, 1012, 2906], [4173, 5409, 1891], [4560, 5140, 5143], [134, 5182, 5183], [3782, 5184, 5228], [5160, 5229, 5262], [3621, 3737, 1731], [4170, 4268, 5120], [3956, 5264, 5283], [3398, 5333, 5415], [4434, 5416, 5442], [4580, 5474, 5481], [4130, 5197, 5486], [3239, 5493, 5499], [4854, 5502, 5504], [4902, 5521, 5531], [138, 1, 9], [4133, 10, 11], [4581, 20, 40], [1050, 492, 3563], [3988, 41, 46], [4132, 48, 53], [4920, 59, 65], [3992, 68, 71], [5398, 72, 73], [301, 3348, 5080], [3987, 3302, 75], [4973, 77, 78], [951, 86, 95], [3625, 97, 103], [4905, 104, 110], [4561, 112, 115], [4521, 868, 125, 126], [2535, 128, 129], [3320, 5099, 1046], [3781, 4911, 131, 148], [3784, 151, 156], [4681, 190, 207], [5271, 211, 223], [139, 224, 241], [457, 265, 268], [3286, 270, 273], [3397, 278, 281], [3819, 296, 564, 299], [4360, 4376, 307, 317], [4481, 319, 320], [3804, 328, 329], [3944, 332, 337], [4680, 4465, 4448], [2207, 2388, 340], [3785, 343, 345], [4574, 352, 353], [3965, 354, 359], [3827, 360, 423, 377, 2236], [2206, 398, 399], [3244, 400, 401], [1611, 2850, 2892], [3190, 267, 1678], [5209, 1094, 2426], [2539, 1764, 406, 418], [3418, 3739, 422], [3319, 424, 426], [3958, 428, 431], [4436, 432, 433], [4679, 437, 439], [4140, 440, 441], [4488, 3152, 1801, 2849], [4564, 442, 448], [3786, 449, 454], [3821, 461, 470], [4006, 473, 478], [4983, 483, 485], [2565, 487, 489], [1019, 490, 503], [5125, 3085, 3474], [4819, 3105, 3142], [1868, 508, 510], [3867, 3573, 4037], [4478, 481, 514], [3186, 525, 529], [91, 531, 533], [5355, 150, 2029], [4347, 2377, 2847], [914, 535, 536], [5492, 538, 543], [5420, 183, 550, 553], [3005, 556, 560], [1465, 561, 562], [2455, 1255, 3016], [2519, 563, 568], [2836, 3017, 569], [3004, 3450, 573], [1269, 575, 580], [3448, 583, 593], [4732, 594, 601], [2282, 604, 606], [789, 2074, 2747], [2461, 1282, 607], [4122, 3323, 5027], [3310, 2740, 637], [4654, 1614, 641], [5215, 643, 657], [26, 668, 679], [254, 625, 688], [733, 116, 690], [734, 693, 695], [871, 705, 706], [895, 714, 729], [1734, 175, 737], [1783, 5133, 740], [2735, 586, 744], [3325, 747, 757], [3329, 5511, 758], [3617, 766, 775], [3796, 146, 5275, 776], [3798, 794, 799], [3846, 800, 804], [3999, 805, 808], [27, 815, 821], [184, 822, 823], [1270, 824, 827], [1455, 834, 837], [1587, 841, 846], [1984, 847, 848], [2958, 856, 858], [3309, 861, 862], [3326, 863, 875], [3366, 876, 880], [3369, 881, 882], [3370, 883, 887], [3446, 897, 899], [3818, 497, 900], [3847, 908, 909], [4842, 911, 917], [4867, 923, 927], [5443, 786, 1573, 928], [5525, 929, 934], [4, 935, 936], [29, 938, 942], [30, 944, 946], [185, 947, 955], [186, 958, 960], [222, 963, 964], [257, 1383, 965], [258, 968, 971], [363, 974, 976], [790, 977, 984], [894, 985, 993], [1271, 997, 999], [1310, 1001, 1002], [1551, 1007, 1016], [1553, 1024, 1026], [1763, 1027, 1028], [1901, 1029, 1035], [2085, 1042, 1043], [2088, 1045, 1051], [2214, 1055, 1056], [2349, 1068, 1072], [2353, 1338, 1077], [2520, 1201, 1080], [2527, 1084, 1092], [2659, 1105, 1112], [2688, 1113, 1116], [2756, 1118, 1124], [2957, 1136, 1142], [3002, 1152, 1154], [3139, 1160, 1162], [3165, 1163, 1165], [3311, 1171, 1177], [3371, 1178, 1186], [3447, 1189, 1192], [3608, 1208, 1210], [3615, 1215, 1219], [3616, 1223, 1224], [3823, 1225, 1227], [4102, 1238, 1243], [4280, 1252, 1256], [4636, 145, 5022, 1260], [4840, 1275, 1276], [4843, 1202, 1281], [4880, 1286, 1287], [4881, 1290, 1304], [4883, 1308, 1312], [4928, 1313, 1322], [5213, 1323, 1330], [5216, 1339, 1344], [5318, 1346, 1352], [5423, 1355, 1359], [5424, 1361, 1369], [5, 1370, 1376], [255, 1377, 1397], [613, 1404, 1423], [614, 1425, 1440], [1311, 1441, 1447], [1467, 1451, 1453], [1550, 1457, 1459], [2522, 1475, 1477], [2660, 1482, 1484], [2956, 1485, 1491], [3119, 1495, 1496], [3120, 1498, 1499], [3177, 1505, 1512], [3609, 1513, 1514], [3800, 1518, 1519], [3824, 1522, 1523], [3841, 1525, 1539], [3845, 1555, 1556], [3884, 1557, 1577], [4110, 1580, 1585], [4913, 1590, 1591], [4607, 3087, 3440], [4776, 5284, 12], [2731, 3076, 4601], [4480, 961, 2355], [4670, 1082, 1684], [1401, 1605, 1607], [660, 1610, 1619], [1947, 1575, 1631], [1948, 1635, 1645], [2512, 1646, 1653], [4101, 1654, 1655], [213, 2319, 119], [4658, 4729, 293], [5396, 1958, 988], [1692, 488, 2776, 1374], [4649, 1390, 1438], [1586, 1668, 1669], [1822, 1683, 1687], [4395, 1691, 1694], [995, 1697, 1702], [2037, 1705, 1708], [4482, 2620, 4591], [1127, 1711, 1724], [1267, 1729, 1745], [1468, 194, 1746], [1743, 1751, 1752], [1800, 1754, 1757], [2003, 1758, 1767], [2935, 1771, 1785], [3368, 1792, 1795], [3614, 1808, 1809], [3802, 1811, 1812], [3812, 1063, 1815], [3855, 1817, 1818], [4551, 1823, 1826], [4762, 1832, 1833], [4802, 1836, 1837], [5317, 1844, 1846], [240, 1852, 1856], [631, 1857, 1859], [659, 1864, 1866], [1126, 1870, 1873], [1309, 1876, 1879], [1456, 1880, 1881], [1659, 1887, 1890], [1733, 1892, 2694, 1893], [1949, 1895, 1896], [2408, 1902, 1907], [2496, 1910, 1911], [2499, 1921, 1928], [2596, 1936, 1938], [3996, 1939, 1946], [4841, 1951, 1960], [1176, 1961, 1965], [3924, 3745, 42], [3859, 635, 1744], [4269, 1972, 1973], [1326, 1974, 1976], [2413, 600, 3525], [4878, 1989, 1991], [4970, 3071, 3250], [5121, 1995, 1998], [814, 1999, 2000], [2262, 2006, 2008], [4643, 994, 260], [933, 3080, 4791], [1366, 2010, 2017], [4121, 386, 369], [271, 2020, 2028], [2855, 475, 1793], [4668, 603, 2614], [4296, 1132, 1169], [5056, 2031, 2033], [2564, 2040, 2041], [3663, 2042, 2043], [2272, 2051, 2057], [4642, 3270, 3338], [5167, 1944, 2400, 298], [118, 2064, 2067], [791, 2068, 2072], [61, 1963, 193], [598, 2080, 2083], [3860, 675, 885], [4646, 292, 1185], [361, 171, 2095], [715, 2097, 2100], [1716, 2103, 2104], [1799, 2110, 2116], [1982, 2118, 2132], [2875, 2133, 2137], [4637, 1199, 2139], [4716, 1572, 2140], [4875, 2147, 2149], [4929, 2162, 2163], [5535, 1228, 2170, 3644], [215, 2185, 2187], [2664, 2189, 2193], [3015, 2208, 2217], [4495, 2222, 2226], [3880, 1014, 1071], [63, 2228, 2233], [902, 3271, 1524], [1412, 2234, 2239], [3761, 2241, 2244], [1564, 4600, 5467], [3135, 93, 1962], [5288, 2245, 2247], [2713, 2258, 2263], [4467, 1718, 3442], [3477, 3682, 3730], [819, 2279, 2283], [1250, 2294, 2306], [2291, 2320, 2322], [3735, 2323, 2330], [1672, 2335, 2337], [167, 2339, 2342], [4093, 2344, 2356], [4527, 2060, 3116], [2424, 4951, 2367, 2370], [2333, 3768, 4200], [3792, 2979, 4720, 2374, 2382], [242, 5052, 2604], [5086, 2384, 2389], [90, 2403, 2411], [160, 2415, 2425], [511, 2430, 2432], [1413, 2434, 2435], [1945, 2437, 2438], [1356, 2439, 2441], [1769, 3657, 4851], [743, 2443, 2447], [2159, 2450, 2453], [1216, 2454, 2458], [3099, 1820, 2460, 2471], [3393, 1333, 1119], [5033, 2472, 2478], [2172, 2480, 2483], [2399, 2489, 2492], [5246, 2515, 2516], [802, 2531, 2532], [3273, 1387, 2297], [3500, 2542, 2550], [793, 2552, 2554], [828, 1245, 2569, 2571], [5487, 2584, 2586], [548, 2587, 2589], [920, 2590, 2605], [5477, 2606, 2608], [1831, 2619, 2624], [2196, 2627, 2630], [4181, 66, 1445], [4250, 2634, 2641], [417, 2665, 2666], [602, 2669, 2670], [1193, 2671, 2676], [2905, 1262, 2677, 2678], [4201, 2717, 2854], [774, 2681, 2687], [924, 2690, 2691], [2613, 2695, 2697], [3466, 2701, 2703], [2134, 2704, 4598, 2705], [721, 592, 992], [1839, 127, 5015], [3066, 2715, 2719], [1264, 4978, 2721, 2722], [1565, 1843, 2724], [1314, 259, 2737], [1563, 4002, 2745], [3583, 2749, 2751], [3655, 2759, 2765], [1096, 2885, 2770, 2772], [4446, 2781, 2782], [5054, 2796, 2799], [113, 2800, 2809], [2158, 2811, 2826], [3183, 2828, 2831], [3520, 2833, 2838], [4950, 2848, 2856], [1198, 2858, 2869], [2774, 2876, 2877], [1978, 2888, 2901], [2680, 2914, 2918], [2712, 2920, 2921], [3515, 2929, 2930], [4263, 2931, 2952], [666, 2961, 2964], [1248, 2965, 2966], [1717, 2968, 2969], [1790, 2976, 2980], [4901, 2981, 2996], [5350, 3000, 3009], [2197, 3018, 3020], [2375, 3022, 3024], [2716, 3034, 3035], [4644, 3457, 750], [2304, 3036, 3040], [2576, 3041, 3042], [5520, 3043, 3044], [163, 3045, 3051], [3147, 3058, 3069], [474, 3074, 3075], [998, 3094, 3098], [1414, 3102, 3110], [3602, 3111, 3117], [70, 3121, 3123], [1184, 3125, 3128], [2293, 3138, 3143], [2402, 3150, 3151], [3955, 1428, 74], [5489, 3153, 3158], [164, 3159, 3160], [206, 3168, 3169], [2346, 2081, 1773], [2566, 3170, 3172], [3506, 3181, 3187], [4879, 3189, 3192], [1217, 3197, 3199], [1327, 811, 1033], [2975, 3205, 3210], [453, 3212, 3220], [1701, 3222, 3224], [6, 3230, 3233], [1760, 3246, 3248], [1761, 2475, 3249], [1863, 3251, 3267], [2953, 3268, 3275], [3365, 81, 3290], [5220, 3291, 3295], [5249, 3303, 3304], [7, 3305, 3331], [24, 3339, 3352], [735, 3353, 3354], [869, 3355, 3356], [1020, 3357, 3372], [1240, 3374, 3378], [1552, 3381, 3382], [2372, 3383, 3386], [3801, 3428, 3433], [3866, 3437, 3444], [4763, 3455, 3458], [4882, 3460, 3461], [5326, 2338, 2173], [5397, 1424, 3463], [5503, 3468, 3472], [4660, 2824, 3473], [4962, 3377, 3487], [5279, 3488, 3504], [501, 3513, 3518], [547, 3521, 3526], [2523, 3532, 3554], [3213, 3555, 3569], [3496, 3574, 3579], [3799, 3589, 3590], [4555, 261, 3505], [5473, 3597, 3599], [739, 3601, 3604], [1145, 1604, 3612], [1709, 3613, 3637], [1781, 1300, 3643], [4718, 3648, 3650], [4864, 3652, 3654], [4963, 845, 3656], [5204, 2746, 3659], [5267, 2015, 3664], [5327, 554, 3665], [5508, 3670, 3676], [591, 3680, 3686], [2820, 3689, 3691], [3396, 3705, 3707], [3747, 3710, 3711], [4407, 2383, 3712], [4440, 2096, 3720], [4451, 3740, 3746], [4516, 3758, 3766], [5173, 3772, 3776], [5247, 3778, 3788], [5301, 3789, 3813], [5484, 3838, 3885], [283, 3886, 3898], [1674, 3910, 3911], [2414, 3916, 3917], [4497, 3925, 3926], [5008, 3928, 3929], [5237, 3930, 3931], [5506, 3932, 3933], [527, 3974, 3976], [691, 3980, 3982], [1373, 4003, 4009], [1600, 4010, 4011], [2288, 4017, 4025], [2417, 4034, 4035], [2578, 4041, 4044], [2904, 4054, 4055], [3607, 76, 3586], [4332, 4056, 4060], [4971, 1161, 1772], [5088, 4062, 4064], [5446, 2, 16], [117, 4065, 4073], [544, 4077, 4078], [724, 770, 1526], [1406, 4085, 4099], [1641, 2195, 4109, 4112], [1860, 4115, 4118], [2561, 4119, 4155], [5517, 4159, 4160], [913, 4161, 4162], [1578, 3484, 4163, 4203], [2340, 4219, 4220], [2436, 4222, 4224], [2702, 4229, 4231], [3773, 4237, 4253], [4106, 5071, 264], [4592, 4256, 4257], [4707, 4266, 4276], [5380, 4281, 4287], [5419, 4288, 4289], [5444, 877, 1967], [202, 4298, 4301], [325, 4309, 4320], [817, 4324, 4328], [1022, 4329, 4333], [3350, 4342, 4380], [3774, 4386, 4390], [4615, 4397, 4399], [4979, 4400, 4411], [1813, 4414, 4429], [1916, 4452, 4460], [2348, 3687, 4007], [2549, 4499, 4503], [4430, 4508, 4509], [5010, 4514, 4519], [5063, 4541, 4542], [5406, 4543, 4545], [209, 4546, 4586], [2451, 4595, 4602], [2632, 4610, 4612], [2710, 4617, 4620], [2955, 4196, 4365], [3716, 4634, 4647], [4604, 4686, 4689], [1093, 4693, 4703], [1418, 4704, 4706], [1932, 4708, 4713], [2106, 3888, 348], [2658, 220, 438], [3312, 681, 756], [5075, 4717, 4723], [5232, 4724, 4727], [687, 4728, 4730], [907, 4737, 4744], [2708, 4748, 4749], [4088, 421, 480], [4092, 4751, 4752], [4114, 2844, 4321], [4260, 4753, 4757], [4265, 4760, 4765], [5240, 4768, 4775], [5435, 4784, 4788], [238, 4626, 788], [1206, 4794, 4798], [1305, 4810, 4811], [1727, 2215, 435], [3025, 4816, 4827], [4956, 4831, 4838], [192, 4839, 4845], [1623, 4850, 4865], [1914, 4869, 4870], [2329, 4871, 4884], [2988, 4885, 4891], [3467, 4899, 4932], [3901, 4933, 4934], [4190, 4935, 4939], [4771, 4940, 4948], [162, 4952, 4955], [407, 4977, 4995], [1385, 4997, 5000], [1816, 5001, 5005], [2559, 5006, 5013], [2648, 5023, 5024], [3507, 2757, 5035, 5039], [2950, 5041, 5053], [3808, 3272, 3822], [5500, 5059, 5062], [957, 5064, 5065], [1597, 3030, 608], [1617, 5079, 5082], [1810, 5087, 5090], [1950, 5094, 5108], [2560, 5109, 5113], [2668, 5114, 5116], [3701, 728, 1507], [3849, 2069, 5122], [4379, 5135, 5151], [4731, 5152, 5155], [4890, 5172, 5174], [5211, 3649, 4145], [25, 5187, 5188], [180, 5191, 5198], [239, 5203, 5217], [313, 5225, 5226], [612, 5233, 5241], [671, 3694, 5243], [1443, 5244, 5245], [1469, 5250, 5251], [1595, 5252, 5253], [1722, 5266, 5274], [1765, 5134, 5285], [1782, 1615, 5290], [1841, 5295, 5297], [1842, 5314, 5315], [2087, 5332, 5335], [2352, 5341, 5344], [2359, 2506, 3639], [2513, 197, 5378], [2667, 5384, 5386], [2946, 5405, 5407], [3081, 5430, 5431], [3327, 5432, 5437], [3825, 5439, 5448], [3997, 101, 5450], [4100, 5457, 5466], [4632, 5472, 5476], [4640, 5478, 5480], [4764, 5485, 5494], [4954, 5501, 5510], [5212, 5534], [5381], [5445], [5468], [689], [709], [1246], [1417], [1858], [2305], [3592], [3770], [4608, 1899, 2939], [4809, 2313, 1689], [5302], [38], [509], [540], [852], [1520, 1078], [1086], [1236], [1642], [1794], [1862], [2321], [2467], [2900], [3344, 1554], [159], [1032], [1721], [1753], [2301], [2573], [2649], [2986], [3254, 1501, 2442], [3269], [3777], [4792], [4793], [5286], [5522], [88], [450], [472], [1458], [1622], [2723], [2768, 842, 3660], [2895], [3493], [3698], [4261], [4428], [5091], [5349], [390], [720], [773], [1237], [1306], [1500], [1814], [1894], [2302], [2647], [3388], [4314], [5287], [347, 1493, 2738], [414], [567], [719], [820], [1030], [1188], [1396], [2243], [2366], [2405, 140, 297], [2795], [2982, 31, 383], [3340], [4251], [4432, 3600, 684], [4614], [5057], [393], [872], [1021], [1106], [2035], [2261], [2416], [2508, 1196, 1263], [2595, 3376], [2783], [3491], [3673], [4147], [4254], [4754], [5175], [5298], [628, 5171], [707], [803], [1391], [1582], [2165], [2224], [2469, 3968], [2470, 5530, 5382], [2509, 4297], [2510, 3134], [3061], [3140], [3301, 3906], [3332], [3489], [4040], [4293], [4625, 2005], [4886], [5044], [5189], [5385], [83], [772, 3023], [1018], [1394], [1579], [1700], [1821], [1905], [1917], [2427], [2652], [3065], [3103], [3439], [4525], [5103], [482], [537], [551], [792], [1025], [1234], [1298], [1640], [1869, 661, 2468], [2289], [2645], [2867], [3079], [3127], [3434], [4780, 3379], [4789], [4996], [5093], [5282], [5519], [443], [513], [526], [665], [1673], [2223], [2594], [2725], [3104], [3871], [4211], [4848], [5047], [191], [771], [1168, 1421], [1367], [1616], [1913], [1931], [2280], [2579], [2650], [2763], [2797], [3748], [3915, 3141, 410], [4262], [5095], [5186], [5242], [5408], [243, 4766], [2157, 797], [941], [1089], [1190], [1375], [1521], [1819], [1855], [2281], [2653], [2846], [3124], [3156], [3741], [3757], [4045], [4120], [4690], [5014], [5479], [5507], [109], [188], [327], [376], [738], [851], [1670], [2310], [2311], [2368], [2445], [2631], [2711], [3070], [3722], [3894], [4513], [4844], [5258], [5441], [515], [557], [1088], [1214], [1222], [1581], [1730], [2381], [2775], [2817], [2840], [3083], [3451], [3470], [3551], [3940, 1221], [3961, 315], [4097], [4187, 3662], [4352, 3973], [4858], [4909, 3971], [4993], [5111], [5192], [5300], [5311], [5449], [408], [940], [1141], [1153], [1479], [1562], [1825], [1889], [1956], [2098], [2361], [2628], [2646], [2786], [3247], [3387], [3462], [3465], [3755], [3953, 1703], [4299], [4334], [4815], [5119], [5236], [339], [686, 382], [796], [854], [954], [1602], [2308], [2734, 1103, 1686], [2778], [2992], [3742], [4069], [4216], [5304], [5483], [5509], [67], [203], [250], [323], [471], [502], [505], [579], [639], [640], [644], [874], [912], [945], [1004], [1085], [1156], [1197], [1504], [1583], [1878], [1912], [2062], [2160], [2300], [2379], [2563], [2622], [2707], [2767], [2812], [3084], [3351], [3534], [3672], [4787], [4941], [5118], [5263], [5458], [111], [286], [364, 5383], [447], [528], [574], [711], [1353], [1399], [1739], [2326], [2572], [2643], [2802], [3031], [3126], [3206], [3441], [3499], [3523], [3538], [3736], [4242], [4339], [5081], [5235], [43], [413], [798], [970], [1000], [1211], [1342], [1952], [1964], [2190], [2220], [2230], [2392], [2444], [2486], [2829], [2839], [2973], [3032], [3668], [3889], [3979], [4603], [4814], [4944], [4959], [5009], [5170], [5299], [5460], [114], [227], [321], [708], [741], [918], [975], [1054], [1069], [1644], [1897], [1970], [2025], [2059], [2188], [2473], [2623], [2656], [2989], [3027], [3115], [3263], [3443], [3483], [3494], [3550], [3593], [3726], [3828, 4924, 780, 925], [4986, 4038, 4039], [4049], [4096], [4195], [4378, 4575, 1090, 2199], [4427], [4454, 4511], [4570], [4747], [4849], [4915], [5029], [5069], [5462], [44], [92], [94], [96], [204], [251], [269], [375], [378], [425], [549], [873], [884], [1131], [1191], [1331], [1398], [1419], [1448], [1452], [1480], [1621], [1886], [2164], [2221], [2287], [2621], [2682], [2810], [3078], [3375], [3436], [3706], [3765], [3767], [4403], [4630], [4688], [4946], [4987], [5388], [5488], [130], [166], [178], [350], [477], [506], [596], [634], [656], [748], [969], [1166], [4953, 1254], [1257], [1283], [1403], [1490], [1517], [1803], [1835], [1854], [2092], [2264], [2362], [2479], [2615], [2872], [2947], [2984], [3067], [3068], [3086], [3361], [3430], [3575], [3596], [4259], [4517], [4820], [4938], [4947], [5336], [5365], [5404], [5475], [108], [120], [152], [427], [479], [692], [710], [795], [829], [1023], [1041], [1070], [1242], [1294], [1393], [1402], [1444], [1665], [1871], [1942], [2061], [2078], [2125, 2328, 2369], [2135], [2225], [2260], [2418], [2493], [2819], [2882], [2883], [2945], [3131], [3256], [3452], [3459], [3647], [3675], [3688], [4234], [4255], [4290], [4300], [4533], [4540], [4650], [4725], [4892], [4957], [5034], [5178], [5277], [5532], [161], [249], [541], [713], [717], [891], [972], [1006], [1053], [1179], [1212], [1249], [1295], [1341], [1636], [1680], [1898], [1920], [1930], [1969], [2052], [2109], [2142], [2169], [2212], [2562], [2609], [2683], [2727], [2789], [2913], [2951], [3062], [3063], [3211], [3216], [3257], [3278], [3380], [3469], [3486], [3517], [3700], [4048], [4383], [4695], [4711], [5045], [5046], [5066], [5096], [5112], [5296], [5347], [5482], [208], [335], [530], [577], [578], [3426, 589], [595], [694], [697], [765], [813], [937], [1034], [1047], [1139], [1143], [1335], [1389], [1392], [1651], [1937, 2065, 2345], [2009], [2178], [2253], [2290], [2363], [2553], [2602, 5454], [2684], [2830], [2915], [3012], [3207], [3373], [3476], [3490], [3492], [3594, 37, 3584], [3685], [3718], [3725], [3978], [4218], [4247], [4292], [5025], [5031], [5123], [5227], [5303], [5342], [5343], [5465], [5512], [49], [54], [89], [784, 100, 1231], [124], [147], [154], [155], [300], [341], [367], [389], [392], [3696, 522], [545], [676], [812], [860], [1015], [5021, 1062], [1107], [1138], [1151], [1235], [1362], [1386], [1395], [1449], [1516], [1543, 1144, 2126], [1618], [1633], [1643], [1647], [1682], [1713], [1723], [1725], [1756], [1838], [1874], [1882], [1941], [2032], [2034], [2039], [2066], [2153], [2211], [2238], [2286], [2318], [2325], [2364], [2365], [2429], [2616], [2625], [2758], [2771], [2967], [2974], [3060], [3077], [3293], [3464], [3516], [3548], [3641], [3743], [3790], [3993], [4070], [4249], [4687], [4692], [4812], [4942], [5083], [5084], [5092], [5224], [5256], [0], [105], [142], [165], [199], [210], [280], [318], [356], [446], [532], [539], [555], [645], [742], [769], [809], [889], [926], [959], [1081], [1108], [1122], [1187], [1244], [1278], [1345], [1350], [1416], [1439], [1576], [1652], [1742, 1486], [1789], [1888], [1929], [1953], [2054], [2056], [2166], [2235], [2242], [2248], [2296], [2309], [2371], [2481], [2485], [2505], [2726], [2832], [2851], [2891], [2898], [2912], [3001], [3038], [3089], [3149], [3435], [3454], [3546], [3603], [3658], [3727], [3753], [3878], [3909], [4068], [4217], [4462], [4464], [4496], [4596], [4830], [4835], [4912], [4945], [5110], [5234], [5238], [5461], [19], [55], [168], [179], [275], [284], [314], [334], [338], [388], [411], [419], [496, 1704], [576], [674], [746], [767], [768], [778], [840], [850], [921], [939], [1017], [1091], [1120], [1123], [1133], [1134], [1167], [1301], [1372], [1420], [1427], [1433], [1488], [1489], [1506], [1627], [1666], [1695], [1699], [1719], [1797], [1918], [1977], [1985], [2079], [2111], [2138], [2145], [2167], [2171, 4461], [2175], [2249], [2259], [2271], [2303], [2314], [2398], [2440], [2557], [2581], [2611], [2714], [2755], [2777], [2785], [2813], [2881], [2903], [2944], [2978], [3008], [3011], [3014], [3109], [3194], [3195], [3217], [3218, 4213], [3336], [3341], [3359], [3360], [3402, 4067], [3456], [3471], [3501], [3519], [3547], [3576], [3633], [3666], [3681], [3721], [3734], [3816], [3891], [4051], [4066], [4243], [4277], [4369], [4401], [4450], [4613], [4817], [4873], [4958], [5003], [5042], [5067], [5070], [5147], [5281], [5305], [5340], [5433], [5434], [39], [98], [99, 2853], [149], [201], [291], [304], [306], [331], [1481, 342], [387], [391], [396], [405], [409], [415], [463], [542], [597], [646], [654], [753], [764], [836], [855], [916], [922], [973], [991], [1129], [1135], [1155], [1172], [1173], [1241], [1251], [1261], [1348], [1351], [1357], [1371], [1478], [1494], [1530], [1584], [1679]]
    
def analyze_bom():
    consistuent_list = {}   # children
    parent_list = {}    # parent
    composition_list = {}   # production consumption relation between item and sub-components in its subtree 

    for i in item_df.I:
        consistuent_list[i] = []
        parent_list[i] = []
        for bk in df_bom.index[(df_bom.I == i) | (df_bom.II == i)]:
            if (df_bom.I[bk] == i) and (not (df_bom.II[bk] in consistuent_list[i])):
                consistuent_list[i].append(df_bom.II[bk])
            if (df_bom.II[bk] == i) and (not (df_bom.I[bk] in parent_list[i])):
                parent_list[i].append(df_bom.I[bk])

    degree_list = np.zeros(I_norm)
    keep_iter = True
    dg_no = 0.0
    while keep_iter:
        now_degree = copy.deepcopy(degree_list)
        for bk in df_bom.index:
            i1 = df_bom.I[bk]
            i2 = df_bom.II[bk]
            if degree_list[i2] == dg_no:
                now_degree[i1] = max(now_degree[i1], degree_list[i2] + 1)
                composition_list[i1, i2] = df_bom.V[bk]
                current_dict_keys = list(composition_list.keys())
                for item in current_dict_keys:
                    if item[0] == i2:
                        composition_list[i1, item[1]] = df_bom.V[bk] * composition_list[item]

        if np.array_equiv(now_degree, degree_list):
            keep_iter = False
        else:
            degree_list = now_degree
            dg_no += 1

    subsidiary_list = {}   # subtree
    precursor_list = {}   # precursors
    for i in item_df.I:
        subsidiary_list[i] = [item[1] for item in composition_list.keys() if item[0] == i]
        precursor_list[i] = [item[0] for item in composition_list.keys() if item[1] == i]
    
    return degree_list, consistuent_list, parent_list, composition_list, subsidiary_list, precursor_list

def extensive_prob(relax_option):
    ext_prob = gp.Model("extensive_form")
    ext_prob.setParam("Threads", 1)
    #ext_prob.setParam("OutputFlag", 0)

    var_start = time.time()
    # set up model parameters
    u = ext_prob.addVars(item_df.I, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
    s = ext_prob.addVars(df_transit.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t
    z = ext_prob.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="z")  # z_{ijt} for j,t
    v = ext_prob.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t
    yUI = ext_prob.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUO = ext_prob.addVars(item_df.I, plant_df.I, period_df.I, lb=0.0, name="yo")  # y^{O}_{ijt} for j,t
    xC = ext_prob.addVars(prod_key, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t
    rC = ext_prob.addVars(df_alt.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    var_end = time.time()
    print("Finish setting up variables in %r seconds!" % (var_end - var_start))
    # initial condition setup
    for i in item_df.I:
        u[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in item_df.I:
        for j in plant_df.I:
            v[i, j, min(period_df.I) - 1] = init_inv.get((i, j), 0.0)  # initial inventory set to given values
            
            for t in period_df.I:
                if not((i, j, t) in external_purchase.keys()):
                    external_purchase[i, j, t] = 0.0

        for t in period_df.I:
            if not((i, t) in real_demand.keys()):
                real_demand[i, t] = 0.0
    for l in df_transit.Tr:
        for t in range(min(period_df.I) - int(transit_time[l]), min(period_df.I)):
            s[l, t] = 0.0  # initial transportation set to 0
    for i, j in prod_key:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xC[i, j, t] = 0.0  # initial production set to 0

    constr_start = time.time()
    # add constraints for the extensive formulation
    ext_prob.addConstrs((u[i, t] - u[i, t - 1] + gp.quicksum(z[i, j, t] for j in plant_df.I) \
                         == real_demand[i, t] for i in item_df.I for t in period_df.I), name='unmet_demand')
    ext_prob.addConstrs((v[i, j, t] - v[i, j, t - 1] - yUI[i, j, t] + yUO[i, j, t] == 0 \
                         for i in item_df.I for j in plant_df.I for t in period_df.I), name='inventory')

    ext_prob.addConstrs((yUI[i, j, t] == gp.quicksum(
        s[df_transit.Tr[l], t - int(transit_time[l])] for l in
        df_transit.index[(df_transit.I == i) & (df_transit.JJ == j)]) +
                         (xC[i, j, t - lead_time[i, j]] if (i,
                                                            j) in prod_key else 0.0) + external_purchase[i, j, t] + \
                         gp.quicksum(df_alt.V[jta] * rC[jta] for jta in df_alt.index[
                             (df_alt.J == j) & (df_alt.Ti == t) & (df_alt.II == i)])
                         for i in item_df.I for j in plant_df.I for t in period_df.I),
                        name='input_item')

    ext_prob.addConstrs((yUO[i, j, t] == gp.quicksum(
        s[df_transit.Tr[l], t] for l in df_transit.index[(df_transit.I == i) & (df_transit.J == j)]) +
                         gp.quicksum(
                             df_bom.V[bk] * xC[df_bom.I[bk], j, t] for bk in
                             df_bom.index[(df_bom.J == j) & (df_bom.II == i)] if (df_bom.I[bk], j) in prod_key) +
                         z[i, j, t] +
                         gp.quicksum(rC[jta] for jta in df_alt.index[
                             (df_alt.J == j) & (df_alt.Ti == t) & (df_alt.I == i)])
                         for i in item_df.I for j in plant_df.I for t in period_df.I),
                        name="output_item")

    ext_prob.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xC[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys()
        if (j_iter == j) and (ct == ct_iter) and ((i_iter, j_iter) in prod_key)) <= max_cap[ct, j, t]
                         for ct, j, t in max_cap.keys()), name='capacity')
    ext_prob.addConstrs(
        (rC[jta] <= v[df_alt.I[jta], df_alt.J[jta], df_alt.Ti[jta] - 1] for jta in df_alt.index),
        name='r_ub')
    ext_prob.addConstrs((yUO[i, j, t] <= v[i, j, t - 1] for i in item_df.I for j in plant_df.I for t in period_df.I),
                        name='yo_ub')
    
    if relax_option:
        ext_prob.addConstrs(
            (xC[i, j, t] <= max_prod[i, j] for i, j in prod_key for t in period_df.I),
            name='w_ub')
    else:
        w = ext_prob.addVars(prod_key, period_df.I, vtype=GRB.INTEGER, lb=0.0, name="w")

        ext_prob.addConstrs(
            (xC[i, j, t] == w[i, j, t] * lot_size[i, j] for i, j in prod_key for t in period_df.I),
            name='batch')
        ext_prob.addConstrs(
            (xC[i, j, t] <= max_prod[i, j] for i, j in prod_key for t in period_df.I),
            name='w_ub')

   
    constr_end = time.time()
    print("Finish setting up constraints in %r seconds!" % (constr_end - constr_start))

    # set up the subproblem specific objective
    obj = gp.quicksum(holding_cost[i] * v[i, j, t] for i in item_df.I for j in plant_df.I for t in
                      period_df.I) + \
          gp.quicksum(penalty_cost[i] * u[i, t] for i in item_df.I for t in period_df.I) + \
          gp.quicksum(transit_cost[df_transit.Tr[l]] * s[df_transit.Tr[l], t] for l in df_transit.index for t in
                      period_df.I)

    ext_prob.setObjective(obj, GRB.MINIMIZE)

    ext_prob.update()

    obj_end = time.time()
    print("Finish setting up objective in %r seconds!" % (obj_end - constr_end))

    ext_prob.optimize()

    solve_end = time.time()
    print("Finish solving in %r seconds!" % (solve_end - obj_end))

    #print("demand satisfaction rate:")
    #for i in item_df.I:
    #    total_demand = sum(real_demand.get((i, t), 0) for t in period_df.I)
    #    if total_demand > 1e-2:
    #        print(item_df.index[i], " : ", round((1 - u[i, max(period_df.I)].X / total_demand) * 100, 2), " %")
    
    total_HC = sum(holding_cost[i] * v[i, j, t].X for i in item_df.I for j in plant_df.I for t in
                      period_df.I)
    total_PC = sum(penalty_cost[i] * u[i, t].X for i in item_df.I for t in period_df.I)
    total_TC = sum(transit_cost[df_transit.Tr[l]] * s[df_transit.Tr[l], t].X for l in df_transit.index for t in
                      period_df.I)
    
    # write extensive x sol to csv
    header = ['I', 'J', 'Ti', 'V']

    with open("ext_batch.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for i, j in prod_key:
            for t in period_df.I:
                data_output = [i, j, t, xC[i, j, t].X / lot_size[i, j]]
                writer.writerow(data_output)
    
    print("total HC, total PC, total TC", total_HC, total_PC, total_TC)
    return ext_prob.ObjVal

def pi_iter_sparse_feas(k, x_vals, sparse_cuts=True):
    # Attempt to find the flat constraints with better numerical properties
    #worker_init(data)

    unit_ind_list = decomp_units[k]

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    sp.setParam("Threads",1)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
    
    if sparse_cuts:
        # find the related items and create x copies for only them
        item_used_list = []
        for i in unit_ind_list:
            item_used_list.append(i)
            for bk in df_bom.index[df_bom.II == i]:
                item_used_list.append(df_bom.I[bk])
        df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
        prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
    else:
        prod_key_used = list(zip(df_prod.I, df_prod.J))
    xCi = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    x_im_p = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x_p")  # x_imb_p_{ijt} for i,j,t (x copy)
    x_im_m = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x_m")  # x_imb_p_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            if (i,j) in init_inv.keys():
                vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
            else:
                vi[i, j, min(period_df.I) - 1] = 0.0
            for t in period_df.I:
                if not((i, j, t) in external_purchase.keys()):
                    external_purchase[i, j, t] = 0.0
        for t in period_df.I:
            if not((i, t) in real_demand.keys()):
                real_demand[i, t] = 0.0
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - int(transit_time[l]), min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key_used:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xCi[i, j, t] = 0.0  # initial production set to 0
        for t in period_df.I:
            x_im_p[i, j, t].UB = max_prod[i, j]
            x_im_m[i, j, t].UB = max_prod[i, j]

    rCi = sp.addVars(alt_i.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in unit_ind_list for t in period_df.I), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi[i, j, t] == 0 for i in unit_ind_list \
         for j in plant_df.I for t in period_df.I), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t - int(transit_time[l])] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - int(lead_time[i, j])] if (i, j) in prod_key_used else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)] if (df_bom.I[bk], j) in prod_key_used) + zi_p[i, j, t] +
                   gp.quicksum(rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i,j] for i, j in prod_key_used for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key_used) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # fix the local variables to their global counterparts' values
    # solve subproblem to generate feasibility cuts
    sub_fix_x = {}
    for i, j in prod_key_used:
        for t in period_df.I:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] + x_im_p[i,j,t] - x_im_m[i,j,t] == x_vals[i, j, t] * lot_size[i, j])
    
    # set up the subproblem specific objective
    obj = gp.quicksum(x_im_p[i, j, t] + x_im_m[i, j, t] for i,j in prod_key_used for t in period_df.I)
    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    if sp.ObjVal > 0:
        feas_gen = True
        feas_value = sp.ObjVal
        feas_cut_coeff = {}
        for i, j in prod_key_used:
            for t in period_df.I:
                if abs(sub_fix_x[i, j, t].Pi * lot_size[i, j]) > 1e-4:
                    feas_cut_coeff[i, j, t] = sub_fix_x[i, j, t].Pi * lot_size[i, j]
                else:
                    feas_cut_coeff[i, j, t] = 0
    else:
        feas_gen = False
        feas_value = 0
        feas_cut_coeff = 0
    
    return k, feas_gen, feas_cut_coeff, feas_value

def pi_iter_sparse_opt(k, x_vals, theta_vals, lag_cuts=True, sparse_cuts=True, level_lambda=0.5, level_mu=0.8):
    # Attempt to find the flat constraints with better numerical properties
    #worker_init(data)

    unit_ind_list = decomp_units[k]

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    #print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    sp.setParam("Threads",1)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand
     
    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t
   
    if sparse_cuts:
        # find the related items and create x copies for only them
        item_used_list = []
        for i in unit_ind_list:
            item_used_list.append(i)
            for bk in df_bom.index[df_bom.II == i]:
                item_used_list.append(df_bom.I[bk])
        df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
        prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
    else:
        prod_key_used = list(zip(df_prod.I, df_prod.J))
    xCi = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    x_im_p = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x_p")  # x_imb_p_{ijt} for i,j,t (x copy)
    x_im_m = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x_m")  # x_imb_p_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            if (i,j) in init_inv.keys():
                vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
            else:
                vi[i, j, min(period_df.I) - 1] = 0.0
            for t in period_df.I:
                if not((i, j, t) in external_purchase.keys()):
                    external_purchase[i, j, t] = 0.0
        for t in period_df.I:
            if not((i, t) in real_demand.keys()):
                real_demand[i, t] = 0.0
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - int(transit_time[l]), min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key_used:
        for t in range(min(period_df.I) - int(lead_time[i, j]), min(period_df.I)):
            xCi[i, j, t] = 0.0  # initial production set to 0
        for t in period_df.I:
            x_im_p[i, j, t].UB = 1e-6
            x_im_m[i, j, t].UB = 1e-6

    rCi = sp.addVars(alt_i.index, lb=0.0, name="r")  # r_{ajt} for a=(i,i')

    # 0-padding the real demand
    sp.addConstrs((ui[i, t] - ui[i, t - 1] + gp.quicksum(zi_p[i, j, t] for j in plant_df.I) \
                   == real_demand[i, t] for i in unit_ind_list for t in period_df.I), name='unmet_demand')
    sp.addConstrs(
        (vi[i, j, t] - vi[i, j, t - 1] - yUIi[i, j, t] + yUOi[i, j, t] == 0 for i in unit_ind_list \
         for j in plant_df.I for t in period_df.I), name='inventory')

    sp.addConstrs((yUIi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t - int(transit_time[l])] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - int(lead_time[i, j])] if (i, j) in prod_key_used else 0.0) + external_purchase[i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)] if (df_bom.I[bk], j) in prod_key_used) + zi_p[i, j, t] +
                   gp.quicksum(rCi[jta] for jta in alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i,j] for i, j in prod_key_used for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key_used) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # fix the local variables to their global counterparts' values
    # solve subproblem to generate feasibility cuts
    sub_fix_x = {}
    for i, j in prod_key_used:
        for t in period_df.I:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] + x_im_p[i,j,t] - x_im_m[i,j,t] == x_vals[i, j, t] * lot_size[i, j])

    # obtain the LP objective value of the subproblem
    obj = gp.quicksum(
        holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
          gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    #if sp.Status != 2:
    #    print("Error in " + str(k))

    sub_obj = sp.ObjVal

    sub_HC = sum(holding_cost[i] * vi[i, j, t].X for i in unit_ind_list for j in plant_df.I for t in period_df.I)
    sub_PC = sum(penalty_cost[i] * ui[i, t].X for i in unit_ind_list for t in period_df.I)
    sub_TC = sum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t].X for l in transit_list_i.index for t in period_df.I)
    
    #print(k, " ", sub_obj, " ", theta_vals[k])
    if ((sub_obj - theta_vals[k]) > 1e-4) and (sub_obj > 1.00001 * theta_vals[k]):
        cut_gen = True

        dual_coeff = {}
        for i, j in prod_key_used:
            for t in period_df.I:
                if abs(sub_fix_x[i, j, t].Pi) > 1e-4:
                    dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi
                else:
                    dual_coeff[i, j, t] = 0

        iter_num = 0

        if lag_cuts:
            # initialize the pi_hat as 0
            pi_hat = {}
            best_dual = {}
            GList = []
            fList = []

            # set up the lower bound problem
            smcp = gp.Model()
            smcp.setParam("OutputFlag",0)
            smcp.setParam("Threads",1)
            pi = smcp.addVars(prod_key_used, period_df.I, lb=-float('inf'), name="beta")
            pi_abs = smcp.addVars(prod_key_used, period_df.I, lb=0.0, name="beta_abs")
            smcp.setObjective(gp.quicksum(pi_abs[i,j,t] for i,j in prod_key_used for t in period_df.I), GRB.MINIMIZE)
            smcp.addConstrs(pi_abs[i,j,t] >= pi[i,j,t] for i,j in prod_key_used for t in period_df.I)
            smcp.addConstrs(pi_abs[i,j,t] >= -pi[i,j,t] for i,j in prod_key_used for t in period_df.I)
            smcp.update()
            smcp.optimize()
            current_ub = np.inf
            f_star = smcp.ObjVal

            # set up the level problem
            lp = gp.Model()
            lp.setParam("OutputFlag",0)
            lp.setParam("Threads",1)
            pi_level = lp.addVars(prod_key_used, period_df.I, lb=-float('inf'), name="beta")
            pi_abs_level = lp.addVars(prod_key_used, period_df.I, lb=0.0, name="beta_abs")
            theta_lp = lp.addVar(name = "theta")
            lp.addConstrs(pi_abs_level[i,j,t] >= pi_level[i,j,t] for i,j in prod_key_used for t in period_df.I)
            lp.addConstrs(pi_abs_level[i,j,t] >= -pi_level[i,j,t] for i,j in prod_key_used for t in period_df.I)

            for i, j in prod_key_used:
                for t in period_df.I:
                    sp.remove(sub_fix_x[i, j, t])

            alpha = 0.0
            keep_iter = True
            
            while (keep_iter) and (iter_num <= 1000):
                iter_num += 1
                #print(iter_num," ", pi_obj," ",sub_obj)
                
                # generate the cut to characterize the Lagrangian function           
                obj = gp.quicksum(
                    holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
                    gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
                    gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
                    gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t]) for i, j in prod_key_used for t in period_df.I)
                    # gp.quicksum(
                    #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

                sp.setObjective(obj, GRB.MINIMIZE)
                sp.update()
                sp.optimize()

                pi_obj = sp.ObjVal
                fList.append(np.sum(np.abs(list(pi_hat.values()))))
                if iter_num == 1:
                    best_dual = copy.deepcopy(pi_hat)
                elif sub_obj - pi_obj < np.min(GList):
                    best_dual = copy.deepcopy(pi_hat)
                GList.append(sub_obj - pi_obj)

                # if we have obtain a good solution, stop
                if sub_obj - pi_obj <= 0.01 * sub_obj:
                    keep_iter = False
                else:
                    x_grad_vals = {}
                    for i, j in prod_key_used:
                        for t in period_df.I:
                            x_grad_vals[i,j,t] = x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t].X

                    # adding the cuts for the Lagrangian function
                    smcp.addConstr(sub_obj - pi_obj - gp.quicksum(x_grad_vals[i,j,t] * (pi[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key_used for t in period_df.I) <= 0)
                    smcp.update()
                    smcp.optimize()
                    f_star = smcp.ObjVal
                    lp.addConstr(theta_lp >= sub_obj - pi_obj - gp.quicksum(x_grad_vals[i,j,t] * (pi_level[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key_used for t in period_df.I))

                    # calculate f_*, h_i, Delta
                    hp = gp.Model()
                    hp.setParam("OutputFlag",0)
                    alpha_v = hp.addVar(lb = 0.0, ub = 1.0, name = "alpha")
                    h_theta = hp.addVar(name = "h_i")
                    hp.addConstrs(h_theta <= alpha_v * (fList[j] - f_star) + (1 - alpha_v) * (GList[j]) for j in range(len(GList)))

                    # calculate alpha
                    hp.addConstr(h_theta >= 0)
                    hp.setObjective(alpha_v, GRB.MAXIMIZE)
                    hp.update()
                    hp.optimize()
                    alpha_max = hp.ObjVal

                    hp.setObjective(alpha_v, GRB.MINIMIZE)
                    hp.update()
                    hp.optimize()
                    alpha_min = hp.ObjVal

                    if iter_num == 1:
                        alpha = 0.5 * (alpha_max + alpha_min)
                    else:
                        if ((alpha - alpha_min)/(alpha_max - alpha_min) <= 1 - 0.5 * level_mu) and ((alpha - alpha_min)/(alpha_max - alpha_min) >= 0.5 * level_mu):
                            alpha = 0.5 * (alpha_max + alpha_min)

                    # obtain the lb, ub, and level
                    lb_est = alpha * f_star
                    ub_est = np.min([alpha * fList[j] + (1 - alpha) * GList[j] for j in range(len(GList))])
                    level = level_lambda * ub_est + (1 - level_lambda) * lb_est
                    level_con = lp.addConstr(alpha * gp.quicksum(pi_abs_level[i,j,t] for i,j in prod_key_used for t in period_df.I) + (1 - alpha) * theta_lp <= level)
                    lp.setObjective(gp.quicksum((pi_level[i,j,t] - pi_hat[i,j,t]) ** 2 for i,j in prod_key_used for t in period_df.I))

                    # generate the next pi
                    lp.update()
                    lp.optimize()
                    for i,j in prod_key_used:
                        for t in period_df.I:
                            pi_hat[i,j,t] = pi_level[i,j,t].X
                    lp.remove(level_con)

            for i,j in prod_key_used:
                for t in period_df.I:
                    if abs(best_dual[i,j,t]) < 1e-6: 
                        pi_hat[i,j,t] = 0
                    else:
                        pi_hat[i,j,t] = best_dual[i,j,t]

        else:
            pi_hat = dual_coeff
        
        if lag_cuts:
            obj = gp.quicksum(
                holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
                gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
                gp.quicksum(transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in period_df.I) + \
                gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t]) for i, j in prod_key_used for t in period_df.I)
            # gp.quicksum(
            #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

            sp.setObjective(obj, GRB.MINIMIZE)
            sp.update()
            sp.optimize()

        pi_obj = sp.ObjVal
        for i, j in prod_key_used:
            for t in period_df.I:
                pi_hat[i,j,t] *= lot_size[i,j]
        #print(multiprocessing.current_process(), " ", k, " ", pi_obj, " ", sub_obj, " ", iter_num)
    else:
        cut_gen = False
        pi_hat = 0
        pi_obj = sub_obj
        #print(multiprocessing.current_process(), " ", k, " ", sub_obj)

    return k, cut_gen, sub_obj, pi_obj, pi_hat, sub_HC, sub_PC, sub_TC

if __name__ == "__main__":
    # read in data
    data_folder = "./data/fine_tune"   #medium_D0_0418/D1_0222/D4_0801

    df_cost, df_alt, df_iniInv, df_external, df_unitCap, df_prod, df_transit, \
        df_transitTC, df_bom, df_maxCap, df_demand, item_df, plant_df, period_df, set_df, \
        I_norm, J_norm, T_norm, Ct_norm, Tr_norm, item_data = readin_array(data_folder)
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

    worker_init(gbv)

    #extensive_prob(relax_option=False)
    #exit()
        
    # initialize the Benders procedure
    max_iter = 100    
    rel_gap_tol = 0.01
    iter_num = 0
    best_lb = -math.inf
    best_ub = math.inf
    ubList = []
    lbList = []
    
    # Accelerate technique 1: feasibility + optimality cut
    feas_opt_cut_flag = True

    # Accelerate technique 2: regularization - level_set / regularized BD
    reg_BD_flag = True
    level_fractile = 0.5
    level_obj_norm = 2    # 1: 1-norm / 2:2-norm / 1.5:2-piece approximation of 2-norm
    best_incum = {(i, j, t): 0 for i, j in prod_key for t in period_df.I}

    # Accelerate technique 3: LP warm start (Benders for LP relaxation)
    lp_warm_flag = True
    lp_warm_iter = 50      # Benders Maximum Iteration
    lp_rel_gap_tol = 0.05   # Benders Gap Tolerance 5%
    
    # Accelerate technique 4: Add logical constraint to master
    logic_constr_flag = False

    # Generate sparse cut?
    sparse_cut_flag = False

    # fixing technique (Fix production variables that remain unchanged for many iterations)
    fixing_flag = False
    if fixing_flag:
        fixing_x = np.array([0 for i, j in prod_key for t in period_df.I])
        fixing_round = np.array([0 for i, j in prod_key for t in period_df.I])
        thres_FR = 50

    # gomory feasibility cut technique
    gomory_feas_flag = False
    
    # print out result
    result_out_path = "Pool_small.txt"
    # time after each iteration
    time_list = []  

    start = time.time()
    
    if lp_warm_flag:
        mp, mp_handle = master_prob(relax_option=True)
    else:
        mp, mp_handle = master_prob(relax_option=False)

    # level set regularization initialization
    if reg_BD_flag:
        level = 1e10
    #    if cut_opt == "multi_cut":
        level_constr = mp.addConstr(
            gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))) <= level)
    #     elif cut_opt == "single_cut":
    #         level_constr = mp.addConstr(mp.getVarByName(f"theta") <= level)
        
        if level_obj_norm == 1:    # 1-norm regularization
            x_dev_constr1 = mp.addConstrs(
                            mp.getVarByName(f"x[{i},{j},{t}]") + mp.getVarByName(f"x_dev[{i},{j},{t}]") >= best_incum[i, j, t] for i, j in prod_key for t
                            in period_df.I)
            x_dev_constr2 = mp.addConstrs(
                            mp.getVarByName(f"x[{i},{j},{t}]") - mp.getVarByName(f"x_dev[{i},{j},{t}]") <= best_incum[i, j, t] for i, j in prod_key for t
                            in period_df.I)
        
        elif level_obj_norm == 1.5:   #2-piece approximation of 2-norm regularization
            # L1 template constraints
            # y - x >= -xhat
            # y + x >=  xhat
            # RHS will be updated
            # -----------------------------
            abs_pos = mp.addConstrs(
                            mp.getVarByName(f"y[{i},{j},{t}]") - mp.getVarByName(f"x[{i},{j},{t}]") >= 0 for i, j, t in l1_keys)
            abs_neg = mp.addConstrs(
                            mp.getVarByName(f"y[{i},{j},{t}]") + mp.getVarByName(f"x[{i},{j},{t}]") >= 0 for i, j, t in l1_keys)
            
            # PWL templates (dummy init)
            # -----------------------------
            pwl = {}
            for i, j, t in pwl_keys:
                z = mp.getVarByName(f"z[{i},{j},{t}]")
                y = mp.getVarByName(f"y[{i},{j},{t}]")

                # dummy convex PWL
                bp = [-1.0, 0.0, 1.0]
                val = [1.0, 0.0, 1.0]

                pwl[i, j, t] = mp.addGenConstrPWL(z, y, bp, val)
            
        mp.update()
    print(f"Master initialized in {time.time() - start} seconds.")

    pool = Pool(5, initializer=worker_init, initargs=(gbv,))

    # Lp warm start
    if lp_warm_flag:
        LP_iter_num = 0
        while True:
            LP_iter_num += 1

            wr = "\nLP Iteration: " + str(LP_iter_num)
            wr_s = open(result_out_path, "a")
            wr_s.write(wr)
            wr_s.close()
            
            # Conventional Benders
            if not reg_BD_flag:
                mp_start_time = time.time()
                mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
                mp.update()
                mp.optimize()
                mp_end_time = time.time()

                x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
                theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}

                lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))
                lbList.append(lb_cand)
                if lb_cand > best_lb:
                    best_lb = lb_cand
            # Regularized Benders
            else:
                # master
                RB_mp_start_time = time.time()
                level_constr.rhs = math.inf
                mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
                mp.update()
                mp.optimize()
                RB_mp_end_time = time.time()

                theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}
                lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))
                lbList.append(lb_cand)
                if lb_cand > best_lb:
                    best_lb = lb_cand

                # auxiliary problem, refinement to master
                RB_ap_start_time = time.time()
                # update level set auxiliary problem
                if best_ub < math.inf and best_lb > -math.inf:
                    level = best_lb + level_fractile * (best_ub - best_lb)
                    level_constr.rhs = level

                if level_obj_norm == 2:
                    obj = gp.quicksum(
                        (mp.getVarByName(f"x[{i},{j},{t}]") - best_incum[i, j, t]) ** 2 for i, j in prod_key for t in
                        period_df.I)
                elif level_obj_norm == 1:                    
                    for i, j in prod_key:
                        for t in period_df.I:
                            x_dev_constr1[i, j, t].rhs = best_incum[i, j, t]
                            x_dev_constr2[i, j, t].rhs = best_incum[i, j, t]

                    obj = gp.quicksum(
                        mp.getVarByName(f"x_dev[{i},{j},{t}]") for i, j in
                        prod_key for t in period_df.I)
                elif level_obj_norm == 1.5:
                    # ---- update L1 RHS ----
                    for i, j, t in l1_keys:
                        abs_pos[i, j, t].rhs = -best_incum[i, j, t]
                        abs_neg[i, j, t].rhs =  best_incum[i, j, t]

                    # ---- update PWL breakpoints ----
                    for i, j, t in pwl_keys:
                        mp.remove(pwl[i, j, t])

                        z = mp.getVarByName(f"z[{i},{j},{t}]")
                        y = mp.getVarByName(f"y[{i},{j},{t}]")

                        p0 = 0 - best_incum[i, j, t]
                        p2 = max_prod[i, j] / lot_size[i, j] - best_incum[i, j, t]
                        p1 = 0.5 * (p0 + p2)

                        bp = [p0, p1, p2]
                        val = [p**2 for p in bp]

                        pwl[i, j, t] = mp.addGenConstrPWL(z, y, bp, val)
                    
                    obj = gp.quicksum(
                        mp.getVarByName(f"y[{i},{j},{t}]") for i, j in
                        prod_key for t in period_df.I)
                    
                    mp.update()

                mp.setObjective(obj)
                #mp.update()
                mp.optimize()
                RB_ap_end_time = time.time()

                x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
                theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}

                best_incum = copy.deepcopy(x_vals)

            # Generate feasibility & optimality cut simultaneously
            if feas_opt_cut_flag:
                sub_start_time = time.time()
                sub_opt_results = pool.map(partial(pi_iter_sparse, x_vals=x_vals, theta_vals=theta_vals, lag_cuts=False, feas_cuts=True, sparse_cuts=sparse_cut_flag), range(len(decomp_units)))
                sub_time = time.time() - sub_start_time

                ub_cand = 0
                total_HC = 0
                total_PC = 0
                total_TC = 0
                total_DC = 0
                feas_sol_obtained = True
                for item in sub_opt_results:
                    k = item[0]

                    cut_gen = item[1]
                    sub_obj = item[2]
                    pi_obj = item[3]
                    sub_dual = item[4]
                    feas_gen = item[5]
                    feas_dual = item[6]
                    feas_value = item[7]

                    if feas_gen:
                        feas_sol_obtained = False

                    sub_HC = item[8]
                    sub_PC = item[9]
                    sub_TC = item[10]
                    sub_DC = item[11]
                    total_HC += sub_HC
                    total_PC += sub_PC
                    total_TC += sub_TC
                    total_DC += sub_DC

                    ub_cand += sub_obj

                    if feas_gen:
                        mp.addConstr(0 >= feas_value + gp.quicksum(feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                for i, j, t in feas_dual.keys()))

                    if cut_gen:
                        mp.addConstr(mp.getVarByName(f"theta[{k}]") >= pi_obj + gp.quicksum(
                            sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                            for i, j, t in sub_dual.keys()))
                                                
                mp.update()
                if feas_sol_obtained and ub_cand < best_ub:
                    best_ub = ub_cand
                ubList.append(best_ub)
            # Conventional Benders: not generate optimality cut until a feasible solution is obtained
            else:
                sub_start_time = time.time()
                sub_opt_results = pool.map(partial(pi_iter_sparse_feas, x_vals=x_vals, sparse_cuts=sparse_cut_flag), range(len(decomp_units)))
                sub_time = time.time() - sub_start_time
    
                feas_sol_obtained = True   # whether a feasible solution is obtained
                for item in sub_opt_results:
                    k = item[0]

                    feas_gen = item[1]
                    feas_dual = item[2]
                    feas_value = item[3]

                    if feas_gen:
                        feas_sol_obtained = False

                        mp.addConstr(0 >= feas_value + gp.quicksum(feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                for i, j, t in feas_dual.keys()))

                # if a feasible solution is obtained, then generate optimality cut at this solution
                if feas_sol_obtained:
                    sub_start_time = time.time()
                    sub_opt_results = pool.map(partial(pi_iter_sparse_opt, x_vals=x_vals, theta_vals=theta_vals, lag_cuts=False, sparse_cuts=sparse_cut_flag), range(len(decomp_units)))
                    sub_time += time.time() - sub_start_time

                    ub_cand = 0
                    total_HC = 0
                    total_PC = 0
                    total_TC = 0
                    total_DC = 0
                    for item in sub_opt_results:
                        k = item[0]

                        cut_gen = item[1]
                        sub_obj = item[2]
                        pi_obj = item[3]
                        sub_dual = item[4]

                        sub_HC = item[5]
                        sub_PC = item[6]
                        sub_TC = item[7]
                        total_HC += sub_HC
                        total_PC += sub_PC
                        total_TC += sub_TC

                        ub_cand += sub_obj

                        if cut_gen:
                            mp.addConstr(mp.getVarByName(f"theta[{k}]") >= pi_obj + gp.quicksum(
                                sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                for i, j, t in sub_dual.keys()))
                else:
                    ub_cand = math.inf

                    total_HC = math.inf
                    total_PC = math.inf
                    total_TC = math.inf
                    total_DC = math.inf
                
                # upper bound is updated only if a better feasible solution is obtained
                if feas_sol_obtained and ub_cand < best_ub:
                    best_ub = ub_cand
                ubList.append(best_ub)
                mp.update()

            if best_ub < math.inf and best_lb > -math.inf:
                gap = (best_ub - best_lb) / best_ub
            else:
                gap = math.inf

            wr = "\nCurrent Upper Bound: " + str(ub_cand) + "\nCurrent Lower Bound: " + str(lb_cand) + \
                "\nBest Upper Bound: " + str(best_ub) + "\nBest Lower Bound: " + str(best_lb) + \
                "\nFeasible solution obtained? " + str(feas_sol_obtained) 
            if reg_BD_flag:
                wr += "\nMaster & Regularization auxiliary prob time: " + str(round(RB_mp_end_time - RB_mp_start_time, 2)) + ", " + str(round(RB_ap_end_time - RB_ap_start_time, 2))
            else: 
                wr += "\nMaster time: " + str(round(mp_end_time - mp_start_time, 2))
            wr += "\nSub time: " + str(round(sub_time, 2)) + "\n"
            
            wr_s = open(result_out_path, "a")
            wr_s.write(wr)
            wr_s.close()

            if LP_iter_num >= lp_warm_iter or gap <= lp_rel_gap_tol:
                break
            
        wr = "\nLP UB list: " + str(ubList) + "\nLP LB list: " + str(lbList) + "\n"
        wr_s = open(result_out_path, "a")
        wr_s.write(wr)
        wr_s.close()

        # Restore defaults
        best_lb = -math.inf
        best_ub = math.inf
        ubList = []
        lbList = []
        # Begin solving MIP
        for i, j in prod_key:
            for t in period_df.I:
                mp.getVarByName(f"x[{i},{j},{t}]").vtype=GRB.INTEGER
        mp.update()
    time_list.append(time.time() - start)
    
    total_mp_time = 0
    total_subp_time = 0
    best_ub_iter = 0
    while True:
        iter_num += 1

        wr = "\nIteration: " + str(iter_num)
        wr_s = open(result_out_path, "a")
        wr_s.write(wr)
        wr_s.close()
        
        # Fix production variables that remain unchanged for many iterations
        if fixing_flag:
            if iter_num > 1:
                x_vals_pre = iter_x_value.copy()

            counter = 0
            for i, j in prod_key:
                for t in period_df.I:
                    if fixing_round[counter] >= thres_FR:
                        mp.getVarByName(f"x[{i},{j},{t}]").lb = fixing_x[counter]
                        mp.getVarByName(f"x[{i},{j},{t}]").ub = fixing_x[counter]
                        mp.update()
                    counter += 1
            mp.update()

        # Conventional Benders
        if not reg_BD_flag:
            mp_start_time = time.time()
            mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
            mp.update()
            mp.optimize()
            mp_time = time.time() - mp_start_time

            x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
            theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}

            lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))
            lbList.append(lb_cand)
            if lb_cand > best_lb:
                best_lb = lb_cand
        # Regularized Benders
        else:
            # master
            RB_mp_start_time = time.time()
            level_constr.rhs = math.inf
            mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
            mp.update()
            mp.optimize()
            RB_mp_time = time.time() - RB_mp_start_time

            theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}
            lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))
            lbList.append(lb_cand)
            if lb_cand > best_lb:
                best_lb = lb_cand

            # auxiliary problem, refinement to master
            RB_ap_start_time = time.time()
            # update level set auxiliary problem
            if best_ub < math.inf and best_lb > -math.inf:
                level = best_lb + level_fractile * (best_ub - best_lb)
                level_constr.rhs = level

            if level_obj_norm == 2:
                obj = gp.quicksum(
                    (mp.getVarByName(f"x[{i},{j},{t}]") - best_incum[i, j, t]) ** 2 for i, j in prod_key for t in
                    period_df.I)
            elif level_obj_norm == 1:
                for i, j in prod_key:
                    for t in period_df.I:
                        x_dev_constr1[i, j, t].rhs = best_incum[i, j, t]
                        x_dev_constr2[i, j, t].rhs = best_incum[i, j, t]

                obj = gp.quicksum(
                    mp.getVarByName(f"x_dev[{i},{j},{t}]") for i, j in
                    prod_key for t in period_df.I)
            elif level_obj_norm == 1.5:
                # ---- update L1 RHS ----
                for i, j, t in l1_keys:
                    abs_pos[i, j, t].rhs = -best_incum[i, j, t]
                    abs_neg[i, j, t].rhs =  best_incum[i, j, t]

                # ---- update PWL breakpoints ----
                for i, j, t in pwl_keys:
                    mp.remove(pwl[i, j, t])

                    z = mp.getVarByName(f"z[{i},{j},{t}]")
                    y = mp.getVarByName(f"y[{i},{j},{t}]")

                    p0 = 0 - best_incum[i, j, t]
                    p2 = max_prod[i, j] / lot_size[i, j] - best_incum[i, j, t]
                    p1 = 0.5 * (p0 + p2)

                    bp = [p0, p1, p2]
                    val = [p**2 for p in bp]

                    pwl[i, j, t] = mp.addGenConstrPWL(z, y, bp, val)
                
                obj = gp.quicksum(
                    mp.getVarByName(f"y[{i},{j},{t}]") for i, j in
                    prod_key for t in period_df.I)
                
                mp.update()

            mp.setObjective(obj)
            #mp.update()
            mp.optimize()
            RB_ap_time = time.time() - RB_ap_start_time

            x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
            theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}

            best_incum = copy.deepcopy(x_vals)

        # fixing round
        if fixing_flag:
            iter_x_value = [x_vals[i, j, t] for i, j in prod_key for t in period_df.I]

            if iter_num == 1:
                fixing_round = np.array([1 for i, j in prod_key for t in period_df.I])
            else:
                fixing_round = np.multiply(fixing_round, np.array(x_vals_pre) == np.array(iter_x_value)) + 1
            fixing_x = np.array(iter_x_value)
            fixing_ratio = np.count_nonzero(fixing_round >= thres_FR) / len(fixing_round)

        # Generate feasibility & optimality cut simultaneously
        if feas_opt_cut_flag: 
            sub_start_time = time.time()
            sub_opt_results = pool.map(partial(pi_iter_sparse, x_vals=x_vals, theta_vals=theta_vals, lag_cuts=False, feas_cuts=True, sparse_cuts=sparse_cut_flag), range(len(decomp_units)))
            sub_time = time.time() - sub_start_time
            ub_cand = 0
            total_HC = 0
            total_PC = 0
            total_TC = 0
            total_DC = 0
            feas_sol_obtained = True
            for item in sub_opt_results:
                k = item[0]

                cut_gen = item[1]
                sub_obj = item[2]
                pi_obj = item[3]
                sub_dual = item[4]
                feas_gen = item[5]
                feas_dual = item[6]
                feas_value = item[7]

                if feas_gen:
                    feas_sol_obtained = False

                sub_HC = item[8]
                sub_PC = item[9]
                sub_TC = item[10]
                sub_DC = item[11]
                total_HC += sub_HC
                total_PC += sub_PC
                total_TC += sub_TC
                total_DC += sub_DC

                ub_cand += sub_obj

                if feas_gen:
                    if gomory_feas_flag:
                        feas_dual_round = {}
                        for i, j, t in feas_dual.keys():
                            if abs(feas_dual[i, j, t]) > 1e-5:
                                feas_dual_round[i, j, t] = math.floor(feas_dual[i, j, t])
                            else:
                                feas_dual_round[i, j, t] = 0
                        mp.addConstr(math.floor(-feas_value) >= gp.quicksum(feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                for i, j, t in feas_dual.keys()))
                    else:
                        mp.addConstr(0 >= feas_value + gp.quicksum(feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                for i, j, t in feas_dual.keys()))

                if cut_gen:
                    mp.addConstr(mp.getVarByName(f"theta[{k}]") >= pi_obj + gp.quicksum(
                        sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                        for i, j, t in sub_dual.keys()))
            mp.update()
            if feas_sol_obtained and ub_cand < best_ub:
                best_ub = ub_cand
                best_ub_iter = iter_num
            ubList.append(best_ub)
        # Conventional BD: not generate optimality cut until a feasible solution is obtained
        else:
            sub_start_time = time.time()
            sub_opt_results = pool.map(partial(pi_iter_sparse_feas, x_vals=x_vals, sparse_cuts=sparse_cut_flag), range(len(decomp_units)))
            sub_time = time.time() - sub_start_time

            # whether a feasible solution is obtained
            feas_sol_obtained = True
            for item in sub_opt_results:
                k = item[0]

                feas_gen = item[1]
                feas_dual = item[2]
                feas_value = item[3]

                if feas_gen:
                    feas_sol_obtained = False

                    mp.addConstr(0 >= feas_value + gp.quicksum(feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                            for i, j, t in feas_dual.keys()))
            
            # if a feasible solution is obtained, then generate an optimality cut at this solution
            if feas_sol_obtained:
                sub_start_time = time.time()
                sub_opt_results = pool.map(partial(pi_iter_sparse_opt, x_vals=x_vals, theta_vals=theta_vals, lag_cuts=False, sparse_cuts=sparse_cut_flag), range(len(decomp_units)))
                sub_end_time = time.time()
                sub_time += sub_end_time - sub_start_time

                ub_cand = 0
                total_HC = 0
                total_PC = 0
                total_TC = 0
                total_DC = 0
                for item in sub_opt_results:
                    k = item[0]

                    cut_gen = item[1]
                    sub_obj = item[2]
                    pi_obj = item[3]
                    sub_dual = item[4]

                    sub_HC = item[5]
                    sub_PC = item[6]
                    sub_TC = item[7]
                    total_HC += sub_HC
                    total_PC += sub_PC
                    total_TC += sub_TC

                    ub_cand += sub_obj

                    if cut_gen:
                        mp.addConstr(mp.getVarByName(f"theta[{k}]") >= pi_obj + gp.quicksum(
                            sub_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                            for i, j, t in sub_dual.keys()))
            else:
                ub_cand = math.inf

                total_HC = math.inf
                total_PC = math.inf
                total_TC = math.inf
                total_DC = math.inf
            
            # update upper bound only if a better feasible solution is obtained
            if feas_sol_obtained and ub_cand < best_ub:
                best_ub = ub_cand
                best_ub_iter = iter_num
            ubList.append(best_ub)
            mp.update()

        if best_ub < math.inf and best_lb > -math.inf:
            gap = (best_ub - best_lb) / best_ub
        else:
            gap = math.inf
        
        # record time
        time_list.append(time.time() - start)

        wr = "\nCurrent Upper Bound: " + str(ub_cand) + "\nCurrent Lower Bound: " + str(lb_cand) + \
            "\nBest Upper Bound: " + str(best_ub) + "\nBest Lower Bound: " + str(best_lb) + \
            "\nFeasible solution obtained? " + str(feas_sol_obtained) 
        if reg_BD_flag:
            wr += "\nMaster & Regularization auxiliary prob time: " + str(round(RB_mp_time, 2)) + ", " + str(round(RB_ap_time, 2))
            total_mp_time += RB_mp_time + RB_ap_time
        else: 
            wr += "\nMaster time: " + str(round(mp_time, 2))
            total_mp_time += mp_time
        wr += "\nSub time: " + str(round(sub_time, 2)) + "\n"
        total_subp_time += sub_time

        if fixing_flag:
            wr += "Fixing ratio: " + str(round(fixing_ratio * 100, 2)) + "%\n" 
        wr_s = open(result_out_path, "a")
        wr_s.write(wr)
        wr_s.close()

        if iter_num >= max_iter or gap <= rel_gap_tol:
            break
    
    pool.close()
    end = time.time()
    
    wr = f"\nBest UB found in iteration {best_ub_iter}."
    wr += "\nUB list: " + str(ubList) + "\nLB list: " + str(lbList) + "\nTime evolution: " + str(time_list) + "\nTotal Master/Sub prob: " + str((round(total_mp_time, 2), round(total_subp_time, 2)))
    if lp_warm_flag:
        wr += f"\nLP end in {time_list[0]} seconds, MIP BD end in {end - start - time_list[0]} seconds."
    wr += "\n\nTotal BD end in " + str(end - start) + " seconds!"
    wr_s = open(result_out_path, "a")
    wr_s.write(wr)
    wr_s.close()
    