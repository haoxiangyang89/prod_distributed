import os
print(os.getcwd())
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

import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import pickle
#from scipy.sparse import csr_array, coo_array
import scipy.sparse

import itertools
import copy
import csv

def master_prob(relax_option=False, relative_gap=0.01):
    mp = gp.Model("master_prob")
    
    #mp.Params.TimeLimit = 10 * 60
    mp.Params.MIPGap = relative_gap

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

    # set up the simple production bounds
    # mp.addConstrs((gp.quicksum(
    #         bom_value[i,ii,j] * x[i,j,tau] * lot_size[i,j] for j in plant_df.I for tau in range(t+1) if (i,j) in prod_key) -
    #         sum(init_inv[ii,j] for j in plant_df.I) + sum(external_purchase[ii, j ,tau] for j in plant_df.I for tau in range(t)) - 
    #         gp.quicksum(x[ii,j,tau] * lot_size[ii,j] for j in plant_df.I for tau in range(t) if (ii,j) in prod_key) <= 0 
    #             for i,ii in bom_list for t in period_df.I), name = "simple") 

    mp.update()

    return mp, [x, theta]

def pi_iter_sparse(k, x_vals, theta_vals, lag_cuts=True, feas_cuts=True, level_lambda=0.5, level_mu=0.8):
    # Attempt to find the flat constraints with better numerical properties
    #worker_init(data)

    penalty_mag = len(period_df.I) * np.max(list(penalty_cost.values()))
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

    # find the related items and create x copies for only them
    item_used_list = []
    for i in unit_ind_list:
        item_used_list.append(i)
        for bk in df_bom.index[df_bom.II == i]:
            item_used_list.append(df_bom.I[bk])
    df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
    prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
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

    # item clustering such that items with substitutution relationship are in the same cluster
    global alt_item_pair_list, decomp_units, bom_list
    alt_item_pair_list = list(zip(df_alt.I, df_alt.II))
    alt_item_pair_list = [tuple(sorted(tuple_)) for tuple_ in alt_item_pair_list]
    alt_item_pair_list = list(dict.fromkeys(alt_item_pair_list))

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
    # decomp_units = [list(alt_pair) for alt_pair in alt_item_pair_list]
    # for i in list(item_df.I):
    #     if i not in alt_items_list:
    #         decomp_units.append([i])

    bom_pair = df_bom[["I", "II"]].drop_duplicates()
    bom_list = list(zip(bom_pair.I, bom_pair.II))

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

def sub_fix_x(k, x_vals):
    # Attempt to find the flat constraints with better numerical properties
    # worker_init(data)
    unit_ind_list = decomp_units[k]

    transit_list_i = df_transit[df_transit.I.isin(unit_ind_list)]
    alt_i = df_alt[df_alt.I.isin(unit_ind_list) | df_alt.II.isin(unit_ind_list)]

    # print(multiprocessing.current_process(), "\tSolving Subproblem " + str(unit_ind_list) + "!\n")

    sp = gp.Model("item_{}".format(unit_ind_list))
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    sp.setParam("Threads", 1)
    # sp.Params.TimeLimit = 5 * 60

    # set up model parameters (M: plant, T: time, L: transit,
    ui = sp.addVars(unit_ind_list, period_df.I, lb=0.0, name="u")  # u_{it} for t, unmet demand

    si = sp.addVars(transit_list_i.Tr, period_df.I, lb=0.0, name="s")  # s_{ilt} for l,t

    zi_p = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="z_p")  # z^+_{ijt} for j,t

    vi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="v")  # v_{ijt} for j,t

    yUIi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yi")  # y^{I}_{ijt} for j,t
    yUOi = sp.addVars(unit_ind_list, plant_df.I, period_df.I, lb=0.0, name="yo_p")  # y^{O,+}_{ijt} for j,t

    # find the related items and create x copies for only them
    item_used_list = []
    for i in unit_ind_list:
        item_used_list.append(i)
        for bk in df_bom.index[df_bom.II == i]:
            item_used_list.append(df_bom.I[bk])
    df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
    prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
    xCi = sp.addVars(prod_key_used, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)

    # initial condition setup
    for i in unit_ind_list:
        ui[i, min(period_df.I) - 1] = 0.0  # initial unmet demand set to 0
    for i in unit_ind_list:
        for j in plant_df.I:
            if (i, j) in init_inv.keys():
                vi[i, j, min(period_df.I) - 1] = init_inv[i, j]  # initial inventory set to given values
            else:
                vi[i, j, min(period_df.I) - 1] = 0.0
            for t in period_df.I:
                if not ((i, j, t) in external_purchase.keys()):
                    external_purchase[i, j, t] = 0.0
        for t in period_df.I:
            if not ((i, t) in real_demand.keys()):
                real_demand[i, t] = 0.0
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - int(transit_time[l]), min(period_df.I)):
            si[l, t] = 0.0  # initial transportation set to 0
    for i, j in prod_key_used:
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
        si[transit_list_i.Tr[l], t - int(transit_time[l])] for l in
        transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.JJ == j)]) +
                   (xCi[i, j, t - int(lead_time[i, j])] if (i, j) in prod_key_used else 0.0) + external_purchase[
                       i, j, t] +
                   gp.quicksum(alt_i.V[jta] * rCi[jta] for jta in
                               alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.II == i)]) \
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name='input_item')

    sp.addConstrs((yUOi[i, j, t] == gp.quicksum(
        si[transit_list_i.Tr[l], t] for l in transit_list_i.index[(transit_list_i.I == i) & (transit_list_i.J == j)]) +
                   gp.quicksum(
                       df_bom.V[bk] * xCi[df_bom.I[bk], j, t] for bk in df_bom.index[(df_bom.J == j) & (df_bom.II == i)]
                       if (df_bom.I[bk], j) in prod_key_used) + zi_p[i, j, t] +
                   gp.quicksum(rCi[jta] for jta in
                               alt_i.index[(alt_i.J == j) & (alt_i.Ti == t) & (alt_i.I == i)])
                   for i in unit_ind_list for j in plant_df.I for t in period_df.I),
                  name="output_item")

    sp.addConstrs(
        (xCi[i, j, t] <= max_prod[i, j] for i, j in prod_key_used for t in period_df.I),
        name='batch')
    sp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * xCi[i_iter, j_iter, t] for i_iter, j_iter, ct_iter in unit_cap.keys() if
        (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key_used) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    sp.addConstrs((rCi[jta] <= vi[alt_i.I[jta], alt_i.J[jta], alt_i.Ti[jta] - 1] for jta in alt_i.index), name='r_ub')

    sp.addConstrs((yUOi[i, j, t] <= vi[i, j, t - 1] for i in unit_ind_list
                   for j in plant_df.I for t in period_df.I),
                  name='yo_ub')

    # fix the local variables to their global counterparts' values
    sp.addConstrs(
        xCi[i, j, t] == x_vals[i, j, t] * lot_size[i, j] for i, j in prod_key_used for t in period_df.I)

    # obtain the LP objective value of the subproblem
    obj = gp.quicksum(
        holding_cost[i] * vi[i, j, t] for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \
          gp.quicksum(penalty_cost[i] * ui[i, t] for i in unit_ind_list for t in period_df.I) + \
          gp.quicksum(
              transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t] for l in transit_list_i.index for t in
              period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    sub_obj = sp.ObjVal


    for i in unit_ind_list:
        hc_i = sum(holding_cost[i] * vi[i, j, t].X for j in plant_df.I for t in period_df.I)
        pc_i = sum(penalty_cost[i] * ui[i, t].X for t in period_df.I)
        tc_i = sum(
        transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t].X for l in transit_list_i.index[transit_list_i.I == i] for t in
        period_df.I)
        if max(hc_i, pc_i, tc_i) > 1e-2:
            print(item_df.index[i], "hc/pc/tc: ", hc_i, pc_i, tc_i)

    sub_HC = sum(holding_cost[i] * vi[i, j, t].X for i in unit_ind_list for j in plant_df.I for t in period_df.I)
    sub_PC = sum(penalty_cost[i] * ui[i, t].X for i in unit_ind_list for t in period_df.I)
    sub_TC = sum(
        transit_cost[transit_list_i.Tr[l]] * si[transit_list_i.Tr[l], t].X for l in transit_list_i.index for t in
        period_df.I)

    # write inventory evolution into csv
    with open("cand_inv.csv", 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the data
        for i in unit_ind_list:
            for j in plant_df.I:
                max_inv_through = max(max(vi[i, j, t].X for t in period_df.I), init_inv.get((i, j), 0))
                if max_inv_through > 1e-2:
                    # initial inventory
                    data_output = [item_df.index[i], plant_df.index[j], 0, init_inv.get((i, j), 0)]
                    writer.writerow(data_output)
                    # inventory evolution
                    for t in period_df.I:
                        data_output = [item_df.index[i], plant_df.index[j], period_df.index[t], vi[i, j, t].X]
                        writer.writerow(data_output)

    # write unused inventory into csv
    with open("unused_inv.csv", 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in unit_ind_list:
            for j in plant_df.I:
                min_inv_through = min(min(vi[i, j, t].X for t in period_df.I), init_inv.get((i, j), 0))
                if min_inv_through > 1e-2:
                    data_output = [plant_df.index[j], min_inv_through, item_df.index[i]]
                    writer.writerow(data_output)


    return k, sub_HC, sub_PC, sub_TC

def pi_iter_sparse_feas(k, x_vals):
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

    # find the related items and create x copies for only them
    item_used_list = []
    for i in unit_ind_list:
        item_used_list.append(i)
        for bk in df_bom.index[df_bom.II == i]:
            item_used_list.append(df_bom.I[bk])
    df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
    prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
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

def pi_iter_sparse_opt(k, x_vals, theta_vals, lag_cuts=True, level_lambda=0.5, level_mu=0.8):
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
   
    # find the related items and create x copies for only them
    item_used_list = []
    for i in unit_ind_list:
        item_used_list.append(i)
        for bk in df_bom.index[df_bom.II == i]:
            item_used_list.append(df_bom.I[bk])
    df_prod_used = df_prod[df_prod.I.isin(item_used_list)]
    prod_key_used = list(zip(df_prod_used.I, df_prod_used.J))
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
    #data_folder = "/home/xingyulin/Decomposition_modular_2023_07/test_dataset/medium_D0_0418"
    # data_folder = "../data/fine_tune"  # data path
    data_folder = "../data/small_test_fine_tune"  # data path
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
    
    # Accelerate technique 4: Add logical constraint to master
    logic_constr_flag = True

    # fixing technique (Fix production variables that remain unchanged for many iterations)
    fixing_flag = False
    if fixing_flag:
        fixing_x = np.array([0 for i, j in prod_key for t in period_df.I])
        fixing_round = np.array([0 for i, j in prod_key for t in period_df.I])
        thres_FR = 50

    # gomory feasibility cut technique
    gomory_feas_flag = False
    
    # print out result
    result_out_path = "BD.txt"
    # time after each iteration
    time_list = []  

    start = time.time()
    # initialize the master problem
    mp, mp_handle = master_prob(relax_option=False, relative_gap=rel_gap_tol)
    mp.Params.LazyConstraints = 1

    pool = Pool(30, initializer=worker_init, initargs=(gbv,))
    
    # no Lp warm start, no regularization
    mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
    mp.update()
    sol_list = []

    # set up the callback function for the master problem
    def bd_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # get the current solution
            x_vals = {(i,j,t):model.cbGetSolution(model.getVarByName(f"x[{i},{j},{t}]")) for i,j in prod_key for t in gbv["period_df"].I}
            theta_vals = {i: model.cbGetSolution(model.getVarByName(f"theta[{i}]")) for i in range(len(decomp_units))}
            sol_list.append(x_vals)
            # solve the subproblems in parallel
            sub_opt_results = pool.map(partial(pi_iter_sparse, x_vals=x_vals, theta_vals=theta_vals, lag_cuts=False, feas_cuts=True), range(len(decomp_units)))

            # process the results and generate Benders cuts
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
                    # generate Lazy feasibility cut
                    model.cbLazy(0 >= feas_value + gp.quicksum(feas_dual[i, j, t] * (model.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                                for i, j, t in feas_dual.keys()))
                if cut_gen:
                    # generate Lazy optimality cut
                    model.cbLazy(model.getVarByName(f"theta[{k}]") >= pi_obj + gp.quicksum(
                        sub_dual[i, j, t] * (model.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                        for i, j, t in sub_dual.keys()))

    # solve the master problem with the callback function
    mp.optimize(bd_callback)
    pool.close()

    x_best = [mp.getVarByName(f"x[{i},{j},{t}]").X for i,j in prod_key for t in gbv["period_df"].I]
    with open('final_sol.pkl', 'wb') as wp:
        pickle.dump(x_best, wp, protocol=pickle.HIGHEST_PROTOCOL)

    # wr = "\nUB list: " + str(ubList) + "\nLB list: " + str(lbList) + "\nTime evolution: " + str(time_list)
    # if lp_warm_flag:
    #     wr += "\nMIP BD end in " + str(end - end_lp) + " seconds!"
    # wr += "\n\nTotal BD end in " + str(end - start) + " seconds!"
    # wr_s = open(result_out_path, "a")
    # wr_s.write(wr)
    # wr_s.close()