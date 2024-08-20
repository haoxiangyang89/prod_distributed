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

def master_prob(reg_method, level_obj_norm):
    mp = gp.Model("master_prob")
    
    mp.Params.TimeLimit = 10 * 60

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
    item_selected = np.zeros(I_norm)
    for i in item_df.I:
        if item_selected[i] == 0:
            decomp_units.append(decomp_dict[i])
            for item in decomp_dict[i]:
                item_selected[item] = 1

    x = mp.addVars(prod_key, period_df.I, vtype=GRB.INTEGER, lb=0.0, name="x")
    
    theta = mp.addVars(range(len(decomp_units)), vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    
    mp.addConstrs(
        (x[i, j, t] * lot_size[i, j] <= max_prod[i,j] for i, j in prod_key for t in range(T_norm)),
        name='batch')
    mp.addConstrs((gp.quicksum(
        unit_cap[i_iter, j_iter, ct_iter] * x[i_iter, j_iter, t] * lot_size[i_iter, j_iter] for i_iter, j_iter, ct_iter in unit_cap.keys() if (j_iter == j) and (ct == ct_iter) and (i_iter, j_iter) in prod_key) <= max_cap[ct, j, t]
                   for ct, j, t in max_cap.keys()), name='capacity')

    mp.setObjective(gp.quicksum(theta[unit_idx] for unit_idx in range(len(decomp_units))), GRB.MINIMIZE)

    # set up the simple production bounds
    # mp.addConstrs((gp.quicksum(
    #         bom_value[i,ii,j] * x[i,j,tau] * lot_size[i,j] for j in plant_df.I for tau in range(t+1) if (i,j) in prod_key) -
    #         sum(init_inv[ii,j] for j in plant_df.I) + sum(external_purchase[ii, j ,tau] for j in plant_df.I for tau in range(t)) - 
    #         gp.quicksum(x[ii,j,tau] * lot_size[ii,j] for j in plant_df.I for tau in range(t) if (ii,j) in prod_key) <= 0 
    #             for i,ii in bom_list for t in period_df.I), name = "simple") 

    mp.update()

    return mp, [x, theta]

def pi_iter(k, data, x_vals, theta_vals, lag_cuts=True, feas_cuts=True, level_lambda=0.5, level_mu=0.8):
    # Attempt to find the flat constraints with better numerical properties
    penalty_mag = 20 * np.max(list(penalty_cost.values()))
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
    xCi = sp.addVars(prod_key, period_df.I, lb=0.0, name="x")  # x_{ijt} for i,j,t (x copy)
    x_im_p = sp.addVars(prod_key, period_df.I, lb=0.0, name="x_p")  # x_imb_p_{ijt} for i,j,t (x copy)
    x_im_m = sp.addVars(prod_key, period_df.I, lb=0.0, name="x_m")  # x_imb_p_{ijt} for i,j,t (x copy)

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
    for l in transit_list_i.Tr:
        for t in range(min(period_df.I) - int(transit_time[l]), min(period_df.I)):
            si[l,t] = 0.0  # initial transportation set to 0
    for i, j in prod_key:
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
                   (xCi[i, j, t - int(lead_time[i, j])] if (i, j) in prod_key else 0.0) + external_purchase[i, j, t] +
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
    # solve subproblem to generate feasibility cuts
    sub_fix_x = {}
    for i, j in prod_key:
        for t in period_df.I:
            sub_fix_x[i, j, t] = sp.addConstr(xCi[i, j, t] + x_im_p[i,j,t] - x_im_m[i,j,t] == x_vals[i, j, t] * lot_size[i, j])
    if feas_cuts:
        # set up the subproblem specific objective
        obj = gp.quicksum(x_im_p[i, j, t] + x_im_m[i, j, t] for i,j in prod_key for t in period_df.I)
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
          gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key for t in period_df.I)

    sp.setObjective(obj, GRB.MINIMIZE)
    sp.update()
    sp.optimize()

    sub_obj = sp.ObjVal

    #print(k, " ", sub_obj, " ", theta_vals[k])
    if ((sub_obj - theta_vals[k]) > 1e-4) and (sub_obj > 1.00001 * theta_vals[k]):
        cut_gen = True

        dual_coeff = np.zeros((I_norm, J_norm, T_norm))
        for i, j in prod_key:
            for t in period_df.I:
                if abs(sub_fix_x[i, j, t].Pi) > 1e-4:
                    dual_coeff[i, j, t] = sub_fix_x[i, j, t].Pi
                else:
                    dual_coeff[i, j, t] = 0

        iter_num = 0

        if lag_cuts:
            # initialize the pi_hat as 0
            pi_hat = np.zeros((I_norm, J_norm, T_norm))
            best_dual = np.zeros((I_norm,J_norm,T_norm))
            GList = []
            fList = []

            # set up the lower bound problem
            smcp = gp.Model()
            smcp.setParam("OutputFlag",0)
            smcp.setParam("Threads",1)
            pi = smcp.addVars(prod_key, period_df.I, lb=-float('inf'), name="beta")
            pi_abs = smcp.addVars(prod_key, period_df.I, lb=0.0, name="beta_abs")
            smcp.setObjective(gp.quicksum(pi_abs[i,j,t] for i,j in prod_key for t in period_df.I), GRB.MINIMIZE)
            smcp.addConstrs(pi_abs[i,j,t] >= pi[i,j,t] for i,j in prod_key for t in period_df.I)
            smcp.addConstrs(pi_abs[i,j,t] >= -pi[i,j,t] for i,j in prod_key for t in period_df.I)
            smcp.update()
            smcp.optimize()
            current_ub = np.inf
            f_star = smcp.ObjVal

            # set up the level problem
            lp = gp.Model()
            lp.setParam("OutputFlag",0)
            lp.setParam("Threads",1)
            pi_level = lp.addVars(prod_key, period_df.I, lb=-float('inf'), name="beta")
            pi_abs_level = lp.addVars(prod_key, period_df.I, lb=0.0, name="beta_abs")
            theta_lp = lp.addVar(name = "theta")
            lp.addConstrs(pi_abs_level[i,j,t] >= pi_level[i,j,t] for i,j in prod_key for t in period_df.I)
            lp.addConstrs(pi_abs_level[i,j,t] >= -pi_level[i,j,t] for i,j in prod_key for t in period_df.I)

            for i, j in prod_key:
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
                    gp.quicksum(penalty_mag * (x_im_p[i, j, t] + x_im_m[i, j, t]) for i,j in prod_key for t in period_df.I) + \
                    gp.quicksum(pi_hat[i,j,t] * (x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t] - x_im_p[i,j,t] + x_im_m[i,j,t]) for i, j in prod_key for t in period_df.I)
                    # gp.quicksum(
                    #     penalty_mag * (zi_m[i, j, t] + yUOi_m[i, j, t]) for i in unit_ind_list for j in plant_df.I for t in period_df.I) + \

                sp.setObjective(obj, GRB.MINIMIZE)
                sp.update()
                sp.optimize()

                pi_obj = sp.ObjVal
                fList.append(np.sum(abs(pi_hat)))
                if iter_num == 1:
                    best_dual = copy.deepcopy(pi_hat)
                elif sub_obj - pi_obj < np.min(GList):
                    best_dual = copy.deepcopy(pi_hat)
                GList.append(sub_obj - pi_obj)

                # if we have obtain a good solution, stop
                if sub_obj - pi_obj <= 0.01 * sub_obj:
                    keep_iter = False
                else:
                    x_grad_vals = np.zeros((I_norm, J_norm, T_norm))
                    for i, j in prod_key:
                        for t in period_df.I:
                            x_grad_vals[i,j,t] = x_vals[i, j, t] * lot_size[i, j] - xCi[i, j, t].X  - x_im_p[i,j,t].X + x_im_m[i,j,t].X

                    # adding the cuts for the Lagrangian function
                    smcp.addConstr(sub_obj - pi_obj - gp.quicksum(x_grad_vals[i,j,t] * (pi[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key for t in period_df.I) <= 0)
                    smcp.update()
                    smcp.optimize()
                    f_star = smcp.ObjVal
                    lp.addConstr(theta_lp >= sub_obj - pi_obj - gp.quicksum(x_grad_vals[i,j,t] * (pi_level[i,j,t] - pi_hat[i,j,t]) for i,j in prod_key for t in period_df.I))

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
                    level_con = lp.addConstr(alpha * gp.quicksum(pi_abs_level[i,j,t] for i,j in prod_key for t in period_df.I) + (1 - alpha) * theta_lp <= level)
                    lp.setObjective(gp.quicksum((pi_level[i,j,t] - pi_hat[i,j,t]) ** 2 for i,j in prod_key for t in period_df.I))

                    # generate the next pi
                    lp.update()
                    lp.optimize()
                    for i,j in prod_key:
                        for t in period_df.I:
                            pi_hat[i,j,t] = pi_level[i,j,t].X
                    lp.remove(level_con)

            for i,j in prod_key:
                for t in period_df.I:
                    if abs(best_dual[i,j,t]) < 1e-6: 
                        pi_hat[i,j,t] = 0
                    else:
                        pi_hat[i,j,t] = best_dual[i,j,t]

        else:
            pi_hat = dual_coeff

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
        pi_hat = 0
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
    max_prod = dict(zip(prod_key, df_prod.MaxProd))
    unit_cap = dict(zip(list(zip(df_unitCap.I, df_unitCap.J, df_unitCap.Ct)), df_unitCap.V))
    max_cap = dict(zip(list(zip(df_maxCap.Ct, df_maxCap.J, df_maxCap.Ti)), df_maxCap.V))

    global lead_time, real_demand, external_purchase, init_inv, holding_cost, penalty_cost, transit_cost, transit_time 
    lead_time = np.zeros((I_norm, J_norm))
    lead_time[df_prod.I, df_prod.J] = df_prod.LT
    real_demand = np.zeros((I_norm,T_norm))
    real_demand[df_demand.I, df_demand.Ti] = df_demand.V
    external_purchase_key = list(zip(df_external.I, df_external.J, df_external.Ti))
    external_purchase = dict(zip(external_purchase_key, df_external.V))
    init_inv = np.zeros((I_norm,J_norm))
    init_inv[df_iniInv.I, df_iniInv.J] = df_iniInv.V
    holding_cost = dict(zip(df_cost.I, df_cost.HC))
    penalty_cost = dict(zip(df_cost.I, df_cost.PC))
    transit_cost = dict(zip(df_transitTC.Tr, df_transitTC.TC))
    transit_time = dict(zip(df_transitTC.Tr, df_transitTC.V))
    
    # item clustering such that items with substitutution relationship are in the same cluster
    global alt_item_pair_list, decomp_units, bom_list
    alt_item_pair_list = list(zip(df_alt.I, df_alt.II))
    alt_item_pair_list = [tuple(sorted(tuple_)) for tuple_ in alt_item_pair_list]
    alt_item_pair_list = list(dict.fromkeys(alt_item_pair_list))
    
    decomp_dict = {i:[i] for i in item_df.I}
    for alt_pair in alt_item_pair_list:
        joined_list = list(set(decomp_dict[alt_pair[0]] + decomp_dict[alt_pair[1]]))
        for item in joined_list:
            decomp_dict[item] = joined_list
    decomp_units = []
    item_selected = np.zeros(I_norm)
    for i in item_df.I:
        if item_selected[i] == 0:
            decomp_units.append(decomp_dict[i])
            for item in decomp_dict[i]:
                item_selected[item] = 1
    
    bom_pair = df_bom[["I","II"]].drop_duplicates()
    bom_list = list(zip(bom_pair.I, bom_pair.II))

if __name__ == "__main__":
    # read in data from specified path
    data_folder = "./data/fine_tune"  
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
        
    # initialize the Benders procedure
    max_iter = 500
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

    # set up the master problem (model (4))
    mp, mp_handle = master_prob(reg_method, level_obj_norm)

    start = time.time()
    # initiate the data in the master process
    worker_init(gbv)
    mp.update()

    ubList = []
    lbList = []
    # You can change the number of threads to use here
    # initiate the data in the parallel sessions
    pool = Pool(6, initializer=worker_init, initargs=(gbv,))

    while True:
        iter_num += 1

        wr = "\nIteration: " + str(iter_num)
        wr_s = open("benders.txt", "a")
        wr_s.write(wr)
        wr_s.close()

        # Currently using multi-cut, no level set in the master level
        mp.setObjective(gp.quicksum(mp.getVarByName(f"theta[{i}]") for i in range(len(decomp_units))))
        mp.update()
        mp.optimize()

        x_vals = {xKey:mp_handle[0][xKey].X for xKey in mp_handle[0].keys()}
        theta_vals = {thetaKey: mp_handle[1][thetaKey].X for thetaKey in mp_handle[1].keys()}
        lb_cand = sum(theta_vals[k] for k in range(len(decomp_units)))

        lbList.append(lb_cand)
        if lb_cand > best_lb:
            best_lb = lb_cand

        if iter_num % 1 == 0:
            wr = "\nMaster x_val:"
            for i, j in prod_key:
                for t in period_df.I:
                    wr += str((i, j, t)) + " : " + str(x_vals[i, j, t]) + "\t,"
            wr += "\nMaster theta_val:"
            for k in range(len(decomp_units)):
                wr += str(k) + " : " + str(theta_vals[k]) + ",\t"

        # subproblem solution given a production level

        # solve the subproblems in parallel to generate feasibility/optimality cut
        # here you should use lag_cuts=False, feas_cuts=True
        sub_opt_results = pool.imap(partial(pi_iter, data=gbv, x_vals=x_vals, theta_vals=theta_vals, lag_cuts=False, feas_cuts=True), range(len(decomp_units)))
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
                # append the feasibility cuts
                mp.addConstr(0 >= feas_value + gp.quicksum(
                    feas_dual[i, j, t] * (mp.getVarByName(f"x[{i},{j},{t}]") - x_vals[i, j, t])
                    for i, j in prod_key for t in period_df.I))
                mp.update()

            if cut_gen:
                # append the optimality cuts
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
            
        mp.update()

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
    pool.close()