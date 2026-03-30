# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:58:01 2022

@author: haoxiang
"""

'''
input the data of production planning problem
'''

import pandas as pd
import numpy as np
import copy
import os
from GlobalVariable import *

# We need data structure for items, plants, calendar

def readin_csv(file_add):
    return pd.read_csv(file_add)

#%%
def data_proc(filepath):
    # create the data handle
    opt_data = dict()

    # ep.holding_cost = op.Param(ep.I, name = "holding_cost")
    df_cost = pd.DataFrame(data = {'I': [i for i in range(I_norm)],
                                   'HC': [gbv.holding_cost[gbv.item_list[i]] for i in range(I_norm)],
                                   'PC': [gbv.penalty_cost[gbv.item_list[i]] for i in range(I_norm)]})
    opt_data['holding_cost'] = op.df_to_coo(df = df_cost,
                                           index_columns = ['I', ],
                                           data_column = 'HC',
                                           shape = (I_norm, ))
    
    # ep.penalty_cost = op.Param(ep.I, name = "penalty_cost")
    opt_data['penalty_cost'] = op.df_to_coo(df = df_cost,
                                           index_columns = ['I', ],
                                           data_column = 'PC',
                                           shape = (I_norm, ))

    I_norm = len(gbv.item_list)
    J_norm = len(gbv.plant_list)
    T_norm = len(gbv.period_list)
    Tr_norm = len(gbv.transit_list)
    Ct_norm = len(gbv.cap_list)

    # ep.real_demand = op.Param(ep.I, ep.T, name="demand")
    itList = [(i, t) for (i, t) in gbv.real_demand.keys()]
    df_demand = pd.DataFrame(data = {'I': [gbv.item_list.index(it[0]) for it in itList],
                                     'T': [gbv.period_list.index(it[1]) for it in itList],
                                     'V': [gbv.real_demand[it[0], it[1]] for it in itList]})
    opt_data['real_demand'] = op.df_to_coo(df = df_demand,
                                           index_columns = ['I', 'T'],
                                           data_column = 'V',
                                           shape = (I_norm, T_norm))
    
    # ep.external_purchase = op.Param(ep.I, ep.J, ep.T, name = "external_purchase")
    ijtList = [(i, j, t) for (i, j, t) in gbv.external_purchase.keys()]
    df_external = pd.DataFrame(data = {'I': [gbv.item_list.index(it[0]) for it in ijtList],
                                     'J': [gbv.plant_list.index(it[1]) for it in ijtList],
                                     'T': [gbv.period_list.index(it[2]) for it in ijtList],
                                     'V': [gbv.external_purchase[it[0], it[1], it[2]] for it in ijtList]})
    opt_data['external_purchase'] = op.df_to_coo(df = df_external,
                                           index_columns = ['I', 'J', 'T'],
                                           data_column = 'V',
                                           shape = (I_norm, J_norm, T_norm))
    
    # ep.transit_time = op.Param(ep.Tr, name = "transit_time")
    df_transitTC = pd.DataFrame(data = {'Tr': [tr for tr in range(Tr_norm)],
                                      'V': [gbv.transit_time[gbv.transit_list[tr]] for tr in range(Tr_norm)],
                                      'TC': [gbv.transit_cost[gbv.transit_list[tr]] for tr in range(Tr_norm)]})
    opt_data['transit_time'] = op.df_to_coo(df = df_transitTC,
                                           index_columns = ['Tr',],
                                           data_column = 'V',
                                           shape = (Tr_norm,))
    
    # ep.transit_cost = op.Param(ep.Tr, name = "transit_cost")
    opt_data['transit_cost'] = op.df_to_coo(df = df_transitTC,
                                        index_columns = ['Tr',],
                                        data_column = 'TC',
                                        shape = (Tr_norm,))

    # ep.transit_data = op.Param(ep.I, ep.J, ep.Tr, name = "transit_data")
    df_transit = pd.DataFrame(data = {'I': [gbv.item_list.index(gbv.transit_list[tr][0]) for tr in range(Tr_norm)],
                                     'J': [gbv.plant_list.index(gbv.transit_list[tr][2]) for tr in range(Tr_norm)],
                                     'Tr': [tr for tr in range(Tr_norm)],
                                     'V': [1 for tr in range(Tr_norm)]})
    opt_data['transit_data'] = op.df_to_coo(df = df_transit,
                                           index_columns = ['I', 'J', 'Tr'],
                                           data_column = 'V',
                                           shape = (I_norm, J_norm, Tr_norm))
    
    # ep.r_transit_data = op.Param(ep.I, ep.J, ep.Tr, name = "r_transit_data")
    df_rtransit = pd.DataFrame(data = {'I': [gbv.item_list.index(gbv.transit_list[tr][0]) for tr in range(Tr_norm)],
                                     'J': [gbv.plant_list.index(gbv.transit_list[tr][1]) for tr in range(Tr_norm)],
                                     'Tr': [tr for tr in range(Tr_norm)],
                                     'V': [1 for tr in range(Tr_norm)]})
    opt_data['r_transit_data'] = op.df_to_coo(df = df_rtransit,
                                           index_columns = ['I', 'J', 'Tr'],
                                           data_column = 'V',
                                           shape = (I_norm, J_norm, Tr_norm))
    
    # ep.lead_time = op.Param(ep.I, ep.J, name = "lead_time")
    df_prod = pd.DataFrame(data = {'I': [gbv.item_list.index(it[0]) for it in gbv.prod_key],
                                    'J': [gbv.item_list.index(it[1]) for it in gbv.prod_key],
                                    'LT': [gbv.lead_time[it[0], it[1]] for it in gbv.prod_key],
                                    'pKey': [1 for it in gbv.prod_key],
                                    'MaxProd': [gbv.max_prod[it[0], it[1]] for it in gbv.prod_key],
                                    'LS': [gbv.lot_size[it[0], it[1]] for it in gbv.prod_key]})
    opt_data['lead_time'] = op.df_to_coo(df = df_prod,
                                           index_columns = ['I', 'J'],
                                           data_column = 'LT',
                                           shape = (I_norm, J_norm))
    
    # ep.prod_key = op.Param(ep.I, ep.J, name = "prod_key")
    opt_data['prod_key'] = op.df_to_coo(df = df_prod,
                                           index_columns = ['I', 'J'],
                                           data_column = 'pKey',
                                           shape = (I_norm, J_norm))
    
    # ep.lot_size = op.Param(ep.I, ep.J, name = "lot_size")
    opt_data['lot_size'] = op.df_to_coo(df = df_prod,
                                           index_columns = ['I', 'J'],
                                           data_column = 'LS',
                                           shape = (I_norm, J_norm))
    
    # ep.max_prod = op.Param(ep.I, ep.J, name = "max_prod")
    opt_data['max_prod'] = op.df_to_coo(df = df_prod,
                                           index_columns = ['I', 'J'],
                                           data_column = 'MaxProd',
                                           shape = (I_norm, J_norm))
    
    # ep.bom_dict = op.Param(ep.I, ep.I, ep.J, name = "bom_dict")
    bom_keys = [(bk[0], bk[1], j) for j in gbv.plant_list for bk in gbv.bom_key[j]]
    df_bom = pd.DataFrame(data = {'II': [gbv.item_list.index(it[0]) for it in bom_keys],
                                  'I': [gbv.item_list.index(it[1]) for it in bom_keys],
                                  'J': [gbv.item_list.index(it[2]) for it in bom_keys],
                                  'V': [gbv.bom_dict[it[2]][(it[0], it[1])] for it in bom_keys]})
    opt_data['bom_dict'] = op.df_to_coo(df = df_bom,
                                           index_columns = ['II', 'I', 'J'],
                                           data_column = 'V',
                                           shape = (I_norm, I_norm, J_norm))
    
    # ep.alt_dict = op.Param(ep.I, ep.I, ep.J, ep.T, name = "alt_dict")
    df_alt = pd.DataFrame(data = {'I': [gbv.item_list.index(jta[2][0]) for jta in gbv.alt_list],
                                  'II': [gbv.item_list.index(jta[2][1]) for jta in gbv.alt_list],
                                  'J': [gbv.plant_list.index(jta[0]) for jta in gbv.alt_list],
                                  'T': [gbv.period_list.index(jta[1]) for jta in gbv.alt_list],
                                  'V': [gbv.alt_dict[jta] for jta in gbv.alt_list]})
    opt_data['alt_dict'] = op.df_to_coo(df = df_alt,
                                           index_columns = ['I', 'II', 'J', 'T'],
                                           data_column = 'V',
                                           shape = (I_norm, I_norm, J_norm, T_norm))
    
    # ep.unit_cap = op.Param(ep.I, ep.J, ep.Ct, name = "unit_cap")
    itList = [item for item in gbv.unit_cap.keys()]
    df_unitCap = pd.DataFrame(data = {'I': [gbv.item_list.index(it[0]) for it in itList],
                                     'J': [gbv.plant_list.index(it[1]) for it in itList],
                                     'Ct': [gbv.cap_list.index(it[2]) for it in itList],
                                     'V': [gbv.unit_cap[it] for it in itList]})
    opt_data['unit_cap'] = op.df_to_coo(df = df_unitCap,
                                           index_columns = ['I', 'J', 'Ct'],
                                           data_column = 'V',
                                           shape = (I_norm, J_norm, Ct_norm))

    # ep.max_cap = op.Param(ep.J, ep.Ct, name = "max_cap")
    itList = [item for item in gbv.max_cap.keys()]
    df_maxCap = pd.DataFrame(data = {'J': [gbv.plant_list.index(it[1]) for it in itList],
                                     'Ct': [gbv.cap_list.index(it[0]) for it in itList],
                                     'V': [gbv.max_cap[it] for it in itList]})
    opt_data['max_cap'] = op.df_to_coo(df = df_maxCap,
                                           index_columns = ['J', 'Ct'],
                                           data_column = 'V',
                                           shape = (J_norm, Ct_norm))

    return opt_data


#%%
def read_tables(path_file):
    gbv = GlobalVariable()

    # readin all tables
    # item related data
    gbv.item_data = readin_csv(os.path.join(path_file, "dm_df_item.csv"))
    gbv.set_data = readin_csv(os.path.join(path_file, "dm_df_set.csv"))
    gbv.item_set_data = readin_csv(os.path.join(path_file, "dm_df_item_set.csv"))
    gbv.alter_data = readin_csv(os.path.join(path_file, "dm_df_alternate_item.csv"))
    gbv.unit_cap_data = readin_csv(os.path.join(path_file, "dm_df_unit_capacity.csv"))
    gbv.item_plant_data = readin_csv(os.path.join(path_file, "dm_df_item_plant.csv"))
    gbv.bom_data = readin_csv(os.path.join(path_file, "dm_df_bom.csv"))
    gbv.production_data = readin_csv(os.path.join(path_file, "dm_df_production.csv"))
    # plant related data
    gbv.plant_data = readin_csv(os.path.join(path_file, "dm_df_plant.csv"))
    gbv.cap_data = readin_csv(os.path.join(path_file, "dm_df_max_capacity.csv"))
    gbv.transit_data = readin_csv(os.path.join(path_file, "dm_df_transit.csv"))
    gbv.init_inv_data = readin_csv(os.path.join(path_file, "dm_df_inv.csv"))
    # period related data
    gbv.periods_data = readin_csv(os.path.join(path_file, "dm_df_periods.csv"))
    gbv.po_data = readin_csv(os.path.join(path_file, "dm_df_po.csv"))
    gbv.demand_data = readin_csv(os.path.join(path_file, "dm_df_demand.csv"))

    return gbv

#%%
# Make necessary adjustment to the data

def timeline_adjustment(gbv):
    # adjust the gbv's data time to start from 1 with a difference of 1
    time_start = min(gbv.period_list)
    time_end = max(gbv.period_list)
    time_sorted = sorted(gbv.period_list)

    # set up the new list of periods and construct a dictionary for period lookup
    period_list = [t for t in range(1, time_end - time_start + 2)]
    period_dict = {}
    for t in time_sorted:
        period_dict[t] = t - time_start + 1
    gbv.period_list = period_list
    gbv.period_dict = period_dict
    gbv.T = len(gbv.period_list)

    # change the data in gbv which contains the time aspect
    # cap_period_list
    cap_period_list = [gbv.period_dict[t] for t in gbv.cap_period_list]
    gbv.cap_period_list = cap_period_list
    # maximum capacity
    max_cap = {}
    for ckey in gbv.max_cap.keys():
        max_cap[ckey] = {}
        for t in gbv.max_cap[ckey].keys():
            max_cap[ckey][gbv.period_dict[t]] = gbv.max_cap[ckey][t]
    gbv.max_cap = max_cap

    # alternative list
    alt_list = []
    for item in gbv.alt_list:
        if not((item[0], gbv.period_dict[item[1]], item[2]) in alt_list):
            alt_list.append((item[0], gbv.period_dict[item[1]], item[2]))
    alt_dict = {}
    alt_cost = {}
    for akey in gbv.alt_dict.keys():
        akey_update = (akey[0], gbv.period_dict[akey[1]], akey[2])
        alt_dict[akey_update] = gbv.alt_dict[akey]
        alt_cost[akey_update] = gbv.alt_cost[akey]
    gbv.alt_list = alt_list
    gbv.alt_dict = alt_dict
    gbv.alt_cost = alt_cost

    # demand
    real_demand = {}
    forecast_demand = {}
    for dkey in gbv.real_demand.keys():
        dkey_update = (dkey[0], gbv.period_dict[dkey[1]])
        real_demand[dkey_update] = gbv.real_demand[dkey]
        forecast_demand[dkey_update] = gbv.forecast_demand[dkey]
    gbv.real_demand = real_demand
    gbv.forecast_demand = forecast_demand

    # external purchase
    external_purchase = {}
    for pkey in gbv.external_purchase:
        pkey_update = (pkey[0],pkey[1],gbv.period_dict[pkey[2]])
        external_purchase[pkey_update] = gbv.external_purchase[pkey]
    gbv.external_purchase = external_purchase

    return gbv

def cap_adjustment(gbv):
    unit_cap = {}
    item_plant_set_list = {}
    for skey in gbv.set_dict.keys():
        unit_cap[skey[0],skey[1],gbv.set_dict[skey]] = gbv.unit_cap_set[skey[0],skey[2]]
        if (skey[0], skey[1]) in item_plant_set_list.keys():
            item_plant_set_list[skey[0], skey[1]].append(gbv.set_dict[skey])
        else:
            item_plant_set_list[skey[0], skey[1]] = [gbv.set_dict[skey]]

    gbv.unit_cap = unit_cap
    gbv.item_plant_set_list = item_plant_set_list
    return gbv

def obtain_bounds(gbv):
    # obtain the upper bound for an item at all locations without production
    X_list = {}
    for i in gbv.item_list:
        X_list[i] = sum([gbv.init_inv[i,j] for j in gbv.plant_list if (i,j) in gbv.init_inv.keys()])
    X_po_list = {}
    for i in gbv.item_list:
        X_po_list[i] = np.ones(len(gbv.period_list)) * X_list[i]
        for t in range(len(gbv.period_list)):
            X_po_list[i][t:] += sum([gbv.external_purchase[i,j,t] for j in gbv.plant_list if (i,j,t) in gbv.external_purchase.keys()])

    # obtain the upper bound for an item at any specific location without production
    X_j_po_list = {}
    for i in gbv.item_list:
        for j in gbv.plant_list:
            X_j_po_list[i,j] = np.ones(len(gbv.period_list)) * gbv.init_inv.get((i,j),0)
            for t in range(len(gbv.period_list)):
                X_j_po_list[i,j][t:] += gbv.external_purchase.get((i,j,t),0)

    # some issues here
    X_bar_j = {}
    for i in gbv.item_list:
        for j in gbv.plant_list:
            X_bar_j[i,j] = copy.deepcopy(X_j_po_list[i,j])
            for j2 in gbv.plant_list:
                if (j2 != j):
                    for t in range(len(gbv.period_list)):
                        transit_time = len(gbv.period_list)
                        if (i,j2,j) in gbv.transit_time.keys():
                            transit_time = gbv.transit_time[i,j2,j] + 1
                        else:
                            if ((i,j2,gbv.central) in gbv.transit_time.keys()) and ((i,gbv.central,j) in gbv.transit_time.keys()):
                                transit_time = gbv.transit_time[i,j2,gbv.central] + 2 + gbv.transit_time[i,gbv.central,j]
                        if (t >= transit_time):
                            X_bar_j[i,j][t] += X_j_po_list[i,j2][t - transit_time]
    
    gbv.X_outside_ub = X_po_list
    gbv.X_outside_ub_plant = X_bar_j

    return gbv

def analyze_bom(gbv):
    consistuent_list = {}
    composition_list = {}
    for i in gbv.item_list:
        consistuent_list[i] = []
        for j in gbv.plant_list:
            for i1,i2 in gbv.bom_key[j]:
                if (i1 == i)and(not(i2 in consistuent_list[i])):
                    consistuent_list[i].append(i2)

    degree_list = np.zeros(len(gbv.item_list))
    keep_iter = True
    dg_no = 0.0
    while keep_iter:
        now_degree = copy.deepcopy(degree_list)
        for j in gbv.plant_list:
            for i1,i2 in gbv.bom_key[j]:
                if degree_list[gbv.item_list.index(i2)] == dg_no:
                    now_degree[gbv.item_list.index(i1)] = max(now_degree[gbv.item_list.index(i1)], degree_list[gbv.item_list.index(i2)] + 1)
                    composition_list[i1,i2] = gbv.bom_dict[j][i1,i2]
                    current_dict_keys = list(composition_list.keys())
                    for item in current_dict_keys:
                        if item[0] == i2:
                            composition_list[i1,item[1]] = gbv.bom_dict[j][i1,i2] * composition_list[item]

        if np.array_equiv(now_degree, degree_list):
            keep_iter = False
        else:
            degree_list = now_degree
            dg_no += 1

    subsidiary_list = {}
    precursor_list = {}
    for i in gbv.item_list:
        subsidiary_list[i] = [item[1] for item in composition_list.keys() if item[0] == i]
        precursor_list[i] = [item[0] for item in composition_list.keys() if item[1] == i]
    
    gbv.degree_list = degree_list
    gbv.consistuent_list = consistuent_list
    gbv.composition_list = composition_list
    gbv.subsidiary_list = subsidiary_list
    gbv.precursor_list = precursor_list

    return gbv