# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# We need data structure for items, plants, calendar


def readin_csv(file_add):
    return pd.read_csv(file_add)

# %%
# item related data


def input_item(item_file):
    # input dm_df_item: item basic information
    # output the item list and holding costs
    item_data = readin_csv(item_file)
    item_list = list(item_data.item_code)       # list of items
    holding_cost = {item_list[i]: item_data.holding_cost[i] for i in range(len(item_list))}
    return item_list, holding_cost
    

def input_item_set(item_set_file):
    # input dm_df_item_set: item set information
    # output the item production set 
    item_set_data = readin_csv(item_set_file)
    item_list = list(item_set_data.item_code)       # list of items
    item_set = {item_list[i]: item_set_data.set[i] for i in range(len(item_list))}
    return item_set
    

def input_item_alternate(alternate_file):
    # input dm_df_alternate_item: item replacement information
    # output the alternate item production multipliers
    alter_data = readin_csv(alternate_file)
    subbed_list = list(alter_data.item_code1)       # list of items being subbed
    sub_list = list(alter_data.item_code2)          # list of items to sub
    plant_list = list(alter_data.plant)             # list of plants
    period_list = list(alter_data.period)           # list of periods
    alt_multi = {(subbed_list[i], sub_list[i], plant_list[i], period_list[i]): alter_data.multiplier[i] for i in
                 range(len(subbed_list))}
    return alt_multi
    

def input_unit_capacity(unitc_file):
    # input dm_df_unit_capacity: item unit capacity consumption information
    # output unit capacity, unit capacity type
    unit_cap_data = readin_csv(unitc_file)
    item_list = list(unit_cap_data.item_code)       # list of items
    plant_list = list(unit_cap_data.plant)             # list of plants
    unit_cap = {(item_list[i], plant_list[i]): unit_cap_data.unit_capacity[i] for i in range(len(item_list))}
    unit_cap_type = {(item_list[i], plant_list[i]): unit_cap_data.capacity_type[i] for i in range(len(item_list))}
    return unit_cap, unit_cap_type
    

def input_item_plant(item_plant_file):
    # input dm_df_item_plant: item plant information
    # output item plant mapping
    item_plant_data = readin_csv(item_plant_file)
    item_list = list(item_plant_data.item_code)
    item_set = set(item_plant_data.item_code)
    item_plant = {item: [] for item in item_set}
    for i in range(len(item_list)):
        if not(item_plant_data.plant[i] in item_plant[item_plant_data.item_code[i]]):
            item_plant[item_plant_data.item_code[i]].append(item_plant_data.plant[i])
    return item_plant


def input_bom(bom_file):
    # input dm_df_bom: item bom relationship information
    # output item bom relationship
    bom_data = readin_csv(bom_file)
    bom_key = []
    bom_dict = {}
    for i in range(len(bom_data.assembly)):
        if not((bom_data.assembly[i], bom_data.component[i]) in bom_key):
            bom_key.append((bom_data.assembly[i], bom_data.component[i]))
            bom_dict[(bom_data.assembly[i], bom_data.component[i])] = {bom_data.plant[i]: bom_data.qty[i]}
        else:
            bom_dict[(bom_data.assembly[i], bom_data.component[i])][bom_data.plant[i]] = bom_data.qty[i]
    return bom_key, bom_dict
    

def input_production(production_file):
    # input dm_df_production: item production relationship information
    production_data = readin_csv(production_file)
    prod_key = []
    lot_size = {}
    lead_time = {}
    holding_cost = {}
    prod_cost = {}
    min_prod = {}
    max_prod = {}
    for i in range(len(production_data.item_code)):
        line_key = (production_data.item_code[i], production_data.plant[i])
        prod_key.append(line_key)
        lot_size[line_key] = production_data.lot_size[i]
        lead_time[line_key] = production_data.lead_time[i]
        holding_cost[line_key] = production_data.component_holding_cost[i]
        prod_cost[line_key] = production_data.production_cost[i]
        min_prod[line_key] = production_data.min_production[i]
        max_prod[line_key] = production_data.max_production[i]
    return prod_key, lot_size, lead_time, holding_cost, prod_cost, min_prod, max_prod
    
# %%
# plant related data


def input_plant(plant_file):
    # input dm_df_plant: plant basic information
    plant_data = readin_csv(plant_file)
    plant_list = list(plant_data.plant)
    return plant_list
    

def input_capacity(cap_file):
    # input dm_df_max_capacity: plant capacity information
    cap_data = readin_csv(cap_file)
    period_list = list(cap_data.period.unique())
    cap_key = []
    max_cap = {}
    for i in range(len(cap_data.set)):
        line_key = (cap_data.plant[i], cap_data.period[i])
        cap_key.append(line_key)
        if not(line_key in max_cap.keys()):
            max_cap[line_key] = cap_data.max_capacity[i]
        else:
            max_cap[line_key][cap_data.period[i]] = cap_data.max_capacity[i]
    return period_list, cap_key, max_cap
    

def input_transit(transit_file):
    # input dm_df_transit: plant transit information
    transit_data = readin_csv(transit_file)
    transit_time = {
        (transit_data.item_code[i], transit_data.src_plant[i], transit_data.dest_plant[i]): transit_data.lead_time[i]
        for i in range(len(transit_data.item_code))}
    transit_cost = {
        (transit_data.item_code[i], transit_data.src_plant[i], transit_data.dest_plant[i]): transit_data.transit_cost[i]
        for i in range(len(transit_data.item_code))}
    return transit_time, transit_cost
    

def input_init_inv(init_inv_file):
    # input dm_df_inv: plant initial inventory information
    init_inv_data = readin_csv(init_inv_file)
    init_inv = {(init_inv_data.item_code[i], init_inv_data.plant[i]): init_inv_data.qty[i] for i in
                range(len(init_inv_data.item_code))}
    return init_inv
    
# %%
# time related data


def input_periods(periods_file):
    # input dm_df_periods: periods information
    periods_data = readin_csv(periods_file)
    period_delay = {periods_data.period[i]: periods_data.delay[i] for i in range(len(periods_data.period))}
    return period_delay
    

def input_po(po_file):
    # input dm_df_po: external purchase information
    po_data = readin_csv(po_file)
    external_purchase = {(po_data.item_code[i], po_data.plant[i], po_data.period[i]): po_data.qty[i] for i in
                         range(len(po_data.period))}
    return external_purchase
    

def input_demand(demand_file):
    # input dm_df_demand: external demand information
    demand_data = readin_csv(demand_file)
    real_demand = {(demand_data.item_code[i], demand_data.period[i]): demand_data.order_demand[i] for i in
                   range(len(demand_data.item_code))}
    forecast_demand = {(demand_data.item_code[i], demand_data.period[i]): demand_data.forecast_demand_with_fr[i] for i
                       in range(len(demand_data.item_code))}
    return real_demand, forecast_demand
