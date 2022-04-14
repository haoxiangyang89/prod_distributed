# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:58:01 2022

@author: haoxiang
"""

'''
input the data of production planning problem
'''

import pandas as pd

# We need data structure for items, plants, calendar

def readin_csv(file_add):
    return pd.read_csv(file_add)

#%%
# item related data
def input_item(item_file):
    # input dm_df_item: item basic information
    # output the item list and holding costs
    item_data = readin_csv(item_file)
    item_list = list(item_data.item_code)       # list of items
    holding_cost = {item_list[i]:item_data.holding_cost[i] for i in range(len(item_list))}
    return item_list, holding_cost
    
def input_item_set(item_set_file):
    # input dm_df_item_set: item set information
    # output the item production set 
    item_set_data = readin_csv(item_set_file)
    item_list = list(item_set_data.item_code)       # list of items
    item_set = {item_list[i]:item_set_data.set[i] for i in range(len(item_list))}
    return item_set
    
def input_item_alternate(alternate_file):
    # input dm_df_alternate_item: item replacement information
    # output the alternate item production multipliers
    alter_data = readin_csv(alternate_file)
    subbed_list = list(alter_data.item_code1)       # list of items being subbed
    sub_list = list(alter_data.item_code2)          # list of items to sub
    plant_list = list(alter_data.plant)             # list of plants
    period_list = list(alter_data.period)           # list of periods
    alt_multi = {(subbed_list[i], sub_list[i], plant_list[i], period_list[i]): alter_data.multiplier[i] for i in range(len(subbed_list))}
    return alt_multi
    
def input_unit_capacity(unitC_file):
    # input dm_df_unit_capacity: item unit capacity consumption information
    # output unit capacity, unit capacity type
    unit_cap_data = readin_csv(unitC_file)
    item_list = list(unit_cap_data.item_code)       # list of items
    plant_list = list(unit_cap_data.plant)             # list of plants
    unit_cap = {(item_list[i], plant_list[i]):unit_cap_data.unit_capacity[i] for i in range(len(item_list))}
    unit_cap_type = {(item_list[i], plant_list[i]):unit_cap_data.capacity_type[i] for i in range(len(item_list))}
    return unit_cap, unit_cap_type
    
def input_item_plant(item_plant_file):
    # input dm_df_item_plant: item plant information
    # output item plant mapping
    item_plant_data = readin_csv(item_plant_file)
    item_list = list(item_plant_data.item_code)
    item_set = set(item_plant_data.item_code)
    item_plant = {item:[] for item in item_set}
    for i in range(len(item_list)):
        if not(item_plant_data.plant[i] in item_plant[item_plant_data.item_code[i]]):
            item_plant[item_plant_data.item_code[i]].append(item_plant_data.plant[i])
    return item_plant

def input_bom(bom_file):
    # input dm_df_bom: item bom relationship information
    bom_data = readin_csv(bom_file)
    
def input_production(production_file):
    # input dm_df_production: item production relationship information
    production_data = readin_csv(production_file)
    
#%%
# plant related data
def input_capacity(plant_file):
    # input dm_df_plant: plant basic information
    plant_data = readin_csv(plant_file)
    
def input_capacity(cap_file):
    # input dm_df_max_capacity: plant capacity information
    cap_data = readin_csv(cap_file)
    
def input_transit(transit_file):
    # input dm_df_transit: plant transit information
    transit_data = readin_csv(transit_file)
    
def input_init_inv(init_inv_file):
    # input dm_df_inv: plant initial inventory information
    init_inv_data = readin_csv(init_inv_file)
    
    
#%%
# time related data
def input_periods(periods_file):
    # input dm_df_periods: periods information
    periods_data = readin_csv(periods_file)
    
def input_po(po_file):
    # input dm_df_po: external purchase information
    po_data = readin_csv(po_file)
    
def input_demand(demand_file):
    # input dm_df_demand: external demand information
    demand_data = readin_csv(demand_file)
