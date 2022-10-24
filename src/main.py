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

gbv = GlobalVariable()

# load the data and form a data structure and create the global instance data frame
gbv.item_list, gbv.holding_cost = input_item("./data/pilot_test/dm_df_item.csv")
gbv.K = len(gbv.item_list)
gbv.item_set, gbv.set_list = input_item_set("./data/pilot_test/dm_df_item_set.csv")
gbv.alt_list, gbv.alt_dict, gbv.alt_cost = input_item_alternate("./data/pilot_test/dm_df_alternate_item.csv")
gbv.unit_cap, gbv.unit_cap_type = input_unit_capacity("./data/pilot_test/dm_df_unit_capacity.csv")
gbv.item_plant = input_item_plant("./data/pilot_test/dm_df_item_plant.csv")
gbv.bom_key, gbv.bom_dict = input_bom("./data/pilot_test/dm_df_bom.csv")
gbv.prod_key, gbv.lot_size, gbv.lead_time, gbv.holding_cost, \
    gbv.prod_cost, gbv.min_prod, gbv.max_prod = input_production("./data/pilot_test/dm_df_production.csv")
# need a penalty cost for demand mismatch

gbv.plant_list = input_plant("./data/pilot_test/dm_df_plant.csv")
gbv.M = len(gbv.plant_list)
gbv.cap_period_list, gbv.cap_key, gbv.max_cap = input_capacity("./data/pilot_test/dm_df_max_capacity.csv")
gbv.transit_list, gbv.transit_time, gbv.transit_cost = input_transit("./data/pilot_test/dm_df_transit.csv")
gbv.L = len(gbv.transit_list)
gbv.init_inv = input_init_inv("./data/pilot_test/dm_df_inv.csv")

gbv.period_list = input_periods("./data/pilot_test/dm_df_periods.csv")
gbv.T = len(gbv.period_list)
gbv.external_purchase = input_po("./data/pilot_test/dm_df_po.csv")
gbv.real_demand, gbv.forecast_demand = input_demand("./data/pilot_test/dm_df_demand.csv")

# run the ADMM process