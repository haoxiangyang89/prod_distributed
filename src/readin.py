# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:58:01 2022

@author: haoxiang
"""

'''
Read in the data of production planning problem
'''

import pandas as pd

# We need data structure for items, plants, calendar

def readin_csv(file_add):
    return pd.read_csv(file_add)

def input_bom(file_add, itemList):
    # item bom relationship: read dm_df_bom
    raw_data = readin_csv(file_add)

    #       * dm_df_item_set: item capacity type
    #       * dm_df_unit_capacity: item unit capacity consumption
    #       * dm_df_transit: item transit relationship
    #       * dm_df_inv: item initial inventory

    # initial inventory

    # unit capacity comsumption

class prod_item():
    '''
        Data structure for items.

    '''
    def __init__(self):
        # read in the raw data related to items:
        #       * dm_df_item: item basic information
        #       * other information added through other read-in functions
        pass

class plant():
    '''
        Data structure for plants.

    '''
    def __init__(self):
        # read in the raw data related to items:
        #       * dm_df_item: item basic information
        #       * other information added through other read-in functions
        pass

class calendar():
    '''
        Data structure for external purchase/demand.

    '''
    def __init__(self):
        # read in the raw data related to items:
        #       * dm_df_item: item basic information
        #       * other information added through other read-in functions
        pass
