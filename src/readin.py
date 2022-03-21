# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:58:01 2022

@author: haoxiang
"""

'''
Read in the data of production planning problem 
'''

import pandas as pd

# We need data structure for items, plants, inventory, transit, capacity

def readin_csv(file_add):
    return pd.read_csv(file_add)

class prod_item():
    '''
        Data structure for items.

    '''
    def __init__(self):
        pass
    
    def input_item(file_add):
        # read in the raw data related to items
        raw_data = readin_csv(file_add)