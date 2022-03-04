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