import numpy as np
from gurobipy import *
import multiprocessing

from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing.managers import BaseManager

def create_model(i):
    mp = Model("item_{}".format(i))
    x1 = mp.addVar(lb=0)
    x2 = mp.addVar(lb=0)
    mp.addConstr(x1 + 2 * x2 <= i, name="constr_1")
    mp.addConstr(2 * x1 + x2 <= i, name="constr_2")
    mp.setObjective(x1 + x2, sense=GRB.MAXIMIZE)
    mp.update()
    return mp

def solve_model(i):
    # mp = Model("item_{}".format(i))
    # x1 = mp.addVar(lb=0)
    # x2 = mp.addVar(lb=0)
    # mp.addConstr(x1 + 2 * x2 <= i, name="constr_1")
    # mp.addConstr(2 * x1 + x2 <= i, name="constr_2")
    # mp.setObjective(x1 + x2, sense=GRB.MAXIMIZE)
    # mp.update()
    # mp.optimize()
    global model_list
    mp = model_list[i]
    mp.optimize()

    return mp.objVal

# test the multiprocessing idea
if __name__ == '__main__':
    global model_list
    set_start_method("fork")
    model_list = []
    for i in range(3):
        mp = create_model(i + 1)
        model_list.append(mp)

    with Pool(3) as p:
        model_results = p.map(solve_model, [0, 1, 2])

    print(model_results)