# bound tightening for test cases

from pass_arg_rebuild_model import *
import copy

current_folder = ("../data/small_test_fine_tune")
gbv = create_gbv(current_folder)

hparams = ADMMparams("params.json")

# obtain the constituent dictionary and identify the degree of each item
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

X_list = {}
for i in gbv.item_list:
    X_list[i] = sum([gbv.init_inv[i,j] for j in gbv.plant_list if (i,j) in gbv.init_inv.keys()])

X_po_list = {}
for i in gbv.item_list:
    X_po_list[i] = np.ones(len(gbv.period_list)) * X_list[i]
    for t in range(len(gbv.period_list)):
        X_po_list[i][t:] += sum([gbv.external_purchase[i,j,t] for j in gbv.plant_list if (i,j,t) in gbv.external_purchase.keys()])

X_j_po_list = {}
for i in gbv.item_list:
    for j in gbv.plant_list:
        X_j_po_list[i,j] = np.ones(len(gbv.period_list)) * gbv.init_inv.get((i,j),0)
        for t in range(len(gbv.period_list)):
            X_j_po_list[i,j][t:] += gbv.external_purchase.get((i,j,t),0)

X_bar_j = {}
for i,j in gbv.prod_key:
    X_bar_j[i,j] = X_j_po_list[i,j]
    for t in range(len(gbv.period_list)):
        for j2 in gbv.plant_list:
            if (j2 != j) and (t > gbv.transit_time.get((i,j2,j),len(gbv.period_list))):
                X_bar_j[i,j][t] += X_j_po_list[i,j2][t - gbv.transit_time[i,j2,j]]

# obtain the crude upper bound
UB_list = {}
for i in gbv.item_list:
    if not(consistuent_list[i] == []):
        compareList = [X_po_list[i2]/composition_list[i1,i2] for i1,i2 in composition_list.keys() if i1 == i]
        UB_list[i] = np.minimum.reduce(compareList)