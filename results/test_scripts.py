gbv = create_gbv("../data/small_test_fine_tune")

[(i,sum(v[i,j,30].X for j in gbv.plant_list)+sum(sum(s[l+(30-t,)].X for t in range(gbv.transit_time[l])) for l in gbv.transit_list if l[0] == i)) for i in gbv.item_list]

for rckeys in rC.keys():
    if rC[rckeys].X > 0.01:
        print(rckeys,rC[rckeys])

[(t,[sum(gbv.unit_cap[i_iter, j_iter, ct_iter] * xC[i_iter, j_iter, t].X for i_iter, j_iter, ct_iter in gbv.unit_cap.keys()
                                 if (j_iter == j) and (0 == ct_iter)) for j in gbv.plant_list]) for t in range(1,31)]
consistuent_list = {}
for i in gbv.item_list:
    consistuent_list[i] = []
    for j in gbv.plant_list:
        for i1,i2 in gbv.bom_key[j]:
            if (i1 == i)and(not(i2 in consistuent_list[i])):
                consistuent_list[i].append(i2)

prob.optimize()

stats_out = {}
for i in gbv.item_list:
    inv_in = sum(gbv.init_inv.get((i,j),0) for j in gbv.plant_list)
    po_in = sum(gbv.external_purchase.get((i,j,t),0) for j in gbv.plant_list for t in gbv.period_list)
    prod_in = sum(xC[i,j,t].X for j in gbv.plant_list for t in gbv.period_list if (i,j) in gbv.prod_key)

    demand_out = sum(gbv.real_demand.get((i,t),0) for t in gbv.period_list)
    prod_out = sum(xC[ii,j,t].X * gbv.bom_dict[j][ii,i]  for ii,j in gbv.prod_key for t in gbv.period_list if (ii, i) in gbv.bom_key[j])
    stats_out[i] = [inv_in, po_in, prod_in, demand_out,prod_out, inv_in+po_in+prod_in-demand_out-prod_out]

for i in gbv.item_list:
    for t in gbv.period_list:
        if u[i,t].X > 0.01:
            print(i,t,u[i,t])

# get the interesting items
bool_prod = np.zeros(len(gbv.item_list)).astype(int)
start_ind = gbv.item_list.index(1709)
bool_prod[start_ind] = 1
keep_iter = True
iter_no = 1
while keep_iter:
    temp_bool = copy.deepcopy(bool_prod)
    for j in gbv.bom_key.keys():
        for item in gbv.bom_key[j]:
            if (bool_prod[gbv.item_list.index(item[0])] == 1)or(bool_prod[gbv.item_list.index(item[1])] == 1):
                temp_bool[gbv.item_list.index(item[0])] = 1
                temp_bool[gbv.item_list.index(item[1])] = 1
    if np.array_equiv(bool_prod,temp_bool):
        keep_iter = False
    bool_prod = copy.deepcopy(temp_bool)
    print(iter_no,keep_iter)
    iter_no += 1
item_interest = [i for i in gbv.item_list if bool_prod[i] == 1]

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

component_number = {}
for item in composition_list.keys():
    if item[0] in component_number.keys():
        component_number[item[0]] += 1
    else:
        component_number[item[0]] = 1

import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
nodeList = []
for i in range(len(gbv.item_list)):
    if degree_list[i] > 0:
        nodeList.append(gbv.item_list[i])
G.add_nodes_from(nodeList)
edgeList = []
for j in gbv.plant_list:
    for item in gbv.bom_key[j]:
        if (not(item in edgeList))and(item[0] in nodeList)and(item[1] in nodeList):
            edgeList.append(item)
G.add_edges_from(edgeList)
options = {
    'node_size': 100,
    'width': 1
}
nx.draw(G, with_labels=True,**options)
