import pandas as pd
import numpy as np
import os
import scipy.sparse as sp

# We need data structure for items, plants, calendar

def readin_csv(file_add):
    return pd.read_csv(file_add)

def readin_array(path_file):
    # read in the item data
    # gbv.item_list, gbv.holding_cost, gbv.penalty_cost = input_item(os.path.join(path_file, "dm_df_item.csv"))
    item_path = os.path.join(path_file, "dm_df_item.csv")
    item_data = readin_csv(item_path)
    item_data.drop_duplicates(inplace=True, subset=['item_code'])
    item_list = list(item_data.item_code)
    item_index = list(item_data.index)
    I_norm = len(item_list)
    item_df = pd.DataFrame(data = {'index': list(item_data.item_code),
                                   'I': list(range(len(item_data.item_code)))})
    item_df.set_index('index', inplace=True)

    # read in the plant data
    # gbv.plant_list = input_plant(os.path.join(path_file, "dm_df_plant.csv"))
    # gbv.M = len(gbv.plant_list)
    plant_path = os.path.join(path_file, "dm_df_plant.csv")
    plant_data = readin_csv(plant_path)
    plant_data.drop_duplicates(inplace=True, subset=['plant'])
    # read in the item plant data
    # gbv.item_plant = input_item_plant(os.path.join(path_file, "dm_df_item_plant.csv"))
    item_plant_path = os.path.join(path_file, "dm_df_item_plant.csv")
    item_plant_data = readin_csv(item_plant_path)
    item_plant_data.drop_duplicates(inplace=True)
    item_plant_list = item_plant_data.plant.drop_duplicates()

    # read in the period data
    # gbv.period_list = input_periods(os.path.join(path_file, "dm_df_periods.csv"))
    # gbv.T = len(gbv.period_list)
    period_path = os.path.join(path_file, "dm_df_periods.csv")
    period_data = readin_csv(period_path)
    period_data.drop_duplicates(inplace=True)
    period_list_data = period_data[period_data['length']==1]
    period_list = list(period_list_data.period)
    T_norm = len(period_list)
    period_df = pd.DataFrame(data = {'index': period_list,
                                   'I': list(range(T_norm))})
    period_df.set_index('index', inplace=True)

    # read in the initial inventory data
    # gbv.init_inv = input_init_inv(os.path.join(path_file, "dm_df_inv.csv"))
    init_inv_path = os.path.join(path_file, "dm_df_inv.csv")
    init_inv_data = readin_csv(init_inv_path)
    init_inv_data.drop_duplicates(inplace=True)
    init_inv_data = init_inv_data[init_inv_data.item_code.isin(item_list)]
    init_inv_plant_list = init_inv_data.plant.drop_duplicates()

    # read in the external purchase data
    # gbv.external_purchase = input_po(os.path.join(path_file, "dm_df_po.csv"))
    po_path = os.path.join(path_file, "dm_df_po.csv")
    po_data = readin_csv(po_path)
    po_data.drop_duplicates(inplace=True)
    po_data = po_data[po_data.period.isin(period_list) & po_data.item_code.isin(item_list)]
    po_plant_list = po_data.plant.drop_duplicates()

    # read in the demand data
    # gbv.real_demand, gbv.forecast_demand = input_demand(os.path.join(path_file, "dm_df_demand.csv"))
    demand_path = os.path.join(path_file, "dm_df_demand.csv")
    demand_data = readin_csv(demand_path)
    demand_data.drop_duplicates(inplace=True)
    demand_data = demand_data[(demand_data.period.isin(period_list)) & (demand_data.item_code.isin(item_list))]

    # read in the alternate data
    # gbv.alt_list, gbv.alt_dict, gbv.alt_cost = input_item_alternate(os.path.join(path_file, "dm_df_alternate_item.csv"))
    alternate_path = os.path.join(path_file, "dm_df_alternate_item.csv")
    alternate_data = readin_csv(alternate_path)
    alternate_data.drop_duplicates(inplace=True)
    alternate_data = alternate_data[alternate_data.item_code1 != alternate_data.item_code2]
    alternate_data = alternate_data[alternate_data.period.isin(period_list)]
    alternate_data = alternate_data[(alternate_data.item_code1.isin(item_list)) & (alternate_data.item_code2.isin(item_list))]
    alternate_plant_list = alternate_data.plant.drop_duplicates()
    alternate_sub = pd.concat([alternate_data[alternate_data.priority_col == 2].item_code1, alternate_data[alternate_data.priority_col == 1].item_code2])
    alternate_subbed = pd.concat([alternate_data[alternate_data.priority_col == 1].item_code1, alternate_data[alternate_data.priority_col == 2].item_code2])

    # read in the unit capacity data
    # gbv.unit_cap, gbv.unit_cap_type = input_unit_capacity(os.path.join(path_file, "dm_df_unit_capacity.csv"))
    unit_cap_path = os.path.join(path_file, "dm_df_unit_capacity.csv")
    unit_cap_data = readin_csv(unit_cap_path)
    unit_cap_data.drop_duplicates(inplace=True)
    unit_cap_data = unit_cap_data[unit_cap_data.item_code.isin(item_list)]
#    unit_cap_data = unit_cap_data[unit_cap_data.unit_capacity > 0]
    unit_cap_dict = {}
    for i in unit_cap_data.index:
        unit_cap_dict[unit_cap_data.item_code[i], unit_cap_data.capacity_type[i]] = unit_cap_data.unit_capacity[i]
    # read in the item set
    set_path = os.path.join(path_file, "dm_df_set.csv")
    set_data = readin_csv(set_path)
    set_data.drop_duplicates(inplace=True)
    set_data = set_data[set_data.item_code.isin(item_list)]
    set_list = set_data.set.drop_duplicates()
    Ct_norm = len(set_list)
    set_df = pd.DataFrame(data = {'index': set_data.set[set_list.index],
                                   'I': list(range(Ct_norm))})
    set_df.set_index('index', inplace=True)
    
    # read in the item set data
    # gbv.item_set, gbv.set_list = input_item_set(os.path.join(path_file, "dm_df_item_set.csv"))
    item_set_path = os.path.join(path_file, "dm_df_item_set.csv")
    item_set_data = readin_csv(item_set_path)
    item_set_data.drop_duplicates(inplace=True)
    item_set_data = item_set_data[item_set_data.item_code.isin(item_list)]

    # read in the production data
    # gbv.prod_key, gbv.lot_size, gbv.lead_time, gbv.component_holding_cost, \
    #     gbv.prod_cost, gbv.min_prod, gbv.max_prod = input_production(os.path.join(path_file, "dm_df_production.csv"))
    production_path = os.path.join(path_file, "dm_df_production.csv")
    production_data = readin_csv(production_path)
    production_data.drop_duplicates(inplace=True)
    production_data = production_data[production_data.item_code.isin(item_list)]
    prod_plant_list = production_data.plant.drop_duplicates()

    # read in the bom data
    # gbv.bom_key, gbv.bom_dict = input_bom(os.path.join(path_file, "dm_df_bom.csv"))
    bom_path = os.path.join(path_file, "dm_df_bom.csv")
    bom_data = readin_csv(bom_path)
    bom_data.drop_duplicates(inplace=True)
    bom_data = bom_data[(bom_data.assembly.isin(item_list))&((bom_data.component.isin(item_list)))]

    # collect all used plants
    used_plants = np.unique(list(prod_plant_list) + list(alternate_plant_list) + list(po_plant_list) + list(init_inv_plant_list))
    plant_data_used = plant_data[plant_data.plant.isin(used_plants)]
    plant_data_used = plant_data_used.reset_index()
    J_norm = len(plant_data_used)
    plant_df = pd.DataFrame(data = {'index': plant_data_used.plant,
                                   'I': list(range(J_norm))})
    plant_df.set_index('index', inplace=True)

    cap_index = [i for i in set_data.index if (set_data.plant[i] in plant_df.index)\
                and (set_data.item_code[i] in item_df.index)\
                and (set_data.set[i] in set_df.index)]

    # read in the transit data (this must be after obtaining the plant list!!!)
    # gbv.transit_list, gbv.transit_time, gbv.transit_cost = input_transit(os.path.join(path_file, "dm_df_transit.csv"))
    # gbv.L = len(gbv.transit_list)
    transit_path = os.path.join(path_file, "dm_df_transit.csv")
    transit_data = readin_csv(transit_path)
    transit_data = transit_data[transit_data.item_code.isin(item_list)]
    transit_data.drop_duplicates(subset=['item_code','src_plant','dest_plant'], inplace=True)
    transit_data = transit_data[transit_data.src_plant.isin(plant_data_used.plant)]
    transit_data = transit_data[transit_data.dest_plant.isin(plant_data_used.plant)]
    transit_data = transit_data.reset_index()
    transit_data['transit_cost'] = transit_data['transit_cost'].fillna(0)
    transit_data['holding_cost'] = transit_data['holding_cost'].fillna(0)
    Tr_norm = len(transit_data)
    Tr_list = transit_data.index

    # read in the plant capacity data
    # gbv.cap_period_list, gbv.cap_key, gbv.max_cap = input_capacity(os.path.join(path_file, "dm_df_max_capacity.csv"))
    plant_cap_path = os.path.join(path_file, "dm_df_max_capacity.csv")
    plant_cap_data = readin_csv(plant_cap_path)
    plant_cap_data.drop_duplicates(inplace=True)
    plant_cap_data = plant_cap_data[(plant_cap_data.period.isin(period_list)) & \
                                    (plant_cap_data.plant.isin(plant_data_used.plant)) & \
                                    (plant_cap_data.set.isin(set_list))]

    #---------------------------------------------------------------------
    # construct all the data frames for output
    df_cost = pd.DataFrame(data={'I': list(item_df.I[list(item_data.item_code)]),
                            'HC': item_data.holding_cost,
                            'PC': item_data.holding_cost*1000})

    df_alt = pd.DataFrame(data={'I': list(item_df.I[list(alternate_sub[list(alternate_data.index)])]),
                                'II': list(item_df.I[list(alternate_subbed[list(alternate_data.index)])]),
                                'J': list(plant_df.I[list(alternate_data.plant[alternate_data.index])]),
                                'Ti': list(period_df.I[list(alternate_data.period[alternate_data.index])]),
                                'V': alternate_data.multiplier})
    df_alt.drop_duplicates(inplace=True)
    
    df_iniInv = pd.DataFrame(data={'I': list(item_df.I[list(init_inv_data.item_code[init_inv_data.index])]),
                                   'J': list(plant_df.I[list(init_inv_data.plant[init_inv_data.index])]),
                                   'V': init_inv_data.qty})
    
    df_external = pd.DataFrame(data={'I': list(item_df.I[list(po_data.item_code[po_data.index])]),
                                     'J': list(plant_df.I[list(po_data.plant[po_data.index])]),
                                     'Ti': list(period_df.I[list(po_data.period[list(po_data.index)])]),
                                     'V': po_data.qty})

    df_unitCap = pd.DataFrame(data={'I': list(item_df.I[list(set_data.item_code[cap_index])]),
                                'J': list(plant_df.I[list(set_data.plant[cap_index])]),
                                'Ct': list(set_df.I[list(set_data.set[cap_index])]),
                                'V': [unit_cap_dict[set_data.item_code[i], set_data.capacity_type[i]] for i in cap_index]})

    df_prod = pd.DataFrame(data={'I': list(item_df.I[list(production_data.item_code[production_data.index])]),
                                'J': list(plant_df.I[list(production_data.plant[production_data.index])]),
                                'LT': production_data.lead_time,
                                'pKey': [1 for i in range(len(production_data))],
                                'MaxProd': production_data.max_production,
                                'LS': production_data.lot_size})

    df_transit = pd.DataFrame(data={'I': list(item_df.I[list(transit_data.item_code[transit_data.index])]),
                                    'J': list(plant_df.I[list(transit_data.src_plant[transit_data.index])]),
                                    'JJ': list(plant_df.I[list(transit_data.dest_plant[transit_data.index])]),
                                    'Tr': transit_data.index,
                                    'V': [1 for i in transit_data.index]})
    df_transitTC = pd.DataFrame(data={'Tr': transit_data.index,
                                      'V': transit_data.lead_time,
                                      'TC': transit_data.transit_cost})

    df_bom = pd.DataFrame(data={'II': list(item_df.I[list(bom_data.component[bom_data.index])]),
                                'I': list(item_df.I[list(bom_data.assembly[bom_data.index])]),
                                'J': list(plant_df.I[list(bom_data.plant[bom_data.index])]),
                                'V': bom_data.qty})
    
    df_maxCap = pd.DataFrame(data={'J': list(plant_df.I[list(plant_cap_data.plant[plant_cap_data.index])]),
                                'Ti': list(period_df.I[list(plant_cap_data.period[plant_cap_data.index])]),
                                'Ct': list(set_df.I[list(plant_cap_data.set[plant_cap_data.index])]),
                                'V': plant_cap_data.max_capacity})

    df_demand = pd.DataFrame(data={'I': list(item_df.I[list(demand_data.item_code[demand_data.index])]),
                                   'Ti': list(period_df.I[list(demand_data.period[demand_data.index])]),
                                   'V': demand_data.order_demand})

    return df_cost, df_alt, df_iniInv, df_external, df_unitCap, df_prod, df_transit, df_transitTC, \
           df_bom, df_maxCap, df_demand, item_df, plant_df, period_df, set_df, I_norm, J_norm, T_norm, Ct_norm, Tr_norm, item_data