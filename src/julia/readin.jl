# script to read in the data from the csv files

using CSV
using DataFrames

readin_csv(file_add::AbstractString) = CSV.read(file_add, DataFrame)

function _id_map(values)
    return Dict(v => i - 1 for (i, v) in enumerate(values))
end

function readin_array(path_file::AbstractString)
    item_data = readin_csv(joinpath(path_file, "dm_df_item.csv"))
    unique!(item_data, :item_code)
    item_list = collect(item_data.item_code)
    I_norm = length(item_list)
    item_to_idx = _id_map(item_list)
    item_df = DataFrame(index = item_list, I = collect(0:I_norm - 1))

    plant_data = readin_csv(joinpath(path_file, "dm_df_plant.csv"))
    unique!(plant_data, :plant)

    item_plant_data = readin_csv(joinpath(path_file, "dm_df_item_plant.csv"))
    unique!(item_plant_data)

    period_data = readin_csv(joinpath(path_file, "dm_df_periods.csv"))
    unique!(period_data)
    period_list_data = period_data[period_data.length .== 1, :]
    period_list = collect(period_list_data.period)
    T_norm = length(period_list)
    period_to_idx = _id_map(period_list)
    period_df = DataFrame(index = period_list, I = collect(0:T_norm - 1))

    init_inv_data = readin_csv(joinpath(path_file, "dm_df_inv.csv"))
    unique!(init_inv_data)
    init_inv_data = init_inv_data[in.(init_inv_data.item_code, Ref(Set(item_list))), :]

    po_data = readin_csv(joinpath(path_file, "dm_df_po.csv"))
    unique!(po_data)
    po_data = po_data[in.(po_data.period, Ref(Set(period_list))) .& in.(po_data.item_code, Ref(Set(item_list))), :]

    demand_data = readin_csv(joinpath(path_file, "dm_df_demand.csv"))
    unique!(demand_data)
    demand_data = demand_data[in.(demand_data.period, Ref(Set(period_list))) .& in.(demand_data.item_code, Ref(Set(item_list))), :]

    alternate_data = readin_csv(joinpath(path_file, "dm_df_alternate_item.csv"))
    unique!(alternate_data)
    alternate_data = alternate_data[alternate_data.item_code1 .!= alternate_data.item_code2, :]
    alternate_data = alternate_data[in.(alternate_data.period, Ref(Set(period_list))), :]
    alternate_data = alternate_data[in.(alternate_data.item_code1, Ref(Set(item_list))) .& in.(alternate_data.item_code2, Ref(Set(item_list))), :]

    unit_cap_data = readin_csv(joinpath(path_file, "dm_df_unit_capacity.csv"))
    unique!(unit_cap_data)
    unit_cap_data = unit_cap_data[in.(unit_cap_data.item_code, Ref(Set(item_list))), :]
    unit_cap_dict = Dict((r.item_code, r.capacity_type) => r.unit_capacity for r in eachrow(unit_cap_data))

    set_data = readin_csv(joinpath(path_file, "dm_df_set.csv"))
    unique!(set_data)
    set_data = set_data[in.(set_data.item_code, Ref(Set(item_list))), :]
    set_list = unique(set_data.set)
    Ct_norm = length(set_list)
    set_to_idx = _id_map(set_list)
    set_df = DataFrame(index = set_list, I = collect(0:Ct_norm - 1))

    item_set_data = readin_csv(joinpath(path_file, "dm_df_item_set.csv"))
    unique!(item_set_data)
    item_set_data = item_set_data[in.(item_set_data.item_code, Ref(Set(item_list))), :]

    production_data = readin_csv(joinpath(path_file, "dm_df_production.csv"))
    unique!(production_data)
    production_data = production_data[in.(production_data.item_code, Ref(Set(item_list))), :]

    bom_data = readin_csv(joinpath(path_file, "dm_df_bom.csv"))
    unique!(bom_data)
    bom_data = bom_data[in.(bom_data.assembly, Ref(Set(item_list))) .& in.(bom_data.component, Ref(Set(item_list))), :]

    prod_plant_set = Set(production_data.plant)
    alt_plant_set = Set(alternate_data.plant)
    po_plant_set = Set(po_data.plant)
    init_inv_plant_set = Set(init_inv_data.plant)
    used_plants = unique(vcat(collect(prod_plant_set), collect(alt_plant_set), collect(po_plant_set), collect(init_inv_plant_set)))
    plant_data_used = plant_data[in.(plant_data.plant, Ref(Set(used_plants))), :]
    J_norm = nrow(plant_data_used)
    plant_to_idx = _id_map(collect(plant_data_used.plant))
    plant_df = DataFrame(index = collect(plant_data_used.plant), I = collect(0:J_norm - 1))

    cap_mask = in.(set_data.plant, Ref(Set(plant_df.index))) .& in.(set_data.item_code, Ref(Set(item_df.index))) .& in.(set_data.set, Ref(Set(set_df.index)))
    cap_rows = set_data[cap_mask, :]

    transit_data = readin_csv(joinpath(path_file, "dm_df_transit.csv"))
    transit_data = transit_data[in.(transit_data.item_code, Ref(Set(item_list))), :]
    unique!(transit_data, [:item_code, :src_plant, :dest_plant])
    transit_data = transit_data[in.(transit_data.src_plant, Ref(Set(plant_data_used.plant))) .& in.(transit_data.dest_plant, Ref(Set(plant_data_used.plant))), :]
    transit_data.transit_cost = coalesce.(transit_data.transit_cost, 0.0)
    transit_data.holding_cost = coalesce.(transit_data.holding_cost, 0.0)
    Tr_norm = nrow(transit_data)

    plant_cap_data = readin_csv(joinpath(path_file, "dm_df_max_capacity.csv"))
    unique!(plant_cap_data)
    plant_cap_data = plant_cap_data[in.(plant_cap_data.period, Ref(Set(period_list))) .&
                                    in.(plant_cap_data.plant, Ref(Set(plant_data_used.plant))) .&
                                    in.(plant_cap_data.set, Ref(Set(set_list))), :]

    df_cost = DataFrame(
        I = [item_to_idx[c] for c in item_data.item_code],
        HC = item_data.holding_cost,
        PC = item_data.holding_cost .* 1000,
    )

    alt_I_codes = Vector{eltype(alternate_data.item_code1)}(undef, nrow(alternate_data))
    alt_II_codes = Vector{eltype(alternate_data.item_code2)}(undef, nrow(alternate_data))
    for idx in 1:nrow(alternate_data)
        if alternate_data.priority_col[idx] == 2
            alt_I_codes[idx] = alternate_data.item_code1[idx]
            alt_II_codes[idx] = alternate_data.item_code2[idx]
        else
            # Match Python construction:
            # priority 1 rows are reversed (I <- item_code2, II <- item_code1).
            alt_I_codes[idx] = alternate_data.item_code2[idx]
            alt_II_codes[idx] = alternate_data.item_code1[idx]
        end
    end

    df_alt = DataFrame(
        I = [item_to_idx[c] for c in alt_I_codes],
        II = [item_to_idx[c] for c in alt_II_codes],
        J = [plant_to_idx[p] for p in alternate_data.plant],
        Ti = [period_to_idx[t] for t in alternate_data.period],
        V = alternate_data.multiplier,
    )
    unique!(df_alt)

    df_iniInv = DataFrame(
        I = [item_to_idx[c] for c in init_inv_data.item_code],
        J = [plant_to_idx[p] for p in init_inv_data.plant],
        V = init_inv_data.qty,
    )

    df_external = DataFrame(
        I = [item_to_idx[c] for c in po_data.item_code],
        J = [plant_to_idx[p] for p in po_data.plant],
        Ti = [period_to_idx[t] for t in po_data.period],
        V = po_data.qty,
    )

    df_unitCap = DataFrame(
        I = [item_to_idx[c] for c in cap_rows.item_code],
        J = [plant_to_idx[p] for p in cap_rows.plant],
        Ct = [set_to_idx[s] for s in cap_rows.set],
        V = [unit_cap_dict[(r.item_code, r.capacity_type)] for r in eachrow(cap_rows)],
    )

    df_prod = DataFrame(
        I = [item_to_idx[c] for c in production_data.item_code],
        J = [plant_to_idx[p] for p in production_data.plant],
        LT = production_data.lead_time,
        pKey = ones(Int, nrow(production_data)),
        MaxProd = production_data.max_production,
        LS = production_data.lot_size,
    )

    df_transit = DataFrame(
        I = [item_to_idx[c] for c in transit_data.item_code],
        J = [plant_to_idx[p] for p in transit_data.src_plant],
        JJ = [plant_to_idx[p] for p in transit_data.dest_plant],
        Tr = collect(0:Tr_norm - 1),
        V = ones(Int, Tr_norm),
    )

    df_transitTC = DataFrame(
        Tr = collect(0:Tr_norm - 1),
        V = transit_data.lead_time,
        TC = transit_data.transit_cost,
    )

    df_bom = DataFrame(
        II = [item_to_idx[c] for c in bom_data.component],
        I = [item_to_idx[c] for c in bom_data.assembly],
        J = [plant_to_idx[p] for p in bom_data.plant],
        V = bom_data.qty,
    )

    df_maxCap = DataFrame(
        J = [plant_to_idx[p] for p in plant_cap_data.plant],
        Ti = [period_to_idx[t] for t in plant_cap_data.period],
        Ct = [set_to_idx[s] for s in plant_cap_data.set],
        V = plant_cap_data.max_capacity,
    )

    df_demand = DataFrame(
        I = [item_to_idx[c] for c in demand_data.item_code],
        Ti = [period_to_idx[t] for t in demand_data.period],
        V = demand_data.order_demand,
    )

    return (
        df_cost = df_cost,
        df_alt = df_alt,
        df_iniInv = df_iniInv,
        df_external = df_external,
        df_unitCap = df_unitCap,
        df_prod = df_prod,
        df_transit = df_transit,
        df_transitTC = df_transitTC,
        df_bom = df_bom,
        df_maxCap = df_maxCap,
        df_demand = df_demand,
        item_df = item_df,
        plant_df = plant_df,
        period_df = period_df,
        set_df = set_df,
        I_norm = I_norm,
        J_norm = J_norm,
        T_norm = T_norm,
        Ct_norm = Ct_norm,
        Tr_norm = Tr_norm,
        item_data = item_data,
    )
end