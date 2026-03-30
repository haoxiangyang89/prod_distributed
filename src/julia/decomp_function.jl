using JuMP
using Gurobi
using Random

function build_decomp_units(data)
    df_alt = data.df_alt
    df_prod = data.df_prod
    df_bom = data.df_bom
    item_ids = collect(data.item_df.I)

    active = Set(vcat(collect(df_alt.I), collect(df_alt.II), collect(df_prod.I), collect(df_bom.I), collect(df_bom.II)))
    decomp_dict = Dict(i => Set([i]) for i in item_ids if i in active)
    for r in eachrow(df_alt)
        a, b = r.I, r.II
        if !haskey(decomp_dict, a) || !haskey(decomp_dict, b)
            continue
        end
        joined = union(decomp_dict[a], decomp_dict[b])
        for item in joined
            decomp_dict[item] = joined
        end
    end

    selected = Dict(k => false for k in keys(decomp_dict))
    decomp_units = Vector{Vector{Int}}()
    for i in keys(decomp_dict)
        if !selected[i]
            u = sort(collect(decomp_dict[i]))
            push!(decomp_units, u)
            for it in u
                selected[it] = true
            end
        end
    end
    return decomp_units
end

function build_benders_context(data)
    prod_key = [(r.I, r.J) for r in eachrow(data.df_prod)]
    lot_size = Dict((r.I, r.J) => r.LS for r in eachrow(data.df_prod))
    max_prod = Dict((r.I, r.J) => r.MaxProd for r in eachrow(data.df_prod))
    unit_cap = Dict((r.I, r.J, r.Ct) => r.V for r in eachrow(data.df_unitCap))
    max_cap = Dict((r.Ct, r.J, r.Ti) => r.V for r in eachrow(data.df_maxCap))
    lead_time = Dict((r.I, r.J) => Int(round(r.LT)) for r in eachrow(data.df_prod))
    real_demand = Dict((r.I, r.Ti) => r.V for r in eachrow(data.df_demand))
    external_purchase = Dict((r.I, r.J, r.Ti) => r.V for r in eachrow(data.df_external))
    init_inv = Dict((r.I, r.J) => r.V for r in eachrow(data.df_iniInv))
    transit_time = Dict(r.Tr => Int(round(r.V)) for r in eachrow(data.df_transitTC))
    holding_cost = Dict(r.I => r.HC for r in eachrow(data.df_cost))
    penalty_cost = Dict(r.I => r.PC for r in eachrow(data.df_cost))
    transit_cost = Dict(r.Tr => r.TC for r in eachrow(data.df_transitTC))
    decomp_units = build_decomp_units(data)
    return (
        data = data,
        prod_key = prod_key,
        lot_size = lot_size,
        max_prod = max_prod,
        unit_cap = unit_cap,
        max_cap = max_cap,
        lead_time = lead_time,
        real_demand = real_demand,
        external_purchase = external_purchase,
        init_inv = init_inv,
        transit_time = transit_time,
        holding_cost = holding_cost,
        penalty_cost = penalty_cost,
        transit_cost = transit_cost,
        decomp_units = decomp_units,
    )
end

function master_prob(ctx; relax_option::Bool = false, level_obj_norm::Float64 = 2.0, logic_constr_flag::Bool = false, composition_list = Dict(), parent_list = Dict())
    data = ctx.data
    prod_key = ctx.prod_key
    lot_size = ctx.lot_size
    max_prod = ctx.max_prod
    unit_cap = ctx.unit_cap
    max_cap = ctx.max_cap
    decomp_units = ctx.decomp_units

    periods = sort(collect(data.period_df.I))
    items = collect(data.item_df.I)
    plants = collect(data.plant_df.I)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", 600.0)

    if relax_option
        @variable(model, x[prod_key, periods] >= 0)
    else
        @variable(model, x[prod_key, periods] >= 0, Int)
    end

    if level_obj_norm == 1.0
        @variable(model, x_dev[prod_key, periods] >= 0)
    elseif level_obj_norm == 1.5
        all_keys = [(i, j, t) for (i, j) in prod_key for t in periods]
        m = min(5, length(all_keys))
        Random.seed!(22)
        pwl_idx = randperm(length(all_keys))[1:m]
        pwl_keys = Set(all_keys[pwl_idx])
        l1_keys = [k for k in all_keys if !(k in pwl_keys)]
        @variable(model, y[all_keys] >= 0)
        @variable(model, z[collect(pwl_keys)] >= 0)
        @constraint(model, zx_eqn[k in collect(pwl_keys)], z[k] == x[(k[1], k[2]), k[3]])
        set_optimizer_attribute(model, "OutputFlag", 0)
        model.ext[:pwl_keys] = collect(pwl_keys)
        model.ext[:l1_keys] = l1_keys
    end

    @variable(model, theta[1:length(decomp_units)] >= 0)

    @constraint(model, max_prod_cons[pk in prod_key, t in periods], x[pk, t] * lot_size[pk] <= max_prod[pk])
    @constraint(model, capacity_cons[(ct, j, t) in keys(max_cap)],
        sum(get(unit_cap, (i2, j2, ct2), 0.0) * x[(i2, j2), t] * lot_size[(i2, j2)]
            for (i2, j2, ct2) in keys(unit_cap)
            if j2 == j && ct2 == ct && haskey(lot_size, (i2, j2)); init=0.0) <= max_cap[(ct, j, t)]
    )

    if logic_constr_flag
        producible = Set(data.df_prod.I)
        for i in items, t in periods
            parents = get(parent_list, i, Int[])
            if !isempty(parents) && !isempty(intersect(producible, Set(parents)))
                @constraint(
                    model,
                    sum(
                        x[(ii, j), t] * get(lot_size, (ii, j), 0.0) * get(composition_list, (ii, i), 0.0)
                        for ii in parents for j in plants if haskey(lot_size, (ii, j))
                    ) <=
                    sum(get(ctx.init_inv, (i, j), 0.0) for j in plants) +
                    sum(get(ctx.external_purchase, (i, j, tt), 0.0) for j in plants for tt in periods if tt <= t) +
                    sum(
                        x[(i, j), tt] * lot_size[(i, j)]
                        for j in plants, tt in periods
                        if haskey(lot_size, (i, j)) && tt <= t - get(ctx.lead_time, (i, j), 0)
                    ),
                )
            end
        end
    end

    @objective(model, Min, sum(theta[k] for k in 1:length(decomp_units)))
    return model, (x = x, theta = theta)
end

function _build_prod_key_used(ctx, unit_ind_list, sparse_cuts::Bool, with_capacity_neighbors::Bool)
    data = ctx.data
    if !sparse_cuts
        return ctx.prod_key
    end
    item_used = Set(unit_ind_list)
    for r in eachrow(data.df_bom)
        if r.II in unit_ind_list
            push!(item_used, r.I)
        end
    end
    if with_capacity_neighbors
        plant_list = Set(r.J for r in eachrow(data.df_prod) if r.I in unit_ind_list)
        for r in eachrow(data.df_prod)
            if r.J in plant_list
                push!(item_used, r.I)
            end
        end
    end
    return [(r.I, r.J) for r in eachrow(data.df_prod) if r.I in item_used]
end

function _subproblem_common(ctx, k; sparse_cuts::Bool, with_capacity_neighbors::Bool, deviation_ub = :max_prod)
    """
    _subproblem_common(ctx, k; sparse_cuts, with_capacity_neighbors, deviation_ub=:max_prod)

    Build the shared JuMP subproblem used by `pi_iter_sparse`, `pi_iter_sparse_feas`,
    and `pi_iter_sparse_opt`.

    This routine:
    - selects the decomposition unit `k` (the item cluster handled by one subproblem),
    - filters transit/alternate/production structures to a sparse working set,
    - creates all primal variables for inventory flow and production-copy constraints,
    - adds common material-balance, capacity, alternate-use, and inventory bound constraints

    Keyword options:
    - `sparse_cuts`: if `true`, use a reduced production key set.
    - `with_capacity_neighbors`: if `true`, include items sharing capacity resources.
    - `deviation_ub`: controls upper bound on deviation variables:
    `:max_prod` (feasibility/common case) or `:tiny` (near-fixed opt-cut case).

    Returns a named tuple containing:
    - `sp`: JuMP/Gurobi model,
    - index sets (`unit_ind_list`, `periods`, `plants`, `prod_key_used`, `transit_rows`),
    - `sub_fix_x`: master-linking constraint references (for dual extraction),
    - `vars`: a named tuple of JuMP variable containers used by caller-specific objectives.
    """

    data = ctx.data
    unit_ind_list = ctx.decomp_units[k]
    periods = sort(collect(data.period_df.I))
    first_t = minimum(periods)
    plants = collect(data.plant_df.I)

    transit_rows = [r for r in eachrow(data.df_transit) if r.I in unit_ind_list]
    transit_ids = [Int(r.Tr) for r in transit_rows]
    alt_idx = [i for i in 1:nrow(data.df_alt) if data.df_alt.I[i] in unit_ind_list || data.df_alt.II[i] in unit_ind_list]
    prod_key_used = _build_prod_key_used(ctx, unit_ind_list, sparse_cuts, with_capacity_neighbors)

    sp = Model(Gurobi.Optimizer)
    set_optimizer_attribute(sp, "OutputFlag", 0)
    set_optimizer_attribute(sp, "DualReductions", 0)
    set_optimizer_attribute(sp, "Threads", 1)
    set_optimizer_attribute(sp, "Method", 1)

    @variable(sp, ui[unit_ind_list, periods] >= 0)
    @variable(sp, si[transit_ids, periods] >= 0)
    @variable(sp, zi_p[unit_ind_list, plants, periods] >= 0)
    @variable(sp, vi[unit_ind_list, plants, periods] >= 0)
    @variable(sp, yUIi[unit_ind_list, plants, periods] >= 0)
    @variable(sp, yUOi[unit_ind_list, plants, periods] >= 0)
    @variable(sp, xCi[prod_key_used, periods] >= 0)
    @variable(sp, x_im_p[prod_key_used, periods] >= 0)
    @variable(sp, x_im_m[prod_key_used, periods] >= 0)
    @variable(sp, rCi[alt_idx] >= 0)

    for (i, j) in prod_key_used, t in periods
        ubv = deviation_ub == :tiny ? 1e-6 : get(ctx.max_prod, (i, j), 0.0)
        set_upper_bound(x_im_p[(i, j), t], ubv)
        set_upper_bound(x_im_m[(i, j), t], ubv)
    end

    @constraint(sp, unmet_demand[i in unit_ind_list, t in periods],
        ui[i, t] - (t == first_t ? 0.0 : ui[i, t - 1]) + sum(zi_p[i, j, t] for j in plants) == get(ctx.real_demand, (i, t), 0.0))
    @constraint(sp, inventory[i in unit_ind_list, j in plants, t in periods],
        vi[i, j, t] - (t == first_t ? get(ctx.init_inv, (i, j), 0.0) : vi[i, j, t - 1]) - yUIi[i, j, t] + yUOi[i, j, t] == 0)

    prod_key_used_set = Set(prod_key_used)
    @constraint(
        sp,
        input_item[i in unit_ind_list, j in plants, t in periods],
        yUIi[i, j, t] ==
        sum(
            (t - get(ctx.transit_time, Int(r.Tr), 0) >= first_t ? si[Int(r.Tr), t - get(ctx.transit_time, Int(r.Tr), 0)] : 0.0)
            for r in transit_rows if r.I == i && r.JJ == j; init=0.0
        ) +
        (haskey(ctx.lead_time, (i, j)) && ((i, j) in prod_key_used_set) && t - ctx.lead_time[(i, j)] >= first_t ? xCi[(i, j), t - ctx.lead_time[(i, j)]] : 0.0) +
        get(ctx.external_purchase, (i, j, t), 0.0) +
        sum(
            data.df_alt.V[id] * rCi[id]
            for id in alt_idx
            if data.df_alt.J[id] == j && data.df_alt.Ti[id] == t && data.df_alt.II[id] == i; init=0.0
        )
    )

    @constraint(
        sp,
        output_item[i in unit_ind_list, j in plants, t in periods],
        yUOi[i, j, t] ==
        sum(si[Int(r.Tr), t] for r in transit_rows if r.I == i && r.J == j; init=0.0) +
        sum(
            data.df_bom.V[bk] * xCi[(data.df_bom.I[bk], j), t]
            for bk in 1:nrow(data.df_bom)
            if data.df_bom.J[bk] == j && data.df_bom.II[bk] == i && ((data.df_bom.I[bk], j) in prod_key_used_set); init=0.0
        ) +
        zi_p[i, j, t] +
        sum(rCi[id] for id in alt_idx if data.df_alt.J[id] == j && data.df_alt.Ti[id] == t && data.df_alt.I[id] == i; init=0.0)
    )

    @constraint(sp, batch_cons[pk in prod_key_used, t in periods], xCi[pk, t] <= get(ctx.max_prod, pk, 0.0))
    @constraint(sp, capacity_cons[(ct, j, t) in keys(ctx.max_cap)],
        sum(get(ctx.unit_cap, (i2, j2, ct2), 0.0) * xCi[(i2, j2), t]
            for (i2, j2, ct2) in keys(ctx.unit_cap) if j2 == j && ct2 == ct && ((i2, j2) in prod_key_used); init=0.0) <= ctx.max_cap[(ct, j, t)]
    )

    @constraint(sp, r_ub[id in alt_idx], rCi[id] <= (data.df_alt.Ti[id] == first_t ? get(ctx.init_inv, (data.df_alt.I[id], data.df_alt.J[id]), 0.0) : vi[data.df_alt.I[id], data.df_alt.J[id], data.df_alt.Ti[id] - 1]))
    @constraint(sp, yo_ub[i in unit_ind_list, j in plants, t in periods], yUOi[i, j, t] <= (t == first_t ? get(ctx.init_inv, (i, j), 0.0) : vi[i, j, t - 1]))

    @constraint(sp, sub_fix_x[(i, j) in prod_key_used, t in periods], xCi[(i, j), t] + x_im_p[(i, j), t] - x_im_m[(i, j), t] == 0.0)

    return sp, prod_key_used;
end

function _subproblem_fixing_cons(sp, ctx, prod_key_used, x_vals)
    # fix the right hand side of the sub_fix_x constraints
    periods = sort(collect(ctx.data.period_df.I));

    for (i, j) in prod_key_used, t in periods
        set_normalized_rhs(sp[:sub_fix_x][(i, j), t], x_vals[(i, j), t] * get(ctx.lot_size, (i, j), 0.0));
    end
end

function _subproblem_fixing_obj(sp, ctx, prod_key_used, k; feas_cut_flag::Bool = false, feas_cut_threshold::Float64 = 1e-2, opt_cut_threshold::Float64 = 1e-4)
    # set up the objective function of the subproblem
    plants = collect(ctx.data.plant_df.I);
    periods = sort(collect(ctx.data.period_df.I));
    unit_ind_list = ctx.decomp_units[k];
    transit_rows = [r for r in eachrow(ctx.data.df_transit) if r.I in unit_ind_list];

    if feas_cut_flag
        @objective(sp, Min, sum(sp[:x_im_p][pk, t] + sp[:x_im_m][pk, t] for pk in prod_key_used for t in periods));
    else
        penalty_mag = length(ctx.data.period_df.I) * maximum(values(ctx.penalty_cost));
        @objective(sp, Min, sum(ctx.holding_cost[i] * sp[:vi][i, j, t] for i in unit_ind_list for j in plants for t in periods) + 
            sum(ctx.penalty_cost[i] * sp[:ui][i, t] for i in unit_ind_list for t in periods) + 
            sum(ctx.transit_cost[r.Tr] * sp[:si][r.Tr, t] for r in transit_rows for t in periods) + 
            sum(penalty_mag * (sp[:x_im_p][pk, t] + sp[:x_im_m][pk, t]) for pk in prod_key_used for t in periods));
    end
end

"""
Per-decomposition-unit Benders subproblem solves (feasibility then optimality objective).
Designed for `Threads.@threads`: only touches `sp` and read-only `ctx`; master cuts are added later via `_master_add_benders_cuts!`.
"""
function _benders_subproblem_cut_info(ctx, sp, prod_key_used, k, x_vals, theta_k; feas_opt_cut_flag::Bool, feas_cut_threshold::Float64 = 1e-2, opt_cut_threshold::Float64 = 1e-4)
    _subproblem_fixing_cons(sp, ctx, prod_key_used, x_vals)
    _subproblem_fixing_obj(sp, ctx, prod_key_used, k; feas_cut_flag = feas_opt_cut_flag)
    optimize!(sp)

    feas_info = nothing
    if objective_value(sp) > feas_cut_threshold
        feas_value = objective_value(sp)
        feas_cut_coeff = Dict()
        for (i, j) in prod_key_used, t in ctx.data.period_df.I
            feas_cut_coeff[(i, j, t)] = dual(sp[:sub_fix_x][(i, j), t]) * get(ctx.lot_size, (i, j), 0.0)
        end
        feas_info = (feas_value, feas_cut_coeff)
    end

    _subproblem_fixing_obj(sp, ctx, prod_key_used, k; feas_cut_flag = false)
    optimize!(sp)
    sub_obj_vals = objective_value(sp)

    opt_coeff = nothing
    if sub_obj_vals > theta_k * (1 + opt_cut_threshold)
        opt_cut_coeff = Dict()
        for (i, j) in prod_key_used, t in ctx.data.period_df.I
            opt_cut_coeff[(i, j, t)] = dual(sp[:sub_fix_x][(i, j), t]) * get(ctx.lot_size, (i, j), 0.0)
        end
        opt_coeff = opt_cut_coeff
    end

    return (; feas_info, sub_obj_vals, opt_coeff)
end

function _master_add_benders_cuts!(
    mp,
    ctx,
    prod_key_used_list,
    x_vals,
    theta_vals,
    results::AbstractVector,
)
    feas_gen = false
    sub_obj_vals_list = Dict{Int, Float64}()
    n = length(results)
    for k in 1:n
        r = results[k]
        pk_used = prod_key_used_list[k]
        if r.feas_info !== nothing
            feas_gen = true
            feas_value, feas_cut_coeff = r.feas_info
            @constraint(mp, sum(feas_cut_coeff[(i, j, t)] * (mp[:x][(i, j), t] - x_vals[(i, j), t]) for (i, j) in pk_used for t in ctx.data.period_df.I) + feas_value <= 0)
            println("Feasibility cut added for unit $(ctx.decomp_units[k]), feasibility violation: $(feas_value)")
        end
        sub_obj_vals_list[k] = r.sub_obj_vals
        if r.opt_coeff !== nothing
            opt_cut_coeff = r.opt_coeff
            sub_obj_vals = r.sub_obj_vals
            @constraint(mp, mp[:theta][k] >= sum(opt_cut_coeff[(i, j, t)] * (mp[:x][(i, j), t] - x_vals[(i, j), t]) for (i, j) in pk_used for t in ctx.data.period_df.I) + sub_obj_vals)
            println("Optimality cut added for unit $(ctx.decomp_units[k]), optimality violation: $(sub_obj_vals - theta_vals[k])")
        end
    end
    return feas_gen, sub_obj_vals_list
end