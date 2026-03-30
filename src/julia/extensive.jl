using JuMP
using Gurobi

function extensive(data; optimizer = Gurobi.Optimizer)
    df_cost = data.df_cost
    df_alt = data.df_alt
    df_iniInv = data.df_iniInv
    df_external = data.df_external
    df_unitCap = data.df_unitCap
    df_prod = data.df_prod
    df_transit = data.df_transit
    df_transitTC = data.df_transitTC
    df_bom = data.df_bom
    df_maxCap = data.df_maxCap
    df_demand = data.df_demand
    item_df = data.item_df
    plant_df = data.plant_df
    period_df = data.period_df

    items = collect(item_df.I)
    plants = collect(plant_df.I)
    periods = sort(collect(period_df.I))
    first_t = minimum(periods)

    prod_key = [(r.I, r.J) for r in eachrow(df_prod)]
    lot_size = Dict((r.I, r.J) => r.LS for r in eachrow(df_prod))
    max_prod = Dict((r.I, r.J) => r.MaxProd for r in eachrow(df_prod))
    unit_cap = Dict((r.I, r.J, r.Ct) => r.V for r in eachrow(df_unitCap))
    max_cap = Dict((r.Ct, r.J, r.Ti) => r.V for r in eachrow(df_maxCap))
    lead_time = Dict((r.I, r.J) => Int(round(r.LT)) for r in eachrow(df_prod))
    real_demand = Dict((r.I, r.Ti) => r.V for r in eachrow(df_demand))
    external_purchase = Dict((r.I, r.J, r.Ti) => r.V for r in eachrow(df_external))
    init_inv = Dict((r.I, r.J) => r.V for r in eachrow(df_iniInv))
    holding_cost = Dict(r.I => r.HC for r in eachrow(df_cost))
    penalty_cost = Dict(r.I => r.PC for r in eachrow(df_cost))
    transit_cost = Dict(r.Tr => r.TC for r in eachrow(df_transitTC))
    transit_time = Dict(r.Tr => Int(round(r.V)) for r in eachrow(df_transitTC))

    model = Model(optimizer)

    @variable(model, x[prod_key, periods] >= 0, Int)
    @variable(model, ui[items, periods] >= 0)
    @variable(model, si[df_transit.Tr, periods] >= 0)
    @variable(model, zi[items, plants, periods] >= 0)
    @variable(model, vi[items, plants, periods] >= 0)
    @variable(model, yUIi[items, plants, periods] >= 0)
    @variable(model, yUOi[items, plants, periods] >= 0)
    @variable(model, rCi[1:nrow(df_alt)] >= 0)

    @constraint(model, [pk in prod_key, t in periods], x[pk, t] * lot_size[pk] <= max_prod[pk])

    for r in eachrow(df_maxCap)
        ct, j, t = r.Ct, r.J, r.Ti
        @constraint(
            model,
            sum(get(unit_cap, (i2, j2, ct2), 0.0) * x[(i2, j2), t] * lot_size[(i2, j2)]
                for (i2, j2, ct2) in keys(unit_cap)
                if j2 == j && ct2 == ct && haskey(lot_size, (i2, j2)); init=0.0) <= max_cap[(ct, j, t)],
        )
    end

    @constraint(model, [i in items, t in periods],
        ui[i, t] - (t == first_t ? 0.0 : ui[i, t - 1]) + sum(zi[i, j, t] for j in plants) == get(real_demand, (i, t), 0.0))

    @constraint(model, [i in items, j in plants, t in periods],
        vi[i, j, t] - (t == first_t ? get(init_inv, (i, j), 0.0) : vi[i, j, t - 1]) - yUIi[i, j, t] + yUOi[i, j, t] == 0)

    for i in items, j in plants, t in periods
        in_transit = sum(
            (t - get(transit_time, Int(r.Tr), 0) >= first_t ? si[Int(r.Tr), t - get(transit_time, Int(r.Tr), 0)] : 0.0)
            for r in eachrow(df_transit) if r.I == i && r.JJ == j
        ; init=0.0)
        prod_term = haskey(lead_time, (i, j)) && (t - lead_time[(i, j)] >= first_t) ? x[(i, j), t - lead_time[(i, j)]] * lot_size[(i, j)] : 0.0
        ext_term = get(external_purchase, (i, j, t), 0.0)
        alt_term = sum(
            df_alt.V[k] * rCi[k]
            for k in 1:nrow(df_alt)
            if df_alt.J[k] == j && df_alt.Ti[k] == t && df_alt.II[k] == i
        ; init=0.0)
        @constraint(model, yUIi[i, j, t] == in_transit + prod_term + ext_term + alt_term)
    end

    for i in items, j in plants, t in periods
        out_transit = sum(si[Int(r.Tr), t] for r in eachrow(df_transit) if r.I == i && r.J == j; init=0.0)
        bom_term = sum(
            df_bom.V[k] * x[(df_bom.I[k], j), t] * lot_size[(df_bom.I[k], j)]
            for k in 1:nrow(df_bom)
            if df_bom.J[k] == j && df_bom.II[k] == i && haskey(lot_size, (df_bom.I[k], j))
        ; init=0.0)
        alt_term = sum(
            df_alt.V[k] * rCi[k]
            for k in 1:nrow(df_alt)
            if df_alt.J[k] == j && df_alt.Ti[k] == t && df_alt.I[k] == i
        ; init=0.0)
        @constraint(model, yUOi[i, j, t] == out_transit + bom_term + zi[i, j, t] + alt_term)
    end

    @constraint(model, [k in 1:nrow(df_alt)],
        rCi[k] <= (df_alt.Ti[k] == first_t ? get(init_inv, (df_alt.I[k], df_alt.J[k]), 0.0) : vi[df_alt.I[k], df_alt.J[k], df_alt.Ti[k] - 1]))
    @constraint(model, [i in items, j in plants, t in periods],
        yUOi[i, j, t] <= (t == first_t ? get(init_inv, (i, j), 0.0) : vi[i, j, t - 1]))

    @objective(model, Min,
        sum(get(holding_cost, i, 0.0) * vi[i, j, t] for i in items, j in plants, t in periods) +
        sum(get(penalty_cost, i, 0.0) * ui[i, t] for i in items, t in periods) +
        sum(get(transit_cost, Int(r.Tr), 0.0) * si[Int(r.Tr), t] for r in eachrow(df_transit), t in periods)
    )

    optimize!(model)

    sol = Dict{Tuple{Int, Int, Int}, Float64}()
    for (i, j) in prod_key, t in periods
        sol[(i, j, t)] = value(x[(i, j), t])
    end

    return sol, objective_value(model), model
end
