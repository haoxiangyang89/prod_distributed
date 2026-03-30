using Dates

include(joinpath(@__DIR__, "readin.jl"))
include(joinpath(@__DIR__, "extensive.jl"))

function compare_inputs(data)
    println("=== Input Diagnostics ===")
    println(
        "rows: ",
        "df_cost=", nrow(data.df_cost), ", ",
        "df_alt=", nrow(data.df_alt), ", ",
        "df_iniInv=", nrow(data.df_iniInv), ", ",
        "df_external=", nrow(data.df_external), ", ",
        "df_unitCap=", nrow(data.df_unitCap), ", ",
        "df_prod=", nrow(data.df_prod), ", ",
        "df_transit=", nrow(data.df_transit), ", ",
        "df_transitTC=", nrow(data.df_transitTC), ", ",
        "df_bom=", nrow(data.df_bom), ", ",
        "df_maxCap=", nrow(data.df_maxCap), ", ",
        "df_demand=", nrow(data.df_demand),
    )
    println(
        "dims: ",
        "I_norm=", data.I_norm, ", ",
        "J_norm=", data.J_norm, ", ",
        "T_norm=", data.T_norm, ", ",
        "Ct_norm=", data.Ct_norm, ", ",
        "Tr_norm=", data.Tr_norm,
    )
    println(
        "sums: ",
        "df_alt.V=", sum(data.df_alt.V), ", ",
        "df_iniInv.V=", sum(data.df_iniInv.V), ", ",
        "df_external.V=", sum(data.df_external.V), ", ",
        "df_unitCap.V=", sum(data.df_unitCap.V), ", ",
        "df_prod.MaxProd=", sum(data.df_prod.MaxProd), ", ",
        "df_prod.LS=", sum(data.df_prod.LS), ", ",
        "df_transitTC.TC=", sum(data.df_transitTC.TC), ", ",
        "df_bom.V=", sum(data.df_bom.V), ", ",
        "df_maxCap.V=", sum(data.df_maxCap.V), ", ",
        "df_demand.V=", sum(data.df_demand.V),
    )
end

function run_main_extensive(; data_folder = joinpath(dirname(@__DIR__), "data", "fine_tune"), print_diagnostics = true)
    println("=== Extensive Formulation Runner ===")
    println("Start time: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Data folder: ", data_folder)

    if !isdir(data_folder)
        error("Data folder not found: $data_folder")
    end

    t0 = time()
    data = readin_array(data_folder)
    println("Data loaded in $(round(time() - t0, digits=2)) seconds.")
    if print_diagnostics
        compare_inputs(data)
    end

    t1 = time()
    sol, obj, _ = extensive(data)
    solve_time = time() - t1

    println("Solved extensive formulation.")
    println("Objective value: ", obj)
    println("Solve time: ", round(solve_time, digits=2), " seconds.")
    println("Production variable count in solution dict: ", length(sol))
    println("Total elapsed: ", round(time() - t0, digits=2), " seconds.")

    return (sol = sol, obj = obj, solve_time = solve_time, data = data)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_main_extensive()
end
