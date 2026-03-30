# The main procedure to run the decomposition algorithm in Julia
using JuMP, Gurobi, Random, Dates, Base.Threads

include(joinpath(@__DIR__, "readin.jl"))
include(joinpath(@__DIR__, "decomp_function.jl"))

function construct_data_ctx(data_folder)
    if !isdir(data_folder)
        error("Data folder not found: $data_folder")
    end

    t0 = time()
    data = readin_array(data_folder)
    println("Data loaded in $(round(time() - t0, digits=2)) seconds.")

    t1 = time()
    ctx = build_benders_context(data)
    println("Benders context built in $(round(time() - t1, digits=2)) seconds.")

    return (data = data, ctx = ctx)
end

function construct_master_problem(ctx;
    relax_option::Bool = false,
    level_obj_norm::Float64 = 2.0,
    logic_constr_flag::Bool = false,
)
    println("=== Master Problem Builder ===")
    println("Start time: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Data folder: ", data_folder)

    t2 = time()
    mp, mp_handle = master_prob(
        ctx;
        relax_option = relax_option,
        level_obj_norm = level_obj_norm,
        logic_constr_flag = logic_constr_flag,
    )
    println("Master model constructed in $(round(time() - t2, digits=2)) seconds.")
    println("Decomposition unit count: ", length(ctx.decomp_units))

    return (mp = mp, mp_handle = mp_handle)
end

# read in the hyperparameters
# Accelerate technique 1: feasibility + optimality cut
feas_opt_cut_flag = true

# Accelerate technique 2: regularization - level_set / regularized BD
reg_BD_flag = true
level_fractile = 0.5
level_obj_norm = 2.0    # 1: 1-norm / 2:2-norm / 1.5:2-piece approximation of 2-norm
best_incum = Dict{Tuple{Int, Int, Int}, Float64}()

# Accelerate technique 3: LP warm start (Benders for LP relaxation)
lp_warm_flag = true
lp_warm_iter = 50      # Benders Maximum Iteration
lp_rel_gap_tol = 0.05   # Benders Gap Tolerance 5%

# Accelerate technique 4: Add logical constraint to master
logic_constr_flag = false

# Generate sparse cut?
sparse_cut_flag = false

# fixing technique (Fix production variables that remain unchanged for many iterations)
fixing_flag = false
fixing_x = Float64[]
fixing_round = Int[]
thres_FR = 50
if fixing_flag
    # Will be initialized after constructing the model, when keys are available.
    fixing_x = Float64[]
    fixing_round = Int[]
    thres_FR = 50
end

# gomory feasibility cut technique
gomory_feas_flag = false

# construct the master problem
data_folder = joinpath(@__DIR__, "../..", "data", "fine_tune");
global data, ctx = construct_data_ctx(data_folder);
global mp, mp_handle = construct_master_problem(ctx; relax_option = lp_warm_flag, level_obj_norm = level_obj_norm,
                                                logic_constr_flag = logic_constr_flag);

# construct all subproblems (parallel build; one Gurobi model per unit, each with Threads=1)
global subprob_list = Dict{Int, Any}();
global prod_key_used_list = Dict{Int, Any}();
let n_unit = length(ctx.decomp_units)
    sp_vec = Vector{Any}(undef, n_unit)
    pk_vec = Vector{Any}(undef, n_unit)
    @threads for k in 1:n_unit
        sp_vec[k], pk_vec[k] = _subproblem_common(ctx, k; sparse_cuts = sparse_cut_flag, with_capacity_neighbors = true, deviation_ub = :max_prod)
    end
    for k in 1:n_unit
        subprob_list[k] = sp_vec[k]
        prod_key_used_list[k] = pk_vec[k]
    end
end

# set up the record lists
LB_List = Float64[];
time_List = Float64[];
keep_iter = true;
max_iter = 100;
epsilon = 1e-4;

if lp_warm_flag
    UB_List_LP = Float64[];
    # solve the master problem for LP relaxation
    for iter_num in 1:lp_warm_iter
        start_time = time();
        # solve the master problem and obtain the solution
        optimize!(mp);
        LB = objective_value(mp);
        push!(LB_List, LB);
        x_vals = value.(mp[:x]);
        theta_vals = Dict();
        for k in 1:length(ctx.decomp_units)
            theta_vals[k] = value(mp[:theta][k]);
        end
        n_unit = length(ctx.decomp_units);
        results = Vector{Any}(undef, n_unit);
        @threads for k in 1:n_unit
            results[k] = _benders_subproblem_cut_info(
                ctx,
                subprob_list[k],
                prod_key_used_list[k],
                k,
                x_vals,
                theta_vals[k];
                feas_opt_cut_flag = feas_opt_cut_flag,
            );
        end
        feas_gen, sub_obj_vals_list = _master_add_benders_cuts!(mp, ctx, prod_key_used_list, x_vals, theta_vals, results);

        if !feas_gen
            UB = sum(sub_obj_vals_list[k] for k in 1:n_unit);
            push!(UB_List_LP, UB);
        end
        push!(time_List, time() - start_time);
    end

    for pkey in ctx.prod_key
        for t in ctx.data.period_df.I
            set_integer(mp[:x][pkey, t]);
        end
    end
end

# solve the master problem for the final solution
iter_num = 0;
UB_List = Float64[];
while keep_iter
    iter_num += 1;
    start_time = time();
    optimize!(mp);
    LB = objective_value(mp);
    push!(LB_List, LB);
    x_vals = value.(mp[:x]);
    theta_vals = Dict();
    for k in 1:length(ctx.decomp_units)
        theta_vals[k] = value(mp[:theta][k]);
    end
    n_unit = length(ctx.decomp_units);
    results = Vector{Any}(undef, n_unit);
    @threads for k in 1:n_unit
        results[k] = _benders_subproblem_cut_info(
            ctx,
            subprob_list[k],
            prod_key_used_list[k],
            k,
            x_vals,
            theta_vals[k];
            feas_opt_cut_flag = feas_opt_cut_flag,
        );
    end
    feas_gen, sub_obj_vals_list = _master_add_benders_cuts!(mp, ctx, prod_key_used_list, x_vals, theta_vals, results);

    if !feas_gen
        UB = sum(sub_obj_vals_list[k] for k in 1:n_unit);
        push!(UB_List, UB);
    end

    # check the stopping criterion
    if (minimum(UB_List, init = Inf) - LB) / minimum(UB_List, init = Inf) < epsilon
        keep_iter = false;
    end
    if iter_num > max_iter
        keep_iter = false;
    end
    elapsed_time = time() - start_time;
    push!(time_List, elapsed_time);
    println(" =================== Iteration: $(iter_num), LB: $(LB), Best UB: $(minimum(UB_List, init = Inf)), Time: $(elapsed_time) seconds ===================");
end
