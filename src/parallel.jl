# parallel.jl

using Distributed
@everywhere include("sequential.jl") # Including sequential everywhere as all workers use the run_seq() function

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

# Master 
function generate_fractal_par!(FP::FractalParams, data::Array{Float64, 3})

    N, F, max_iters, center, alpha, delta = extract_params(FP)

    P = nworkers() # P + 1 processors - 1 Master + P Workers

    # t will be the total time spent in this code
    t = @elapsed begin  

        # Master logic here
        
    end 

end

# Worker
function work(FP::FractalParams, job_channel::RemoteChannel, result_channel::RemoteChannel)

    N, F, max_iters, center, alpha, delta = extract_params(FP)

    # Worker logic here

end

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

function run_par(FP::FractalParams)
    data = zeros(Float64, FP.N, FP.N, FP.F)
    t = generate_fractal_par!(FP, data)
    return t, data
end


