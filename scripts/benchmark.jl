# benchmark.jl

using JSON

include("../src/parallel.jl")

#=============================================================================
                            BENCHMARK FUNCTIONS
=============================================================================#

# Run the implementation for a number of runs and return the minimum timing
function benchmark_fractal(FP::FractalParams, nruns::Int)
 
    timings = Float64[]
    for i in 1:nruns
        t, _ = (nprocs() == 1) ? run_seq(FP) : run_par(FP)
        push!(timings, t)
    end

    return minimum(timings)
end

# Save the benchmark result to a JSON file
function save_benchmark_result(FP::FractalParams, min_timing::Float64)

    P = (nprocs() == 1) ? 0 : nworkers() 

    results = Dict{String, Any}(
        "P" => P,
        "N" => FP.N,
        "F" => FP.F,
        "M" => FP.M,
        "T" => min_timing
    )

    mkpath("results")   

    filename = "results/benchmark_P$(P)_N$(FP.N)_F$(FP.F)_M$(FP.M).json"
    open(filename, "w") do f
        JSON.print(f, results, 4) 
    end
    
    return filename
end

# Load a benchmark result from a JSON file as a dictionary to use in Julia
function load_benchmark_result(filename::String)::Dict{String,Any}
    if !isfile(filename)
        error("File not found: '$filename'")
    end

    return open(filename, "r") do f
        JSON.parse(f)
    end
end

# List of FractalParams to benchmark
function benchmark_list()
    # If the job timeouts, comment out the test cases and run them one at a time.
    return [
        FractalParams(1024, 30),
        FractalParams(2048, 5),
        FractalParams(512, 120),
    ]
end

#=============================================================================
                              MAIN
=============================================================================#

function main()

    P = nprocs()

    # BENCHMARK PARAMETERS
    nruns = 5               # The number of runs to perform per benchmark

    println("="^50)
    if P == 1
        println("BENCHMARK: SEQUENTIAL")
    else
        println("BENCHMARK: P=$(nworkers())")
    end
    println("-"^25)
    println("Runs per Benchmark: $nruns")
    println("="^50)

    # BENCHMARK LIST 
    FPs = benchmark_list()

    # WARMUP RUN (FOR JULIA TO COMPILE FUNCTIONS)
    P == 1 ? run_seq(FPs[1]) : run_par(FPs[1])

    # BENCHMARK AND SAVE RESULTS
    for (i, FP) in enumerate(FPs)
        try
            min_timing = benchmark_fractal(FP, nruns)
            filename = save_benchmark_result(FP, min_timing)
            println("[BENCHMARK $(i)/$(length(FPs)) COMPLETE: '$(filename)']")
        catch e
            println("ERROR: $(e)")
        end
    end

    println()
end

if !isinteractive()
    main()
end
