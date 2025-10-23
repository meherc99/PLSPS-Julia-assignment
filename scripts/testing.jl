# testing.jl

include("../src/parallel.jl")

#=============================================================================
                            TESTING FUNCTIONS
=============================================================================#

function test_fractal(FP::FractalParams)

    try
        # Run sequential and parallel implementations
        _, data_seq = run_seq(FP)
        _, data_par = run_par(FP)

        # Compare results
        error = maximum(abs.(data_seq - data_par)) 
        return (error <= 1e-6) ? true : false
    catch e
        return false
    end
end 

function test_list()
    # If the job timeouts, comment out the test cases and run them one at a time. 
    return [
        FractalParams(64, 5),
        FractalParams(1024, 25),
        FractalParams(2048, 2),
        FractalParams(256, 100)
    ]
end

#=============================================================================
                                MAIN 
=============================================================================#

function main()

    println("="^50)
    if nprocs() == 1
        println("TEST: SEQUENTIAL")
    else
        println("TEST: P=$(nworkers())")
    end
    println("="^50)

    # TEST LIST 
    FPs = test_list()

    # TEST FRACTALPARAMS
    passed_tests = 0
    total_tests = length(FPs)
    for (i, FP) in enumerate(FPs)
        if test_fractal(FP)
            passed_tests += 1
            println("[TEST $(i)/$(length(FPs)) PASSED]")
        else
            println("[TEST $(i)/$(length(FPs)) FAILED] - $(output_params(FP))")
        end
    end

    # OUTPUT TEST RESULTS 
    println("-"^25)
    success_rate = (passed_tests / total_tests) * 100
    println("PASS RATE: $passed_tests/$total_tests ($(round(success_rate, digits=2))%)\n")
end 

if !isinteractive()
    main()
end