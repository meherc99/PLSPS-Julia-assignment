# parallel.jl

using Distributed

# Activate the project environment on all workers and install dependencies
@everywhere begin
    using Pkg
    Pkg.activate(dirname(@__DIR__))  # Activates the Assignment directory
    Pkg.instantiate()  # Install all project dependencies on each worker
end

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
        
        # Create channels for communication
        # job_channel: Master sends frame numbers to workers
        # result_channel: Workers send computed frames back to master
        job_channel = RemoteChannel(() -> Channel{Int}(F))
        result_channel = RemoteChannel(() -> Channel{Tuple{Int, Array{Float64, 2}}}(F))
        
        # Populate the job queue with all frame numbers
        for f in 1:F
            put!(job_channel, f)
        end
        
        # Signal workers that no more jobs will be added (close the channel)
        close(job_channel)
        
        # Start workers on all available worker processes
        @sync begin
            for worker_id in workers()
                @async remotecall_wait(work, worker_id, FP, job_channel, result_channel)
            end
        end
        
        # Collect results from workers
        for _ in 1:F
            (f, frame_data) = take!(result_channel)
            data[:, :, f] = frame_data
        end
        
    end 

    return t
end

# Worker - Define on all processes
@everywhere function work(FP::FractalParams, job_channel::RemoteChannel, result_channel::RemoteChannel)

    N, F, max_iters, center, alpha, delta = extract_params(FP)

    # Worker logic here
    
    # Keep processing frames until the job queue is empty
    while true
        local f
        try
            # Try to get a frame number from the job queue
            f = take!(job_channel)
        catch e
            # If the channel is closed and empty, exit gracefully
            # This catches InvalidStateException and any remote call exceptions
            break
        end
        
        # Allocate memory for this frame
        frame_data = zeros(Float64, N, N)
        
        # Compute the frame (same logic as sequential version)
        local_delta = delta * alpha^(f - 1)
        x_min = center[1] - local_delta
        y_min = center[2] - local_delta
        dw = (2 * local_delta) / N
        
        @inbounds for j in 1:N              # Columns
            y = y_min + (j - 1) * dw        
            @inbounds for i in 1:N          # Rows
                x = x_min + (i - 1) * dw
                c = Complex(x, y)
                frame_data[i, j] = mandelbrot(c, max_iters) 
            end
        end
        
        # Send the computed frame back to the master
        put!(result_channel, (f, frame_data))
    end

end

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

function run_par(FP::FractalParams)
    data = zeros(Float64, FP.N, FP.N, FP.F)
    t = generate_fractal_par!(FP, data)
    return t, data
end


