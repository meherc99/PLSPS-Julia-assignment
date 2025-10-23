# render.jl

using CairoMakie
CairoMakie.activate!(type = "png")

include("../src/parallel.jl")

#=============================================================================
                            RENDER FUNCTIONS
=============================================================================#

function render_fractal(FP::FractalParams, data::Array{Float64, 3}, file_name::String, fps::Int, color::Symbol, show_labels::Bool)
    
    N, F, M, center, alpha, delta = extract_params(FP)

    @assert size(data) == (N, N, F) "Data size mismatch: expected ($N, $N, $F), got $(size(data))"

    # Define output file, and create figure
    output_file = joinpath("output", file_name * ".gif")
    mkpath(dirname(output_file))
    fig = Figure(size = (N, N))
    ax = Axis(fig[1, 1], xlabel = "Re", ylabel = "Im")

    # Record frames 
    record(fig, output_file, 1:F; framerate = fps) do f

        empty!(ax)  # Clear previous frame

        # Calculate range in complex plane  
        local_delta = delta * alpha^(f - 1)
        range_x = LinRange(center[1] - local_delta, center[1] + local_delta, N)
        range_y = LinRange(center[2] - local_delta, center[2] + local_delta, N)

        # Create heatmap in figure
        hm = heatmap!(ax, range_x, range_y, data[:, :, f], colormap = color)

        # Show grid lines, labels, ticks, colorbar
        if show_labels
            if f == 1
                Colorbar(fig[1, 2], hm, label="iterations")
            end
        else 
            hidedecorations!(ax)
        end 

        println("[FRAME $f/$F RENDERED]")
    end
end

#=============================================================================
                                MAIN 
=============================================================================#

function main()
    
    # FRACTAL PARAMETERS
    N = 1024                    # Resolution (N×N pixels)
    F = 30                      # Number of frames
    M = 100                     # Maximum iterations per pixel
    center = (-0.7269, 0.1889)  # Complex plane center
    alpha = 0.9                 # Zoom scaling factor
    delta = 0.5                 # Initial region half-width
    FP = FractalParams(N, F, M, center, alpha, delta)

    # RENDER PARAMETERS
    file_name = "fractal"       # Output file name
    fps = 5                     # Frames per second
    color = :oslo               # Colormap to use
    show_labels = false         # Show colorbar and axis labels

    println("="^50)
    if nprocs() == 1
        println("RENDER: SEQUENTIAL")
    else
        println("RENDER: P=$(nworkers())")
    end
    println("-"^25)
    println("Fractal:     $(output_params(FP))")
    println("File Name:   $(joinpath("output", file_name * ".gif"))")
    println("FPS:         $fps")
    println("Color:       $color")
    println("Show Labels: $show_labels")
    println("="^50)

    # RUN SEQUENTIAL OR PARALLEL
    P = nprocs()
    
    if P == 1
        _, data = run_seq(FP)
    else
        _, data = run_par(FP)
    end

    # RENDER FRACTAL
    render_fractal(FP, data, file_name, fps, color, show_labels)
end

if !isinteractive()
    main()
end