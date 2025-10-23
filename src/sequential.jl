# sequential.jl

#=============================================================================
                           MANDELBROT
=============================================================================#

# Computes the sequence z_{n+1} = z_n^2 + c for a complex point c = x + iy for 'M' iterations
function mandelbrot(c::Complex, M::Int)

    z = c # z_0 = 0 + c
    
    for iter in 1:M
        # This is equivalent to abs(z) > 2, but is more efficient since it avoids the square root calculation
        if z.re^2 + z.im^2 > 4
            return smooth_iterations(iter, M, z) # Outside the Mandelbrot set
        end 
        z = z^2 + c  # Mandelbrot iteration
    end
    
    return M # Inside the Mandelbrot set
end

# Smooth the mandelbrot output to create a continuous coloring effect. 
# https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Continuous_(smooth)_coloring
function smooth_iterations(iter::Int, M::Int, curr_z::Complex)
    return iter - (log(log(abs(curr_z))) - log(log(M)))/log(2.0)
end

#=============================================================================
                           FRACTAL PARAMETERS
=============================================================================#

struct FractalParams
    N::Int                         # Resolution (N×N pixels)
    F::Int                         # Number of frames
    M::Int                         # Maximum iterations per pixel
    center::Tuple{Float64,Float64} # Complex plane center
    alpha::Float64                 # Zoom scaling factor
    delta::Float64                 # Initial region half-width

    function FractalParams(N, F, M, center, alpha, delta)
        @assert N > 0           "N must be > 0"
        @assert F > 0           "F must be > 0"
        @assert M > 0           "M must be > 0"
        @assert 0 < alpha < 1   "alpha must be in (0, 1)"
        @assert delta > 0       "delta must be > 0"
        new(N, F, M, center, alpha, delta)
    end
end

# Default Constructor 
function FractalParams(N::Int, 
                       F::Int; 
                       M::Int                          = 100,
                       center::Tuple{Float64, Float64} = (-0.7269, 0.1889),
                       alpha::Float64                  = 0.9,
                       delta::Float64                  = 0.5)
    return FractalParams(N, F, M, center, alpha, delta)
end

# Utility function to extract parameters
function extract_params(FP::FractalParams)
    return FP.N, FP.F, FP.M, FP.center, FP.alpha, FP.delta
end

function output_params(FP::FractalParams)
    N, F, M, center, alpha, delta = extract_params(FP)
    return "FP(N=$N, F=$F, M=$M, center=$center, alpha=$alpha, delta=$delta)"
end


#=============================================================================
                           SEQUENTIAL IMPLEMENTATION
=============================================================================#

# Generate a zoomed fractal sequence for the mandelbrot sequentially
function generate_fractal_seq!(FP::FractalParams,
                               data::Array{Float64, 3})

    N, F, M, center, alpha, delta = extract_params(FP)

    @assert size(data) == (N, N, F)  "Size mismatch: data has size $(size(data)), but expected ($N, $N, $F)"

    t = @elapsed begin
        @inbounds for f=1:F                     # Frames
            # Image to complex plane transformation
            local_delta = delta * alpha^(f -1) 
            x_min = center[1] - local_delta 
            y_min = center[2] - local_delta
            dw = (2 * local_delta) / N

            @inbounds for j in 1:N              # Columns
                y = y_min + (j - 1) * dw        
                @inbounds for i in 1:N          # Rows
                    x = x_min + (i - 1) * dw
                    c = Complex(x, y)
                    data[i, j, f] = mandelbrot(c, M) 
                end
            end
        end
    end
    t
end

# Allocate data and run sequential implementation
function run_seq(FP::FractalParams)
    data = zeros(Float64, FP.N, FP.N, FP.F)
    t = generate_fractal_seq!(FP, data)
    return t, data
end