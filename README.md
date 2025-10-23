# Fractal Zoom Animations in Julia
A Julia-based project for generating and visualising Mandelbrot fractal zoom animations. Can be run either sequentially, or in parallel using SLURM job scripts. 

## Table of Contents
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup Guide](#setup)
- [Source](#source)
- [Scripts](#scripts)
- [Jobs](#jobs)
- [Additional Resources](#additional-resources)

## Requirements
- **Julia**: 1.10.3 or later
- **System**: DAS-5 

## Project Structure
```
PLSPS2025Assignment/
├── README.md                    # Project Documentation
├── Project.toml                 # Julia dependencies
├── setup.sh                     # Environment setup script
│
├── src/                         # Core implementation
│   ├── sequential.jl           # Sequential implementation
│   └── parallel_student.jl     # Parallel implementation (YOUR WORK)
│
├── scripts/                     # Executable Julia scripts
│   ├── testing.jl              
│   ├── benchmark.jl            
│   ├── render.jl   
|   └── merge_results.sh                        
│
├── jobs/                        # SLURM jobs for Julia scripts
│   ├── testing_job.sh          
│   ├── benchmark_job.sh        
│   └── render_job.sh          
│
├── results/                     # Benchmark output files (auto-created)
└── output/                      # Generated renders (auto-created)
```

## Setup
Run the `setup.sh` script whenever you log into DAS-5 in a new shell session. 

```bash
source setup.sh
```
The script loads any required modules on DAS-5, and configures the Julia environment by installing the project dependencies listed in [`Project.toml`](Project.toml). 

```bash
# All available DAS-5 modules
module avail

# Check loaded DAS-5 modules
module list
```

## Source
### Sequential Implementation 

The [`src/sequential.jl`](src/sequential.jl) file contains the core data structures and algorithms. A `FractalParams` object stores all of the parameters needed to perform the mandelbrot fractal zoom. 

```julia
struct FractalParams
    N::Int                         # Frame Resolution (N×N pixels)
    F::Int                         # Number of frames
    M::Int                         # Maximum iterations per pixel in mandelbrot function
    center::Tuple{Float64,Float64} # Region center
    delta::Float64                 # Region half-width
    alpha::Float64                 # Zoom factor (0 < α < 1)
end
```

**Functions**
- `mandelbrot(c, M)`: 
Approximates if a complex number `c` is in the Mandelbrot set, by calculating the Mandelbrot recurrence up to `M` iterations. 
- `generate_fractal_seq!(FP, data)`: Generate Mandelbrot fractal zoom data sequentially. `data` is preallocated 3D array of size `(N, N, F)`, storing each frame. 
- `run_seq(FP)`: Runner for sequential implementation, returns a tuple `(timing::Float64, data::Array)` containing the execution time and the generated fractal data.

### Parallel Implementation (YOUR TASK)
The [`src/parallel.jl`](src/parallel.jl) file contains boilerplate code to implement a Replicated Workers model, using Julia Distributed. 

- `generate_fractal_par!(FP, data)` - Master logic.
- `work(FP)` - Worker logic.
- `run_par(FP)` - Runner for parallel implementation, returns a tuple `(timing::Float64, data::Array)` containing the execution time and the generated fractal data.

## Scripts
You are provided with julia script files for testing, benchmarking, and visualising your implementation. To use a script file, first update the script parameters in the `main` function, then run the script in the main julia process sequentially or in parallel.

```bash
# Sequential
julia --project=. scripts/script.jl      

# Parallel
julia -p 4 --project=. scripts/script.jl 

# Parallel with machine file
julia --project=. --machine-file script_machine_file.txt scripts/script.jl
```
Additional worker processors can be specified with the `-p P` option. In the example above `-p 4` adds 4 workers, this means that the program runs with a total of 5 processors (including the Master process). 

### Testing
The [`testing.jl`](scripts/testing.jl) script **verifies the correcteness** of your parallel implementation. 
- Runs a set of test cases, defined as `FractalParams(N, F)` objects in the `test_list` function.
- Compares the output of the sequential `run_seq` implementation and the parallel `run_par` implementation, for each test case. 
- The script outputs the number of workers used during testing, lists which tests have passed, and reports the overall pass rate as a percentage.

Example testing script output for P=(0, 1, 2, 4) with 2 test cases:
```
==================================================
TEST: SEQUENTIAL
==================================================
[TEST 1/2 PASSED]
[TEST 2/2 PASSED]
-------------------------
PASS RATE: 2/2 (100.0%)

==================================================
TEST: P=1
==================================================
[TEST 1/2 PASSED]
[TEST 2/2 PASSED]
-------------------------
PASS RATE: 2/2 (100.0%)

==================================================
TEST: P=2
==================================================
[TEST 1/2 PASSED]
[TEST 2/2 PASSED]
-------------------------
PASS RATE: 2/2 (100.0%)

==================================================
TEST: P=4
==================================================
[TEST 1/2 PASSED]
[TEST 2/2 PASSED]
-------------------------
PASS RATE: 2/2 (100.0%)
```
---
### Benchmark Script
The [`benchmark.jl`](scripts/benchmark.jl) script is used to **measure and compare the performance**  of the sequential and parallel implementations.
- Runs a set of benchmark cases, defined as `FractalParams` in the `benchmark_list` function.
- For each case, performs `nruns` executions and records the minimum timing `T`.
- Benchmarks either the sequential (`run_seq`) or parallel (`run_par`) implementation, depending on the number of processes (-p N).
- Saves the results for each benchmark as a JSON file.

Example benchmark script output for P=(0, 8, 32) with 3 benchmark cases:
```
==================================================
BENCHMARK: SEQUENTIAL
-------------------------
Runs per Benchmark: 5
==================================================
[BENCHMARK 1/3 COMPLETE: 'results/benchmark_P0_N512_F5_M300.json']
[BENCHMARK 2/3 COMPLETE: 'results/benchmark_P0_N256_F4_M300.json']
[BENCHMARK 3/3 COMPLETE: 'results/benchmark_P0_N128_F16_M300.json']

==================================================
BENCHMARK: P=8
-------------------------
Runs per Benchmark: 5
==================================================
[BENCHMARK 1/3 COMPLETE: 'results/benchmark_P8_N512_F5_M300.json']
[BENCHMARK 2/3 COMPLETE: 'results/benchmark_P8_N256_F4_M300.json']
[BENCHMARK 3/3 COMPLETE: 'results/benchmark_P8_N128_F16_M300.json']

==================================================
BENCHMARK: P=32
-------------------------
Runs per Benchmark: 5
==================================================
[BENCHMARK 1/3 COMPLETE: 'results/benchmark_P32_N512_F5_M300.json']
[BENCHMARK 2/3 COMPLETE: 'results/benchmark_P32_N256_F4_M300.json']
[BENCHMARK 3/3 COMPLETE: 'results/benchmark_P32_N128_F16_M300.json']
```

#### Result File
The benchmark results `benchmark_P$_N$_F$_M$.json` are saved in the [`results`](results/) directory.  The following is an example of the `benchmark_P8_N512_F5_M300.json` file.

```json
{
    "P": 8,                    // Number of workers 
    "N": 512,                  // Image Resolution (N*N)
    "F": 5,                    // Number of frames
    "M": 300,                  // Maxmum iterations
    "T": 0.361865167,          // Minimum timing
}
```

#### Process Results
To process the .json results, you can either use the the `merge_results.sh` script to merge all of the JSON files in the [`results`](results/) directory into a CSV file. 
```bash 
source scripts/merge_results.sh
```

Otherwise, you can use the `load_benchmark_result` function defined in the `benchmark.jl` script to load JSON data in Julia. 

```julia
include("scripts/benchmark.jl")

# Load specific result
result = load_benchmark_result("results/benchmark_P8_N512_F5_M300.json")

# Access data
println("Processes: $(result["P"])")
println("Resolution: $(result["N"])×$(result["N"])")
println("Frames: $(result["F"])")
println("Timing: $(result["T"])")
```
---
### Render Script

The [`render.jl`](scripts/render.jl) script is used to **generate and visualise a fractal animation** using either the sequential or parallel implementation.
- Specify the fractal parameters (`FractalParams`) and rendering parameters in the `main` function.
```julia
# Rendering parameters
file_name = "fractal"           # Output filename (saved as output/fractal.gif)
fps = 5                         # Animation framerate
color = :oslo                   # Colormap (e.g. :oslo, :viridis, :plasma)
show_labels = false             # Show axis labels and colorbar

# Fractal parameters 
FP = FractalParams(
    N = 1080,                   # Image resolution
    F = 10,                     # Number of frames in the animation
    M = 100,                    # Mandelbrot iterations per pixel
    center = (-0.7269, 0.1889), # Center of the zoom
    alpha = 0.9,                # Zoom factor per frame
    delta = 0.5                 # Pixel scale
)
```
- Automatically chooses `run_seq` or `run_par` depending on the number of processes (`-p N`) passed when executing the script.
- Renders a `.gif` animation and saves it to the [`output`](output/) directory.

Example render script output for P=32:
```bash 
==================================================
RENDER: P=32
-------------------------
Fractal:     FP(N=1080, F=10, M=100, center=(-0.743643887037151, 0.13182590420533), alpha=0.9, delta=0.05)
File Name:   output/fractal.gif
FPS:         5
Color:       oslo
Show Labels: false
==================================================
[FRAME 1/10 RENDERED]
[FRAME 2/10 RENDERED]
[FRAME 3/10 RENDERED]
[FRAME 4/10 RENDERED]
[FRAME 5/10 RENDERED]
[FRAME 6/10 RENDERED]
[FRAME 7/10 RENDERED]
[FRAME 8/10 RENDERED]
[FRAME 9/10 RENDERED]
[FRAME 10/10 RENDERED]
```

## Jobs
### What is SLURM?
[SLURM](https://slurm.schedmd.com/quickstart.html) (Simple Linux Utility for Resource Management) is a workload manager used to allocate resources (like CPUs and memory) and schedule jobs on computing clusters such as DAS-5. You submit jobs to the cluster, and SLURM takes care of queueing, running, and managing them.

In our case, a job is defined in the [`jobs`](jobs/) directory for each of the julia scripts. The job allows us to specify the number of processors to use when running your parallel implementation on DAS-5. 

### The Job Script
A job script is a `.sh` shell script with extra `#SBATCH` lines that set SLURM job options. These lines control how your job runs on the cluster. The jobs files in the [`jobs`](jobs/)  directory use the following options:

- `--exclusive` ensures your job has exclusive access to the allocated nodes. 
- `--time=00:15:00` sets a maximum runtime of 15 minutes. 
- `--nodes=2` requests two compute nodes
- `--ntasks-per-node=17` allocates 17 tasks (typically CPU cores) per node. 
- `--output=jobs/testing_%j.out` specifies the location and naming pattern for the output log file, where %j is replaced by the job ID. 

We use 2 nodes because we would like our processors to be distributed, with each node having half the processors (except for the case when `P=1`). The reason for requesting --ntasks-per-node=17 is that you will be testing with a maximum of 32 workers (which need to be distributed across two nodes). The total number of tasks needed is 33 (32 workers + 1 master process). Since this number is not divisible by two, we must round up to the nearest even number that is divisible by 2, so we end up with a total of 34 tasks, meaning 17 tasks per node. This ensures that a total of 34 tasks are available for the 33 processes.

You can run a job on DAS-5 using the `sbatch` command. 
```bash
sbatch jobs/job.sh     
```

### The Number of Processors

You can specify the number of processors for DAS-5 to use within the job script. The `render_job.sh` script takes a single processor count as input, while the `benchmark_job.sh` and `testing_job.sh` jobs allow you to specify an array of processors (`PROCS`) to perform multiple runs in a single job. 

```bash 
PROCS=(0 1 2 4 8 16 32) # 2^0 to 2^5 Workers 

for P in "${PROCS[@]}"; do
    #...
done
```

The script is run with a different number of processors based on the value of P:

When `P=0`, the script is run sequentially using the standard Julia call. 
```bash
if [ "$P" -eq 0 ]; then
    julia --project=. scripts/render.jl   
```

When `P=1`, an additional worker process is started in parallel using the `-p` option.
```bash
elif [[ "$P" -eq 1 ]]; then 
    julia -p 1 --project=. scripts/render.jl  
```

When `P>1`, the workers are distributed across the two nodes. Instead of specifying the total number of processors, the `--machine-file` option allows us to specfify how many processors are used per node. For more information, take a look at [Starting and managing worker processes in Julia](https://docs.julialang.org/en/v1/manual/distributed-computing/#Starting-and-managing-worker-processes).

```bash
# Parallel (P > 1) - Distribute workers across two nodes
else
    HALFP=$((P / 2))                                    

    # Create machine file to distribute workers in Julia
    HOSTS=($(scontrol show hostnames $SLURM_NODELIST))  
    echo "${HALFP}*${HOSTS[0]}" > render_machine_file.txt
    echo "${HALFP}*${HOSTS[1]}" >> render_machine_file.txt

    # Run the Julia script
    julia --project=. --machine-file render_machine_file.txt scripts/render.jl

    # Clean up
    rm render_machine_file.txt
fi
```

### SLURM Commands
Once submitted, you can monitor and manage your jobs using SLURM commands:

```bash
# Check cluster/partition status
sinfo

# View your queued or running jobs
squeue -u $USER

# View job output log (replace JOBID with actual ID)
cat slurm-JOBID.out

# View detailed info about a specific job
scontrol show job JOBID

# Cancel a job
scancel JOBID
```

## Additional Resources
- https://bkamins.github.io/julialang/2020/05/18/project-workflow.html
- [Benchmarking and Scaling Tutorial] (https://hpc-wiki.info/hpc/Benchmarking_%26_Scaling_Tutorial)
- [Julia Distributed Computing Documentation](https://docs.julialang.org/en/v1/stdlib/Distributed/)
- [SLURM Job Scheduler Guide](https://slurm.schedmd.com/quickstart.html)
- [DAS-5 Cluster Documentation](https://www.cs.vu.nl/das5/)
---
