#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=17
#SBATCH --output=jobs/benchmark_%j.out

PROCS=(0 1 2 4 8 16 32) # 2^0 to 2^5 Workers 

for P in "${PROCS[@]}"; do

    # Sequential (P = 0)
    if [ "$P" -eq 0 ]; then
        julia --project=. scripts/benchmark.jl   

    # Parallel (P = 1) - Using a single worker process (On the same node as master)  
    elif [[ "$P" -eq 1 ]]; then 
        julia -p 1 --project=. scripts/benchmark.jl  

    # Parallel (P > 1) - Distribute workers across two nodes
    else
        HALFP=$((P / 2))                                    

        # Create machine file to distribute workers in Julia
        HOSTS=($(scontrol show hostnames $SLURM_NODELIST))  
        echo "${HALFP}*${HOSTS[0]}" > benchmark_machine_file.txt
        echo "${HALFP}*${HOSTS[1]}" >> benchmark_machine_file.txt

        # Run the Julia script
        julia --project=. --machine-file benchmark_machine_file.txt scripts/benchmark.jl

        # Clean up
        rm benchmark_machine_file.txt
    fi
done

