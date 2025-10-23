#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=17
#SBATCH --output=jobs/render_%j.out

# Sequential -> P=0
# Parallel   -> P=[1, 2, 4, 8, 16, 32]
P=0

# Sequential (P = 0)
if [ "$P" -eq 0 ]; then
    julia --project=. scripts/render.jl   

# Parallel (P = 1) - Using a single worker process (On the same node as master)  
elif [[ "$P" -eq 1 ]]; then 
    julia -p 1 --project=. scripts/render.jl  

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

