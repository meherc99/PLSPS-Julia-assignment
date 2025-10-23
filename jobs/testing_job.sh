#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=17
#SBATCH --output=jobs/testing_%j.out

PROCS=(0 1 2 4 8 16 32) # 2^0 to 2^5 Workers

# For an overall summary of tests
total_tests=0
total_passed=0

# Helper to run a command, echo its output, parse PASS RATE and update totals
run_and_collect () {
  # Run the command and capture the output of the command
  output="$("$@")"
  echo "$output"

  # Find the PASS RATE line that gets printed
  line="$(echo "$output" | grep -E 'PASS RATE:' | tail -n 1)"

  if [[ -n "$line" ]]; then
    # Extract X (passed) and Y (tests) from "PASS RATE: X/Y (Z%)"
    passed="$(echo "$line" | sed -n 's/.*PASS RATE:[[:space:]]*\([0-9]\+\)\/\([0-9]\+\).*/\1/p')"
    tests="$( echo "$line" | sed -n 's/.*PASS RATE:[[:space:]]*\([0-9]\+\)\/\([0-9]\+\).*/\2/p')"

    # Only update if both numbers were parsed
    if [[ -n "$passed" && -n "$tests" ]]; then
      total_passed=$(( total_passed + passed ))
      total_tests=$(( total_tests + tests ))
    fi
  fi
}

for P in "${PROCS[@]}"; do

    # Sequential (P = 0)
    if [ "$P" -eq 0 ]; then
        run_and_collect julia --project=. scripts/testing.jl

    # Parallel (P = 1) - Using a single worker process (On the same node as master)  
    elif [[ "$P" -eq 1 ]]; then 
        run_and_collect julia -p 1 --project=. scripts/testing.jl 

    # Parallel (P > 1) - Distribute workers across two nodes
    else
        HALFP=$((P / 2))                                    

        # Create machine file to distribute workers in Julia
        HOSTS=($(scontrol show hostnames $SLURM_NODELIST))  
        echo "${HALFP}*${HOSTS[0]}" > testing_machine_file.txt
        echo "${HALFP}*${HOSTS[1]}" >> testing_machine_file.txt

        # Run the Julia script
        run_and_collect julia --project=. --machine-file testing_machine_file.txt scripts/testing.jl

        # Clean up
        rm testing_machine_file.txt
    fi
done

# Print the overall summary
echo "=================================================="
echo
if [[ "$total_tests" -gt 0 ]]; then
  overall_pct=$(echo "scale=1; 100 * $total_passed / $total_tests" | bc)
  # Choose emoji based on percentage
  if (( $(echo "$overall_pct > 70" | bc -l) )); then
    emoji="🥳"
  else
    emoji="😢"
  fi
  echo "OVERALL PASS RATE: $total_passed/$total_tests (${overall_pct}%) $emoji"
else
  echo "OVERALL PASS RATE: 0/0 (0.0%) 😢"
fi
echo
echo "=================================================="