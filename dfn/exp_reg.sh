#!/bin/bash

epoch=100

# Function to estimate remaining time
estimate_time() {
    local elapsed_time=$1
    local completed=$2
    local total=$3
    local remaining=$((total - completed))
    local est_time=$(echo "$elapsed_time * $remaining / $completed" | bc -l)
    
    # Round the output to the nearest integer
    est_time=$(printf "%.0f" "$est_time")
    
    local hours=$((est_time / 3600))
    local minutes=$(( (est_time % 3600) / 60 ))
    local seconds=$((est_time % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

# Total number of iterations
types=("sin")
seeds=100
total_iterations=$(( ${#types[@]} * seeds ))
completed_iterations=0

start_time=$(date +%s)

for type in "${types[@]}"; do
    for seed in $(seq 1 $seeds); do
        iter_start_time=$(date +%s)
        python train_reg.py --epoch $epoch --type $type --seed $seed
        iter_end_time=$(date +%s)

        # Increment completed iterations and calculate elapsed time
        completed_iterations=$((completed_iterations + 1))
        elapsed_time=$((iter_end_time - start_time))

        # Estimate remaining time
        remaining_time=$(estimate_time "$elapsed_time" "$completed_iterations" "$total_iterations")

        # Output progress and time estimate
        echo "Completed $completed_iterations/$total_iterations. Estimated remaining time: $remaining_time"
    done
done
