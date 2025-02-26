#!/bin/bash

# Directory containing the .yaml files
yaml_dir="."

# Find all .yaml files in the directory
yaml_files=$(find "$yaml_dir" -name "*.yaml")

# Sequentially create environments
for yaml_file in $yaml_files; do
    # Extract the environment name from the file
    env_name=$(basename "$yaml_file" .yaml)
    
    echo "Creating environment: $env_name from $yaml_file"
    
    # Create the Conda environment
    mamba env create -f "$yaml_file" -n "$env_name"
    
    if [ $? -eq 0 ]; then
        echo "Environment $env_name created successfully."
    else
        echo "Failed to create environment $env_name."
        exit 1
    fi
done
