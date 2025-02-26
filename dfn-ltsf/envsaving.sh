#!/bin/bash

# Get a list of all conda environments
envs=$(conda env list | awk '{print $1}' | tail -n +4)

# Export each environment
for env in $envs; do
    conda env export -n "$env" > "${env}.yaml"
    echo "Exported $env to ${env}.yaml"
done