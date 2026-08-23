#!/bin/bash

# X number of nodes with Y number of
# cores in each node.
SBATCH -N 1
S3rBATCH -c 1
SrBATCH --mem=20g

# Partition: cpu | gpu-small | gpu-large
SBATCH -p cpu

# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
SBATCH --qos=long-cpu

# TIME
SBATCH -t 168:00:00

# Source the bash profile (required to use the module command)
 
source /etc/profile
source venv/bin/activate
which python3
which pip3
stdbuf -oL venv/bin/python3 /home2/kgxj22/project/experiment/Dissertation/Code/train.py

# Run your program (replace this with your program)
