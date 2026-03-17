#!/bin/bash
#SBATCH --job-name=compile_grid
#SBATCH --clusters=astro_thin
#SBATCH --partition=def
#SBATCH --qos=astro_thin_def_long
#SBATCH --account=cwilkinson
#SBATCH --nodes=1
#SBATCH --ntasks=1           # Reduced to 1 because the Python code is single-threaded
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00      # 24 hours for sequential processing of many files
#SBATCH --output=compile_%j.out
#SBATCH --error=compile_%j.err

echo "Starting compilation at $(date)"

# Run the Python script
python run_grid.py

echo "Compilation finished at $(date)"