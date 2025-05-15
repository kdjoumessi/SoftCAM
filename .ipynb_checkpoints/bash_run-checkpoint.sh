#!/bin/bash
#SBATCH --ntasks=1                              # Number of tasks (see below)
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --nodes=1                               # Ensure that all cores are on one machine
#SBATCH --time=3-00:00                          # Runtime in D-HH:MM
#SBATCH --partition=a100-galvani                # Which partition will run your job
#SBATCH --gres=gpu:1                            # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                               # Total memory pool for all cores (see also --mem-per-cpu)

#SBATCH --output=logs/hostname_%j.out           # File to which STDOUT will be written
#SBATCH --error=logs/hostname_%j.err            # File to which STDERR will be written

#SBATCH --mail-type=FAIL                         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=djoumessikerol@gmail.com    # Email to which notifications will be sent


# print info about current job. Useful for debugging in case the job fails 
scontrol show job $SLURM_JOB_ID 

echo "begin python script"

python main.py

echo "end python script"


