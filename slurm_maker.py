#Run this script with --quick handle to get a shorter version of the full script >>> python slurm_maker.py --quick 
import subprocess
import numpy as np 
subprocess.run("rm slurm_script.sh", shell=True)
import argparse 
parser = argparse.ArgumentParser()
args = parser.parse_args()

#enter sweep ranges here:
N_I = [0,1,2,3,4]
N_L = [0,1,2,3,4,5,6]
N_T = [0,1,2,3,4,5,6]

# get date and tiem in format DDMMYYYY_HHMM using time library
import time
date_time = time.strftime("%d%m%Y_%H%M")
# make a results directory called results_DDMMYYYY_HHMM using os 
import os
os.mkdir(f"results/{date_time}")

N_SCRIPTS = len(N_I)*len(N_L)*len(N_T)
print(f"{N_SCRIPTS} scripts total")

pre_schpeel = [
"#!/bin/bash \n",
"#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue \n",
"#SBATCH --ntasks=%g                  #how many independent script you are hoping to run \n" %N_SCRIPTS,
"#SBATCH --time=12:00:00                         #compute time \n",
"#SBATCH --mem=20G \n",
"#SBATCH --cpus-per-task=1 \n",
"#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) \n",
"#SBATCH --error=./logs/%j.err                   #where to save output error files \n"
]


with open("slurm_script.sh","a") as new: 
    for line in pre_schpeel:
        new.write(line)

    for n_i in N_I:
        for n_l in N_L:
            for n_t in N_T:
                new.write(f"srun --ntasks=1 --nodes=1 python gp_videos.py --L {n_l} --T {n_t} --i {n_i} --dt {date_time} &")
                new.write("\n")
    new.write("wait") #stops the script from exiting until all the srun commands have finished
