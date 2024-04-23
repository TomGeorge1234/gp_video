#!/bin/bash 
#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=245                  #how many independent script you are hoping to run 
#SBATCH --time=20:00:00                         #compute time 
#SBATCH --mem-per-cpu=6000MB 
#SBATCH --cpus-per-task=1 
#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files 
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 0 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 1 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 2 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 3 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 4 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 5 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 6 --i 0 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 0 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 1 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 2 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 3 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 4 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 5 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 6 --i 1 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 0 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 1 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 2 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 3 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 4 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 5 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 6 --i 2 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 0 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 1 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 2 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 3 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 4 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 5 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 6 --i 3 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 0 --T 6 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 1 --T 6 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 2 --T 6 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 3 --T 6 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 4 --T 6 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 5 --T 6 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 0 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 1 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 2 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 3 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 4 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 5 --i 4 --dt 23042024_2259 &
srun --ntasks=1 --nodes=1 python gp_videos.py --L 6 --T 6 --i 4 --dt 23042024_2259 &
wait