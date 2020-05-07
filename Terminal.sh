#!/bin/bash

#SBATCH --partition=allgpu              #allgpu allows the most parallel searches, thanks to the Checkpoint system superseding is no problem
#SBATCH --time=24:00:00                 #maximum time, lower if the network has a tendency to die after a certain time 
#SBATCH --nodes=1                       #Amount of nodes given to the Job. Only increase if Network is VERY computation heavy
#SBATCH --job-name HP-Core-Runner         # give job unique name
#SBATCH --output ./Training/training-%j.out      # terminal output
# --error ./Training/training-%j.err  #Saves a file with the errors, interesting for debugging
# --mail-type END #sends a mail after finishing the job, currently off
# --mail-user mail@url.de #i do not reccomend turning this on
#SBATCH --constraint=GPU  #Only takes GPU machines
#SBATCH --array = 0-100%10 #Starts the same job multiple times: Starting amount-total amount%Parallel running jobs
                         #allgpu supports maximum of 10 parallel jobs, that should be more than enough
 

 
# source and load modules (GPU drivers, anaconda, .bashrc, etc)
source ~/.bashrc  #should initiallize Anaconda/3
source /etc/profile.d/modules.sh
module load maxwell
module load cuda

conda activate FCN #Name of the environment
 
# go to your folder with your python scripts, change USER and NETWORK to ur name and the folder where all the stuff is
cd /home/USER/NETWORK/
 
# run
python Core.py
