#!/bin/bash
#SBATCH --job-name=palm_detection
#SBATCH --nodes=1  # number of nodes to run this job
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu  
#SBATCH --time=0-24:00:00
#SBATCH --output="/deac/csc/paucaGrp/student-research-team/palm_detection/outputs/palm_detect-%j.o"
#SBATCH --error="/deac/csc/paucaGrp/student-research-team/palm_detection/outputs/palm_detect-%j.e"
#SBATCH --mail-user=cuij@wfu.edu  # change to your own email
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --account="paucaGrp"
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=180gb  # memory allocated



# cd "/deac/csc/paucaGrp/dark_mining/"

# conda activate tensorflow-env

cd "/deac/csc/paucaGrp/student-research-team/palm_detection"

# cuda directory
# /usr/local/cuda
python3 palm_detection.py # full_test palm_detection


# #SBATCH --cpus-per-task=1
