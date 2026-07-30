#!/bin/bash

# Parameters
#SBATCH --account=drf@a100
#SBATCH --job-name=spiral_R1_diffID
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
#SBATCH --partition=a100
#SBATCH --qos=normal@a100
#SBATCH --time=5:00:00
#SBATCH --wckey=submitit
#SBATCH --array=0-3
#SBATCH --output=/ccc/scratch/cont003/drf/pancain/simulate_3T_snake/cache/%x_%A.out
#SBATCH --error=/ccc/scratch/cont003/drf/pancain/simulate_3T_snake/cache/%x_%A.err
#SBATCH -L fs_scratch,fs_work



#set -x

cd $WORK/codes_3T

module unload cuda/11.8 
module load cuda/12.2 cudnn/8.9.71

source $WORK/recon_pipeline/bin/activate

PARAMS=$(tail -n +2 params_r1.txt | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
JOBNAME=$(echo $PARAMS | awk '{print $1}')
ACSZ=$(echo $PARAMS | awk '{print $2}')
ACCELZ=$(echo $PARAMS | awk '{print $3}')
INIT=$(echo $PARAMS | awk '{print $4}')
ID=$(echo $PARAMS | awk '{print $5}')
SNR=$(echo $PARAMS | awk '{print $6}')

python $WORK/snake-fmri/src/snake/toolkit/cli/main.py  \
    --config-name="scenario2-3T_r1" \
    hydra.job.name=$JOBNAME \
    sim_conf.hardware.field=3.0 \
    sim_conf.hardware.n_coils=1 \
    phantom.tissue_file="tissue_3T" \
    engine.snr=$SNR \
    sampler.stack-of-spiral.acsz=$ACSZ \
    sampler.stack-of-spiral.accelz=$ACCELZ \
    sampler.stack-of-spiral.constant=True \
    phantom.sub_id=$ID \




