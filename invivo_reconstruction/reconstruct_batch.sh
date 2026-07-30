#!/bin/bash

# Parameters
#SBATCH --account=drf@a100
#SBATCH --job-name=orc_0210
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
#SBATCH --partition=a100
#SBATCH --qos=normal@a100
#SBATCH --time=24:00:00
#SBATCH --wckey=submitit
#SBATCH --array=0-39
#SBATCH --output=/ccc/scratch/cont003/drf/pancain/results_3T_invivo/recon_0210/logs/%x_%A_%a.out
#SBATCH --error=/ccc/scratch/cont003/drf/pancain/results_3T_invivo/recon_0210/logs/%x_%A_%a.err
#SBATCH -L fs_scratch,fs_work,fs_store



#set -x

cd "$WORK/codes_3T"

module unload cuda/11.8 
module load cuda/12.2 cudnn/8.9.7

source "$WORK/recon_pipeline/bin/activate"


DATA_ROOT="/ccc/scratch/cont003/drf/pancain/data_3T_invivo/rawdata260210"
TRAJ_ROOT="/ccc/scratch/cont003/drf/pancain/data_3T_invivo/traj_240s"
SMAPS_FILE="/ccc/scratch/cont003/drf/pancain/data_3T_invivo/smaps_0210.npy"
OUT_ROOT="/ccc/scratch/cont003/drf/pancain/results_3T_invivo/recon_0210"
LOGDIR="$OUT_ROOT/logs"
mkdir -p "$LOGDIR"

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")

RUNLIST="$WORK/codes_3T/runlist_draft_0210.txt"
CHUNKS_PER_PARAM="${CHUNKS_PER_PARAM:-20}"

mapfile -t RUN_PARAMS < <(grep -v '^\s*#' "$RUNLIST" | grep -v '^\s*$')
NUM_PARAMS="${#RUN_PARAMS[@]}"
TOTAL_TASKS=$((NUM_PARAMS * CHUNKS_PER_PARAM))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for $NUM_PARAMS parameter sets x $CHUNKS_PER_PARAM chunks"
  exit 1
fi

param_idx=$((SLURM_ARRAY_TASK_ID / CHUNKS_PER_PARAM))
chunk_idx=$((SLURM_ARRAY_TASK_ID % CHUNKS_PER_PARAM))
PARAMS="${RUN_PARAMS[$param_idx]}"
echo "RAW PARAMS=[$PARAMS]"
echo "PARAM IDX=$param_idx CHUNK IDX=$chunk_idx"

job_name=$(echo "$PARAMS" | awk '{print $1}')
traj_rel=$(echo "$PARAMS" | awk '{print $2}')
data_rel=$(echo "$PARAMS" | awk '{print $3}')
n_shot_per_frames=$(echo "$PARAMS" | awk '{print $4}')
total_frames=$(echo "$PARAMS" | awk '{print $5}')
reconstructor=$(echo "$PARAMS" | awk '{print $6}')
init_strategy=$(echo "$PARAMS" | awk '{print $7}')
restart_strategy=$(echo "$PARAMS" | awk '{print $8}')
is_time_varying=$(echo "$PARAMS" | awk '{print $9}')

frames_per_task=$(( (total_frames + CHUNKS_PER_PARAM - 1) / CHUNKS_PER_PARAM ))
start_frame=$((chunk_idx * frames_per_task))
end_frame=$((start_frame + frames_per_task))
if [ "$end_frame" -gt "$total_frames" ]; then
  end_frame="$total_frames"
fi
if [ "$start_frame" -ge "$total_frames" ]; then
  echo "Chunk $chunk_idx is empty for total_frames=$total_frames, nothing to do."
  exit 0
fi


data_dir="${DATA_ROOT}/${data_rel}"
traj_dir="${TRAJ_ROOT}/${traj_rel}"
output_dir="${OUT_ROOT}/${job_name}"
mkdir -p "$output_dir"

echo "PARSED job=$job_name traj_dir=$traj_dir data_dir=$data_dir nshot=$n_shot_per_frames total_frames=$total_frames chunk_frames=$frames_per_task start_frame=$start_frame end_frame=$end_frame recon=$reconstructor init=$init_strategy restart=$restart_strategy"

logfile="$LOGDIR/${timestamp}_${job_name}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_p${param_idx}_c${chunk_idx}.out"


exec > >(tee "$logfile") 2>&1

python "$WORK/codes_3T/recon_script_3T.py" \
  --traj_dir="$traj_dir" \
  --data_dir="$data_dir" \
  --smaps_dir="$SMAPS_FILE" \
  --n_shot_per_frames="$n_shot_per_frames" \
  --output_dir="$output_dir" \
  --frames_per_task="$frames_per_task" \
  --task_id="$chunk_idx" \
  --start_frame="$start_frame" \
  --end_frame="$end_frame" \
  --restart_strategy="$restart_strategy" \
  --reconstructor="$reconstructor" \
  --init_strategy="$init_strategy" \
  --job_name="$job_name" \
  --is_time_varying="$is_time_varying" \
  --use_orc=True
