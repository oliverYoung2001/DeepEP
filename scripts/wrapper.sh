#!/bin/bash

export TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

# Default settings:
PARTITION=H800
NNODES=1
GPUS_PER_NODE=8
NPROC_PER_NODE=8
HOST='g0018'
# End

# Envs:
export CLUSTER_NAME=bingxing
export PLATFORM='H800'
# Specific settings on BingXing
# export NVSHMEM_HCA_LIST=^mlx5_2
# export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_3,mlx5_4
# End

source $1   # May overwrite the default settings
mkdir -p logs/${EXP_NAME}

# MEM_PER_CPU=256G
# # --mem-per-cpu $MEM_PER_CPU \
# MEM_PER_NODE=256G
# --mem $MEM_PER_NODE \

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$NPROC_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
-K \
--cpu-bind=none \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi
if [ ! -z "$CPU_PER_TASK" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        --cpus-per-task=$CPU_PER_TASK \
    "
fi

# # Run with Slurm
RUNNER_CMD="srun $SLURM_ARGS"

# -c 16 \
set -x
$RUNNER_CMD \
$EXECUTABLE 2>&1 | tee logs/${EXP_NAME}/output_${TIMESTAMP}.log

set +x
