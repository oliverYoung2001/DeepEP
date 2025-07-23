EXECUTABLE="./scripts/executor.sh python ./tests/test_low_latency.py"

EXP_NAME=test_low_latency
PARTITION=H800
GPUS_PER_NODE=8
NPROC_PER_NODE=8
NNODES=1
HOST='g0009'
# HOST='g0018'
# HOST='g0021'
NNODES=2
HOST='g[0021,0027]'
# HOST='g[0009,0018]'
NNODES=3
HOST='g[0018,0021,0027]'
NNODES=4
HOST='g[0009,0018,0021,0027]'
export MASTER_PORT=12321

CPUS=128
CPU_PER_TASK=$(( CPUS / NPROC_PER_NODE ))   # [NOTE]: Unnecessary for performance.

# # For Debug
# export CUDA_LAUNCH_BLOCKING=1
# export TRITON_DEBUG=1
# # End

export CUDA_DEVICE_MAX_CONNECTIONS=1    # [NOTE]: Important for cc overlap !!!
