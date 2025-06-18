EXECUTABLE="python ./tests/test_intranode.py"

EXP_NAME=test_intranode
PARTITION=H800
GPUS_PER_NODE=8
NPROC_PER_NODE=1
NNODES=1
HOST='g0009'
HOST='g0021'
# export MASTER_ADDR=g0021
export MASTER_PORT=12321

CPUS=128
# CPU_PER_TASK=$(( CPUS / NPROC_PER_NODE ))   # [NOTE]: Unnecessary for performance.

# # For Debug
# export CUDA_LAUNCH_BLOCKING=1
# export TRITON_DEBUG=1
# # End

export CUDA_DEVICE_MAX_CONNECTIONS=1    # [NOTE]: Important for cc overlap !!!
