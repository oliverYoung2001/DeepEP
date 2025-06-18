# # Openmpi (already loaded)
# export OPENMPI_HOME=/home/fit/zhaijd/yhy/.local/openmpi
# export PATH="$OPENMPI_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"

# export C_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$C_INCLUDE_PATH"  # for #include <mpi.h>
# export CPLUS_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$CPLUS_INCLUDE_PATH"  # for #include <mpi.h>
# export LD_LIBRARY_PATH="$(dirname `which nvcc`)/../lib64:$LD_LIBRARY_PATH"  # for -lcudart

USER_ROOT=/ssd/yanghy
# Spack
source $USER_ROOT/.local/spack/share/spack/setup-env.sh
spack load cuda@12.8.1

# conda
source $USER_ROOT/.local/miniconda3/bin/activate
conda deactivate && conda deactivate && conda deactivate
# conda activate deepep
conda activate comm
# source /ssd/tianr/miniconda3/bin/activate
# conda deactivate && conda deactivate && conda deactivate && conda activate sglang-pd

# # set GUROBI licence env
# export GRB_LICENSE_FILE=/home/yhy/mnt/.local/gurobi/gurobi.lic

# [NOTE]: tmux of qy has bug along with conda, thus the solution is copy the $PATH out of tmux into tmux  !!!
export http_proxy=127.0.0.1:18901
export https_proxy=127.0.0.1:18901

# # Triton-distributed
# # export CUDA_DEVICE_MAX_CONNECTIONS=1
# # export CUDA_LAUNCH_BLOCKING=0
# # export TORCH_CPP_LOG_LEVEL=1
# # export NCCL_DEBUG=ERROR

# # SCRIPT_DIR="$(pwd)"
# # SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
# # NVSHMEM_ROOT=${SCRIPT_DIR}/3rdparty/nvshmem/build/install
# # [NOTE]: Use '/home/yhy/mnt/.local/nvshmem_src_3.2.5-1/*' rather than '/opt/nvshmem*' !!!
# export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|/opt/nvshmem/lib:||g' | sed 's|:/opt/nvshmem/lib||g')
# export NVSHMEM_DIR=/home/yhy/mnt/.local/nvshmem_src_3.2.5-1/build/install
# export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
# export PATH="${NVSHMEM_DIR}/bin:$PATH"
