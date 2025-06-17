#!/bin/bash

# export CUDA_HOME=/path/to/cuda
# disable all features except IBGDA
export NVSHMEM_IBGDA_SUPPORT=1

export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

cmake -G Ninja -S . -B build -DCMAKE_INSTALL_PREFIX=/ssd/yanghy/.local/nvshmem_src_3.2.5-1_deepep/build/install
# cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/ssd/yanghy/.local/nvshmem_src_3.2.5-1_deepep/build/install # Use default builder
cmake --build build/ --target install
