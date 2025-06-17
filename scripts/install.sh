#!/bin/bash

# 0. Clone DeepEP repository
pushd /ssd/yanghy/llm
git clone git@github.com:oliverYoung2001/DeepEP.git
git checkout yhy_dev
popd

# 1. Install NVSHMEM
pushd /ssd/yanghy/Software
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
tar -xvf nvshmem_src_3.2.5-1.txz
mv nvshmem_src /ssd/yanghy/.local/nvshmem_src_3.2.5-1_deepep
cd ../.local/nvshmem_src_3.2.5-1_deepep
git apply /ssd/yanghy/llm/DeepEP/third-party/nvshmem.patch
popd

# 2. Configure NVIDIA driver (required by inter-node communication)
#   Enable IBGDA by modifying /etc/modprobe.d/nvidia.conf:
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
#   Update kernel configuration:
sudo update-initramfs -u
sudo reboot
# Done on g[0021,0027]

# 3. Build&Install NVSHMEM
pushd /ssd/yanghy/.local/nvshmem_src_3.2.5-1_deepep
. /ssd/yanghy/llm/DeepEP/scripts/build_nvshmem.sh
popd

#   Post-installation configuration
# Set environment variables in your shell configuration:
export NVSHMEM_DIR=/ssd/yanghy/.local/nvshmem_src_3.2.5-1_deepep/build/install  # Use for DeepEP installation
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

# 4. Install DeepEP
pushd /ssd/yanghy/llm/DeepEP
MAX_JOBS=128 TORCH_CUDA_ARCH_LIST="9.0" DISABLE_SM90_FEATURES=0 python setup.py build
MAX_JOBS=128 TORCH_CUDA_ARCH_LIST="9.0" DISABLE_SM90_FEATURES=0 python setup.py install
popd
