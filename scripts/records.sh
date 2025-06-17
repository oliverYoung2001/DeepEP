
./scripts/wrapper.sh ./scripts/configs/test_internode.sh
NVSHMEM_HCA_LIST=^mlx5_2 python tests/test_intranode.py 2>&1 | tee ./logs/test_intranode.log
NVSHMEM_HCA_LIST=^mlx5_2 python tests/test_low_latency.py 2>&1 | tee ./logs/test_low_latency.log
