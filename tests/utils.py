import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional
import socket
import math
from enum import Enum
import subprocess

class DebugLevel(Enum):
    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    TRACE = 4

    @staticmethod
    def parse(level_str: str):
        level_str = (level_str or "").strip().upper()
        return {
            "OFF": DebugLevel.OFF,
            "ERROR": DebugLevel.ERROR,
            "WARN": DebugLevel.WARN,
            "INFO": DebugLevel.INFO,
            "TRACE": DebugLevel.TRACE,
        }.get(level_str, DebugLevel.OFF)  # 默认 OFF
        
def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    # if mpu.get_data_parallel_rank() == 0:
    if torch.distributed.get_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


BYTE_MULTPLE_UP = 1024
BYTE_MULTPLE_DOWN = 1000
# Helper function to pretty-print message sizes
def convert_throughput(size_bytes, round_=3):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, BYTE_MULTPLE_DOWN)))
    p = math.pow(BYTE_MULTPLE_DOWN, i)
    s = round(size_bytes / p, round_)
    return "%s %s" % (s, size_name[i])

# Helper function to pretty-print message sizes
def convert_size(size_bytes, infix=' ', suffix='B'):
    if size_bytes == 0:
        return "0" + suffix
    # size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, BYTE_MULTPLE_UP)))
    p = math.pow(BYTE_MULTPLE_UP, i)
    # s = int(round(size_bytes / p, 2))
    # s = round(size_bytes / p, 3)
    s = size_bytes / p
    return f"{s:.3g}%s%s%s" % (infix, size_name[i], suffix)

@torch.compile
def force_load_balance_router(NUM_TOKENS, NUM_EXPERTS, TopK, num_groups, group_topk, dtype=torch.int64):
    NUM_EXPERTS_PER_GROUP = NUM_EXPERTS // num_groups
    balance_mod_size = (group_topk // math.gcd(num_groups, group_topk)) * NUM_EXPERTS
    indices = torch.arange(NUM_TOKENS * TopK, dtype=dtype, device=torch.cuda.current_device()) % balance_mod_size
    # Do Permutation: (LCM(NUM_GROUP, GROUP_TOPK) // GROUP_TOPK, NE_per_group, GROUP_TOPK) -> (... * NUM_GROUP, NE_per_group)
    # (idx, idy, idz) -> (idx, idz, idy)
    idz = indices % group_topk
    idy = indices // group_topk % NUM_EXPERTS_PER_GROUP
    idx = indices // (group_topk * NUM_EXPERTS_PER_GROUP)
    indices = ((((idx * group_topk + idz) * NUM_EXPERTS_PER_GROUP + idy)) % NUM_EXPERTS).view(-1, TopK)
    return indices  # [NUM_TOKENS, TopK]; [0, NE)

def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))

def get_master_addr_from_slurm():
    # 获取 SLURM_JOB_ID（如果未设置，则返回 None）
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None

    # 调用 scontrol 获取作业信息并解析第一个节点
    cmd = "scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}'"
    try:
        master_addr = subprocess.run(
            cmd, shell=True, check=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout.strip()
        return master_addr
        # # 提取节点名（如 "BatchHost=node1" -> "node1"）
        # batch_host_line = result.stdout.strip()
        # if "=" in batch_host_line:
        #     master_addr = batch_host_line.split("=")[1].split()[0]
        #     return master_addr
    except subprocess.CalledProcessError as e:
        print(f"Failed to query Slurm: {e.stderr}")
    return None

def parse_slurm_tasks_per_node(tasks_per_node):
    # 4(x2), 8, ...
    return int(tasks_per_node.split('(')[0])

def init_dist_inter(local_rank: int = None, num_local_ranks: int = None):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', None)
    port = int(os.getenv('MASTER_PORT', '8361'))
    if os.getenv('SLURM_PROCID', None) is not None:    # Launch with Slurm
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        num_local_ranks = parse_slurm_tasks_per_node(os.environ['SLURM_TASKS_PER_NODE'])
        node_rank = int(os.environ['SLURM_NODEID'])
        num_nodes = world_size // num_local_ranks
        # print(f'[RANK{rank}] local_rank: {local_rank}, world_size: {world_size}, num_local_ranks: {num_local_ranks}, node_rank: {node_rank}, num_nodes: {num_nodes}', flush=True)
        # ip = os.environ['SLURM_STEP_NODELIST']
        # hostname = socket.gethostname()
        # hostip = socket.gethostbyname(hostname)
        # clustername = os.environ['SLURM_CLUSTER_NAME']
        # nodename = os.environ['SLURMD_NODENAME']
        if not ip:
            ip = get_master_addr_from_slurm()
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:  # Launch with OpenMPI
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        num_local_ranks = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        node_rank = rank // num_local_ranks
        num_nodes = world_size // num_local_ranks
        # print(f'[RANK{rank}] In MPI args initialization !!!', flush=True)
    else:
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
        node_rank = int(os.getenv('RANK', 0))
    if not ip:
        ip = '127.0.0.1'
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank
    # print(f'[Rank{rank}] ip={ip}, port={port}', flush=True)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=world_size,
        rank=rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes))), num_nodes, local_rank, num_local_ranks

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def bench(fn, num_warmups: int = 20, num_tests: int = 30, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = False):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    fn()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tupled = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_times.append(float(time_str.replace(unit, '')) / scale)
                        break
                break
    return tuple(kernel_times) if is_tupled else kernel_times[0]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int64).sum().item()
