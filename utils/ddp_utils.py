import os
import numpy as np
import random

import torch
import torch.distributed as dist


def setup_ddp(rank, world_size):
    """Initialize distributed process group"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    random_seed = 42 + rank
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def cleanup_ddp():
    """Destroy process group"""
    dist.destroy_process_group()
