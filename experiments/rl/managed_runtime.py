"""Device selection and cooperative stop requests for standalone PPO training."""
import argparse
import json
import os
from pathlib import Path

PROTOCOL = 'ppo-update-stop-file-v1'


def positive_workers(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('Workers must be a positive integer')
    if number < 1:
        raise argparse.ArgumentTypeError('Workers must be a positive integer')
    return number


def stop_request():
    value = os.environ.get('ASTRA_RL_STOP_FILE')
    if not value or not Path(value).exists():
        return None
    request = json.loads(Path(value).read_text(encoding='utf-8'))
    if request.get('schema') != 'astra.rl-stop-request.v1' or request.get('reason') not in ('max_duration', 'user_cancelled'):
        raise ValueError('Invalid cooperative stop request')
    return request['reason']


def select_device(requested):
    import torch
    if requested not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Device must be cpu, cuda or auto')
    available = torch.cuda.is_available()
    if requested == 'cuda' and not available:
        raise RuntimeError('CUDA requested but unavailable. Install CUDA-enabled PyTorch in this environment and check the NVIDIA driver; or use --device cpu.')
    selected = 'cuda' if requested == 'cuda' or requested == 'auto' and available else 'cpu'
    if selected == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return selected
