"""Device selection and cooperative stop requests for standalone PPO training."""
import argparse
import json
import math
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
    if requested not in ('cpu', 'cuda', 'mps', 'auto'):
        raise ValueError('Device must be cpu, cuda, mps or auto')
    available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if requested == 'cuda' and not available:
        raise RuntimeError('CUDA requested but unavailable. Install CUDA-enabled PyTorch in this environment and check the NVIDIA driver; or use --device cpu.')
    if requested == 'mps' and not mps_available:
        raise RuntimeError('MPS requested but unavailable. Use an Apple Silicon Mac with MPS-enabled PyTorch.')
    selected = requested if requested != 'auto' else ('cuda' if available else 'mps' if mps_available else 'cpu')
    if selected == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return selected


def positive_learning_rate(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('Learning rate must be positive and finite')
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Learning rate must be positive and finite')
    return number


def configure_learning_rate(model, requested):
    """Override the saved schedule and group rates without resetting Adam state."""
    previous = model.learning_rate if not callable(model.learning_rate) else 'schedule'
    if requested is not None:
        value = positive_learning_rate(requested)
        model.learning_rate = value
        model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group['lr'] = value
    return dict(requested=requested, previous=previous,
                effective=model.learning_rate if not callable(model.learning_rate) else 'schedule',
                optimizer_state_preserved=True)
