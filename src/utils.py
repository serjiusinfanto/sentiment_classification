"""
Utility functions for reproducibility and timing
"""
import torch
import random
import numpy as np
import time


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the device (GPU if available, else CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_hardware_info():
    """
    Get hardware information for reproducibility reporting
    """
    info = {}
    if torch.cuda.is_available():
        info['device'] = 'GPU'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        info['device'] = 'CPU'

    return info


class Timer:
    """
    Timer class for measuring training time
    """
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            self.start_time = None
        return self.elapsed_time

    def get_elapsed(self):
        return self.elapsed_time
