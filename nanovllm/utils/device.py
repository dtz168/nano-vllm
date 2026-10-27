import os
import torch


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def maybe_compile(fn):
    # torch.compile only pays off on CUDA; its MPS backend is incomplete and CPU warmup is slow
    return torch.compile(fn) if torch.cuda.is_available() else fn


def empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_memory_bytes(device: torch.device, gpu_memory_utilization: float, model_bytes: int) -> int:
    # Non-CUDA backends have no peak-memory stats, so the headroom left by
    # gpu_memory_utilization has to cover activations.
    if device.type == "mps":
        return int(torch.mps.recommended_max_memory() * gpu_memory_utilization) - torch.mps.driver_allocated_memory()
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return int(total * gpu_memory_utilization) - model_bytes
