import torch
import logging
import gc
import platform
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_optimal_device():
    """Get the optimal device based on availability"""
    if torch.cuda.is_available():
        # Check available memory on GPU to be safe
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory/1e9
        logger.info(f"GPU memory available: {total_memory_gb:.2f} GB")
        
        # Check if running on high-end GPU like A100
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "a100" in gpu_name or "v100" in gpu_name or "h100" in gpu_name:
            logger.info(f"Detected high-performance GPU: {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Apple Silicon (M1/M2) Macs
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for computation")
    return device

def get_torch_dtype_for_device(device):
    """Get appropriate torch dtype based on device"""
    if device == torch.device("cuda"):
        # For NVIDIA GPUs (A100, etc.), bfloat16 is generally well supported
        return torch.bfloat16
    elif device == torch.device("mps"):
        # For Apple Silicon, float16 is better supported than bfloat16
        return torch.float16
    else:
        # For CPU, use float32 for best precision
        return torch.float32

def free_memory():
    """Free up memory on both CPU and GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # There's no direct equivalent to empty_cache for MPS,
        # but we can try to help the garbage collector
        pass
    gc.collect()

def get_system_info():
    """Get information about the current system"""
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        system_info.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        })
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        system_info["gpu"] = "Apple Silicon (MPS)"
    
    return system_info
