import ray
import torch

ray.init()

@ray.remote(num_gpus=1)
def get_gpu_info():
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

# Test on all available GPUs
futures = [get_gpu_info.remote() for _ in range(4)]  # Try all 4 GPUs
results = ray.get(futures)
print("GPU Information:", results)
