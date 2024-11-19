import ray
import ctypes
from pathlib import Path
import os

@ray.remote(num_gpus=1)
class VanityGenerator:
    def __init__(self):
        lib_path = Path("src/release/libcuda-ed25519-vanity.so")
        if not lib_path.exists():
            raise RuntimeError(f"CUDA library not found at {lib_path}")
            
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.init_vanity.argtypes = [ctypes.c_int]
        self.lib.init_vanity.restype = None
        
        self.gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        print(f"Initialized VanityGenerator on GPU {self.gpu_id}")
        
    def generate(self):
        print(f"Starting generation on GPU {self.gpu_id}")
        self.lib.init_vanity(0)

def main():
    # Initialize Ray with runtime environment
    ray.init(
        address='auto',
        runtime_env={
            "working_dir": ".",
            "excludes": ["**/cuda_ed25519_vanity"]
        }
    )
    
    # Get available GPUs
    gpu_count = int(ray.available_resources().get('GPU', 0))
    if gpu_count == 0:
        raise RuntimeError("No GPUs available through Ray")
    
    print(f"Found {gpu_count} GPUs")
    
    generators = [VanityGenerator.remote() for _ in range(gpu_count)]
    ray.get([g.generate.remote() for g in generators])

if __name__ == "__main__":
    main()