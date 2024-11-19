import ray
import ctypes
from pathlib import Path
import os

# Load the CUDA library
def load_cuda_lib():
    # Get absolute paths
    project_dir = os.path.abspath("/home/ray/solana-vanity-gpu")
    lib_path = Path(project_dir) / "src" / "release" / "libcuda-ed25519-vanity.so"
    
    print(f"Looking for library at: {lib_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    if not lib_path.exists():
        raise RuntimeError(f"CUDA library not found at {lib_path}")
    
    lib = ctypes.CDLL(str(lib_path))
    lib.init_vanity.argtypes = [ctypes.c_int]
    lib.init_vanity.restype = None
    return lib

@ray.remote(num_gpus=1)
class VanityGenerator:
    def __init__(self):
        self.lib = load_cuda_lib()
        self.gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        print(f"Initialized VanityGenerator on GPU {self.gpu_id}")
        
    def generate(self):
        print(f"Starting generation on GPU {self.gpu_id}")
        self.lib.init_vanity(self.gpu_id)

def main():
    ray.init()
    
    # Get available GPUs
    gpu_count = int(ray.available_resources().get('GPU', 0))
    if gpu_count == 0:
        raise RuntimeError("No GPUs available through Ray")
    
    print(f"Found {gpu_count} GPUs")
    
    # Create generators for each GPU
    generators = [VanityGenerator.remote() for _ in range(gpu_count)]
    
    # Run generators in parallel
    ray.get([g.generate.remote() for g in generators])

if __name__ == "__main__":
    main()