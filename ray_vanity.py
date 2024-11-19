import ray
import ctypes
from pathlib import Path
import os

# Load the CUDA library
def load_cuda_lib():
    # Get paths
    source_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    worker_dir = Path("/home/ray/solana-vanity-gpu")
    lib_dir = worker_dir / "src" / "release"
    lib_path = lib_dir / "libcuda-ed25519-vanity.so"
    
    # Create directories if they don't exist
    lib_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy library from source if it exists
    source_lib = source_dir / "src" / "release" / "libcuda-ed25519-vanity.so"
    if source_lib.exists():
        import shutil
        shutil.copy2(source_lib, lib_path)
    
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