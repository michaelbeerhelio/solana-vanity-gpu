import ray
import ctypes
from pathlib import Path
import os

# Load the CUDA library
def load_cuda_lib():
    # Get the runtime context to identify the node
    runtime_env = ray.get_runtime_context()
    node_id = runtime_env.node_id
    print(f"Running on node: {node_id}")
    
    # Use local /tmp directory for each node
    lib_path = Path(f"/tmp/libcuda-ed25519-vanity_{node_id}.so")
    
    # Copy from source if running on head node
    source_lib = Path("/home/ray/solana-vanity-gpu/src/release/libcuda-ed25519-vanity.so")
    if source_lib.exists():
        import shutil
        shutil.copy2(source_lib, lib_path)
        print(f"Copied library to {lib_path}")
    
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