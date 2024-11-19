import ray
import ctypes
from pathlib import Path
import os

# Load the CUDA library
def load_cuda_lib():
    lib_path = Path(__file__).parent / "release" / "libcuda-ed25519-vanity.so"
    if not lib_path.exists():
        raise RuntimeError(f"CUDA library not found at {lib_path}")
    return ctypes.CDLL(str(lib_path))

@ray.remote(num_gpus=1)
class VanityGenerator:
    def __init__(self):
        self.lib = load_cuda_lib()
        self.gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        
    def generate(self):
        # Call the CUDA functions
        self.lib.vanity_setup()
        self.lib.vanity_run()

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