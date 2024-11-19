import ray
import ctypes
from pathlib import Path
import os

@ray.remote(
    num_gpus=1,
    runtime_env={
        "working_dir": ".",
        "py_modules": [],
        "excludes": ["**/__pycache__"]
    }
)
class VanityGenerator:
    def __init__(self):
        # Get current working directory from Ray
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}")
        print(f"Directory contents: {os.listdir(cwd)}")
        
        lib_path = Path(cwd) / "src" / "release" / "libcuda-ed25519-vanity.so"
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
    ray.init(address='auto')
    
    gpu_count = int(ray.available_resources().get('GPU', 0))
    if gpu_count == 0:
        raise RuntimeError("No GPUs available through Ray")
    
    print(f"Found {gpu_count} GPUs")
    
    generators = [VanityGenerator.remote() for _ in range(gpu_count)]
    ray.get([g.generate.remote() for g in generators])

if __name__ == "__main__":
    main()