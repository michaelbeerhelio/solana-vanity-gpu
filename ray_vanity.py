import ray
import ctypes
from pathlib import Path
import os
import tempfile
import shutil

def prepare_package():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    pkg_dir = Path(temp_dir) / "vanity_pkg"
    pkg_dir.mkdir()
    
    # Copy necessary files
    shutil.copy("src/release/libcuda-ed25519-vanity.so", pkg_dir)
    
    # Create package
    pkg_uri = ray.runtime_env.upload_package_to_gcs(
        str(pkg_dir), include_parent_dir=False
    )
    shutil.rmtree(temp_dir)
    return pkg_uri

@ray.remote(num_gpus=1)
class VanityGenerator:
    def __init__(self):
        lib_path = Path("libcuda-ed25519-vanity.so")
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
    # Initialize Ray
    ray.init(address='auto')
    
    # Upload package and get URI
    pkg_uri = prepare_package()
    
    # Create runtime environment
    runtime_env = {"uris": {pkg_uri: None}}
    
    gpu_count = int(ray.available_resources().get('GPU', 0))
    if gpu_count == 0:
        raise RuntimeError("No GPUs available through Ray")
    
    print(f"Found {gpu_count} GPUs")
    
    # Create generators with runtime environment
    VanityGeneratorWithEnv = ray.remote(
        runtime_env=runtime_env
    )(VanityGenerator)
    
    generators = [VanityGeneratorWithEnv.remote() for _ in range(gpu_count)]
    ray.get([g.generate.remote() for g in generators])

if __name__ == "__main__":
    main()