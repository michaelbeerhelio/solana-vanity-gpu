import ray
ray.init()
gpu_count = int(ray.available_resources()['GPU'])
print(','.join(str(i) for i in range(gpu_count)))
