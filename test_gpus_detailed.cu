#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned error %d: %s\n", error, cudaGetErrorString(error));
        return 1;
    }
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaSetDevice(i);
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  UUID: ");
        for (int j = 0; j < 16; j++) {
            printf("%02x", prop.uuid.bytes[j]);
        }
        printf("\n");
        printf("  PCI Bus ID: %d\n", prop.pciBusID);
        printf("  PCI Device ID: %d\n", prop.pciDeviceID);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory: %lu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("  Multi-Processor Count: %d\n", prop.multiProcessorCount);
    }
    
    return 0;
}
