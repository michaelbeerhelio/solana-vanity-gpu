#ifndef GPU_CTX_H
#define GPU_CTX_H

#include <cuda_runtime.h>
#include <pthread.h>

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8

typedef struct {
    pthread_mutex_t mutex;
    cudaStream_t stream;
} gpu_ctx_t;

extern int32_t g_total_gpus;
bool cuda_crypt_init();
bool ed25519_init();
gpu_ctx_t* get_gpu_ctx();

#endif
