#ifndef GPU_CTX_H
#define GPU_CTX_H

#include <cuda_runtime.h>
#include <pthread.h>
#include "ed25519.h"

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8

typedef struct {
    uint8_t* packets;
    uint32_t* message_lens;
    uint32_t* public_key_offsets;
    uint32_t* private_key_offsets;
    uint32_t* signature_offsets;
    uint32_t* message_start_offsets;
    uint32_t* out;
    size_t packets_size_bytes;
    size_t out_size_bytes;
    size_t offsets_len;
} verify_ctx_t;

typedef struct {
    pthread_mutex_t mutex;
    cudaStream_t stream;
    verify_ctx_t verify_ctx;
} gpu_ctx_t;

extern int32_t g_total_gpus;
bool cuda_crypt_init();
bool ed25519_init();
gpu_ctx_t* get_gpu_ctx();
void setup_gpu_ctx(verify_ctx_t* cur_ctx,
                   const gpu_Elems* elems,
                   uint32_t num_elems,
                   uint32_t message_size,
                   uint32_t total_packets,
                   uint32_t total_packets_size,
                   uint32_t total_signatures,
                   const uint32_t* message_lens,
                   const uint32_t* public_key_offsets,
                   const uint32_t* signature_offsets,
                   const uint32_t* message_start_offsets,
                   size_t out_size,
                   cudaStream_t stream);
void release_gpu_ctx(gpu_ctx_t* cur_ctx);
void ed25519_free_gpu_mem();

#endif
