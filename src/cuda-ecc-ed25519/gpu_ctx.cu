#include "ed25519.h"
#include "gpu_ctx.h"
#include <pthread.h>
#include "gpu_common.h"

static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8

int32_t g_total_gpus = -1;
static gpu_ctx_t g_gpu_ctx[MAX_NUM_GPUS][MAX_QUEUE_SIZE] = {0};
static uint32_t g_cur_gpu = 0;
static uint32_t g_cur_queue[MAX_NUM_GPUS] = {0};

static bool cuda_crypt_init_locked() {
    if (g_total_gpus == -1) {
        // Reset all devices first
        cudaDeviceReset();
        LOG("total_gpus: %d\n", g_total_gpus);
        const char* visible_devices = getenv("CUDA_VISIBLE_DEVICES");
        if (visible_devices) {
            printf("CUDA_VISIBLE_DEVICES: %s\n", visible_devices);
        }
        
        cudaError_t err = cudaGetDeviceCount(&g_total_gpus);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get GPU count: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        printf("Raw device count: %d\n", g_total_gpus);
        
        // Don't limit GPUs yet - initialize all available ones
        for (int gpu = 0; gpu < g_total_gpus && gpu < MAX_NUM_GPUS; gpu++) {
            cudaSetDevice(gpu);
            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, gpu);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to get properties for GPU %d: %s\n", 
                        gpu, cudaGetErrorString(err));
                continue;
            }
            
            printf("Initializing GPU %d: %s (Compute %d.%d)\n", 
                   gpu, prop.name, prop.major, prop.minor);
            
            // Initialize streams and mutexes for this GPU
            for (int queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
                int mutex_err = pthread_mutex_init(&g_gpu_ctx[gpu][queue].mutex, NULL);
                if (mutex_err != 0) {
                    fprintf(stderr, "pthread_mutex_init error %d gpu: %d queue: %d\n",
                            mutex_err, gpu, queue);
                    continue;
                }
                
                err = cudaStreamCreate(&g_gpu_ctx[gpu][queue].stream);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to create stream for GPU %d queue %d: %s\n",
                            gpu, queue, cudaGetErrorString(err));
                    continue;
                }
            }
        }
    }
    return g_total_gpus > 0;
}

bool cuda_crypt_init() {
    pthread_mutex_lock(&g_ctx_mutex);
    bool success = cuda_crypt_init_locked();
    pthread_mutex_unlock(&g_ctx_mutex);
    return success;
}

bool ed25519_init() {
    cudaFree(0);
    pthread_mutex_lock(&g_ctx_mutex);
    bool success = cuda_crypt_init_locked();
    pthread_mutex_unlock(&g_ctx_mutex);
    return success;
}

gpu_ctx_t* get_gpu_ctx() {
    int32_t cur_gpu, cur_queue;

    pthread_mutex_lock(&g_ctx_mutex);
    if (!cuda_crypt_init_locked()) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return NULL;
    }
    cur_gpu = g_cur_gpu;
    g_cur_gpu++;
    g_cur_gpu %= g_total_gpus;
    cur_queue = g_cur_queue[cur_gpu];
    g_cur_queue[cur_gpu]++;
    g_cur_queue[cur_gpu] %= MAX_QUEUE_SIZE;
    pthread_mutex_unlock(&g_ctx_mutex);

    gpu_ctx_t* cur_ctx = &g_gpu_ctx[cur_gpu][cur_queue];
    pthread_mutex_lock(&cur_ctx->mutex);
    cudaSetDevice(cur_gpu);
    return cur_ctx;
}

void setup_gpu_ctx(gpu_ctx_t* gpu_ctx,
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
                   cudaStream_t stream) {
    
    verify_ctx_t* cur_ctx = &gpu_ctx->verify_ctx;
    size_t offsets_size = total_signatures * sizeof(uint32_t);

    if (cur_ctx->packets == NULL || total_packets_size > cur_ctx->packets_size_bytes) {
        CUDA_CHK(cudaFree(cur_ctx->packets));
        CUDA_CHK(cudaMalloc(&cur_ctx->packets, total_packets_size));
        cur_ctx->packets_size_bytes = total_packets_size;
    }

    if (cur_ctx->out == NULL || cur_ctx->out_size_bytes < out_size) {
        CUDA_CHK(cudaFree(cur_ctx->out));
        CUDA_CHK(cudaMalloc(&cur_ctx->out, out_size));
        cur_ctx->out_size_bytes = total_signatures;
    }

    if (cur_ctx->public_key_offsets == NULL || cur_ctx->offsets_len < total_signatures) {
        CUDA_CHK(cudaFree(cur_ctx->public_key_offsets));
        CUDA_CHK(cudaMalloc(&cur_ctx->public_key_offsets, offsets_size));

        CUDA_CHK(cudaFree(cur_ctx->signature_offsets));
        CUDA_CHK(cudaMalloc(&cur_ctx->signature_offsets, offsets_size));

        CUDA_CHK(cudaFree(cur_ctx->message_start_offsets));
        CUDA_CHK(cudaMalloc(&cur_ctx->message_start_offsets, offsets_size));

        CUDA_CHK(cudaFree(cur_ctx->message_lens));
        CUDA_CHK(cudaMalloc(&cur_ctx->message_lens, offsets_size));

        cur_ctx->offsets_len = total_signatures;
    }

    CUDA_CHK(cudaMemcpyAsync(cur_ctx->public_key_offsets, public_key_offsets, offsets_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->signature_offsets, signature_offsets, offsets_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->message_start_offsets, message_start_offsets, offsets_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->message_lens, message_lens, offsets_size, cudaMemcpyHostToDevice, stream));

    size_t cur = 0;
    for (size_t i = 0; i < num_elems; i++) {
        CUDA_CHK(cudaMemcpyAsync(&cur_ctx->packets[cur * message_size], 
                                elems[i].elems, 
                                elems[i].num * message_size, 
                                cudaMemcpyHostToDevice, 
                                stream));
        cur += elems[i].num;
    }
}

void release_gpu_ctx(gpu_ctx_t* cur_ctx) {
    pthread_mutex_unlock(&cur_ctx->mutex);
}

void ed25519_free_gpu_mem() {
    for (size_t gpu = 0; gpu < MAX_NUM_GPUS; gpu++) {
        for (size_t queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
            gpu_ctx_t* cur_ctx = &g_gpu_ctx[gpu][queue];
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.packets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.out));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.message_lens));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.public_key_offsets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.private_key_offsets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.signature_offsets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.message_start_offsets));
            if (cur_ctx->stream != 0) {
                CUDA_CHK(cudaStreamDestroy(cur_ctx->stream));
            }
        }
    }
}
