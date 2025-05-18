#include "advanced.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <nccl.h>
#endif

// Mixed precision implementation
bool convert_precision(Matrix* dst, const Matrix* src, PrecisionType dst_precision) {
    ERROR_CHECK_NULL(dst);
    ERROR_CHECK_NULL(src);
    ERROR_CHECK(dst->rows == src->rows && dst->cols == src->cols);
    
    switch (dst_precision) {
        case PRECISION_FP32:
            // Convert to FP32
            for (int i = 0; i < src->rows * src->cols; i++) {
                dst->data[i] = (float)src->data[i];
            }
            break;
            
        case PRECISION_FP16:
            // Convert to FP16
            for (int i = 0; i < src->rows * src->cols; i++) {
                // Simple FP16 conversion (in practice, use proper FP16 conversion)
                dst->data[i] = (float)((int16_t)(src->data[i] * 65536.0f)) / 65536.0f;
            }
            break;
            
        case PRECISION_BF16:
            // Convert to BF16
            for (int i = 0; i < src->rows * src->cols; i++) {
                // Simple BF16 conversion (in practice, use proper BF16 conversion)
                dst->data[i] = (float)((int32_t)(src->data[i] * 16777216.0f)) / 16777216.0f;
            }
            break;
            
        default:
            error_set(ERROR_PRECISION, "Unsupported precision type", __FILE__, __LINE__, __func__);
            return false;
    }
    
    return true;
}

bool convert_precision_inplace(Matrix* matrix, PrecisionType new_precision) {
    ERROR_CHECK_NULL(matrix);
    
    float* new_data = (float*)malloc(matrix->rows * matrix->cols * sizeof(float));
    if (!new_data) {
        error_set(ERROR_MEMORY, "Failed to allocate memory for precision conversion", 
                 __FILE__, __LINE__, __func__);
        return false;
    }
    
    Matrix temp = {
        .rows = matrix->rows,
        .cols = matrix->cols,
        .data = new_data
    };
    
    if (!convert_precision(&temp, matrix, new_precision)) {
        free(new_data);
        return false;
    }
    
    memcpy(matrix->data, new_data, matrix->rows * matrix->cols * sizeof(float));
    free(new_data);
    return true;
}

// Loss scaling for mixed precision
static float g_current_loss_scale = 1.0f;
static int g_consecutive_overflows = 0;

float get_loss_scale(const PrecisionConfig* config) {
    ERROR_CHECK_NULL(config);
    return g_current_loss_scale;
}

void update_loss_scale(PrecisionConfig* config, bool overflow) {
    ERROR_CHECK_NULL(config);
    
    if (overflow) {
        g_consecutive_overflows++;
        if (g_consecutive_overflows >= config->loss_scale_window) {
            g_current_loss_scale /= 2.0f;
            g_consecutive_overflows = 0;
        }
    } else {
        g_consecutive_overflows = 0;
        if (config->dynamic_loss_scaling) {
            g_current_loss_scale *= 2.0f;
        }
    }
}

// Distributed training implementation
#ifdef USE_CUDA
static ncclComm_t g_nccl_comm = NULL;
static cudaStream_t g_cuda_stream = NULL;
#endif

bool distributed_init(const DistributedConfig* config) {
    ERROR_CHECK_NULL(config);
    
#ifdef USE_CUDA
    if (config->use_cuda) {
        // Initialize CUDA
        cudaError_t cuda_error = cudaSetDevice(config->local_rank);
        if (cuda_error != cudaSuccess) {
            error_set(ERROR_DISTRIBUTED, "Failed to set CUDA device", 
                     __FILE__, __LINE__, __func__);
            return false;
        }
        
        // Create CUDA stream
        cuda_error = cudaStreamCreate(&g_cuda_stream);
        if (cuda_error != cudaSuccess) {
            error_set(ERROR_DISTRIBUTED, "Failed to create CUDA stream", 
                     __FILE__, __LINE__, __func__);
            return false;
        }
        
        if (config->use_nccl) {
            // Initialize NCCL
            ncclResult_t nccl_error = ncclCommInitRank(&g_nccl_comm, config->world_size,
                                                      NULL, config->rank);
            if (nccl_error != ncclSuccess) {
                error_set(ERROR_DISTRIBUTED, "Failed to initialize NCCL", 
                         __FILE__, __LINE__, __func__);
                return false;
            }
        }
    }
#endif
    
    return true;
}

void distributed_cleanup(void) {
#ifdef USE_CUDA
    if (g_cuda_stream) {
        cudaStreamDestroy(g_cuda_stream);
        g_cuda_stream = NULL;
    }
    
    if (g_nccl_comm) {
        ncclCommDestroy(g_nccl_comm);
        g_nccl_comm = NULL;
    }
#endif
}

bool all_reduce(Matrix* matrix, const DistributedConfig* config) {
    ERROR_CHECK_NULL(matrix);
    ERROR_CHECK_NULL(config);
    
#ifdef USE_CUDA
    if (config->use_cuda && config->use_nccl) {
        ncclResult_t nccl_error = ncclAllReduce(matrix->data, matrix->data,
                                               matrix->rows * matrix->cols,
                                               ncclFloat, ncclSum, g_nccl_comm,
                                               g_cuda_stream);
        if (nccl_error != ncclSuccess) {
            error_set(ERROR_DISTRIBUTED, "NCCL all-reduce failed", 
                     __FILE__, __LINE__, __func__);
            return false;
        }
        
        // Scale the result
        float scale = 1.0f / config->world_size;
        for (int i = 0; i < matrix->rows * matrix->cols; i++) {
            matrix->data[i] *= scale;
        }
    } else
#endif
    {
        // CPU implementation (simple all-reduce)
        float* temp = (float*)malloc(matrix->rows * matrix->cols * sizeof(float));
        if (!temp) {
            error_set(ERROR_MEMORY, "Failed to allocate memory for all-reduce", 
                     __FILE__, __LINE__, __func__);
            return false;
        }
        
        memcpy(temp, matrix->data, matrix->rows * matrix->cols * sizeof(float));
        
        // In a real implementation, this would be done through MPI or other communication
        // Here we just simulate it by scaling the local data
        float scale = 1.0f / config->world_size;
        for (int i = 0; i < matrix->rows * matrix->cols; i++) {
            matrix->data[i] = temp[i] * scale;
        }
        
        free(temp);
    }
    
    return true;
}

// Scheduler implementation
float get_learning_rate(const SchedulerConfig* config, int step) {
    ERROR_CHECK_NULL(config);
    ERROR_CHECK_RANGE(step, 0, config->total_steps);
    
    float progress = (float)step / config->total_steps;
    
    switch (config->type) {
        case SCHEDULER_CONSTANT:
            return config->initial_lr;
            
        case SCHEDULER_LINEAR:
            return config->initial_lr + (config->final_lr - config->initial_lr) * progress;
            
        case SCHEDULER_COSINE:
            return config->final_lr + 0.5f * (config->initial_lr - config->final_lr) *
                   (1.0f + cosf(M_PI * progress));
            
        case SCHEDULER_WARMUP:
            if (step < config->warmup_steps) {
                return config->initial_lr * ((float)step / config->warmup_steps);
            }
            return config->initial_lr;
            
        case SCHEDULER_ONE_CYCLE:
            if (step < config->warmup_steps) {
                return config->initial_lr * ((float)step / config->warmup_steps);
            } else {
                float cycle_progress = (float)(step - config->warmup_steps) /
                                     (config->total_steps - config->warmup_steps);
                return config->final_lr + 0.5f * (config->initial_lr - config->final_lr) *
                       (1.0f + cosf(M_PI * cycle_progress));
            }
            
        default:
            error_set(ERROR_SCHEDULER, "Unsupported scheduler type", 
                     __FILE__, __LINE__, __func__);
            return config->initial_lr;
    }
}

void update_momentum(SchedulerConfig* config, int step) {
    ERROR_CHECK_NULL(config);
    ERROR_CHECK_RANGE(step, 0, config->total_steps);
    
    if (!config->cycle_momentum) return;
    
    float progress = (float)step / config->total_steps;
    float momentum;
    
    switch (config->type) {
        case SCHEDULER_ONE_CYCLE:
            if (step < config->warmup_steps) {
                momentum = config->max_momentum - (config->max_momentum - config->min_momentum) *
                         ((float)step / config->warmup_steps);
            } else {
                float cycle_progress = (float)(step - config->warmup_steps) /
                                     (config->total_steps - config->warmup_steps);
                momentum = config->min_momentum + 0.5f * (config->max_momentum - config->min_momentum) *
                         (1.0f + cosf(M_PI * cycle_progress));
            }
            break;
            
        default:
            momentum = config->max_momentum;
            break;
    }
    
    // Update momentum in the optimizer (this would be implemented in the optimizer code)
    // For now, we just log it
    LOG_DEBUG("Updated momentum to %f at step %d", momentum, step);
} 