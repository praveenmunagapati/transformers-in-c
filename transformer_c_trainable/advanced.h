#ifndef TRANSFORMER_ADVANCED_H
#define TRANSFORMER_ADVANCED_H

#include <stdbool.h>
#include "matrix.h"
#include "error.h"

// Mixed precision types
typedef enum {
    PRECISION_FP32 = 0,
    PRECISION_FP16 = 1,
    PRECISION_BF16 = 2
} PrecisionType;

// Distributed training configuration
typedef struct {
    int world_size;        // Total number of processes
    int rank;             // Current process rank
    int local_rank;       // Local GPU rank
    bool use_cuda;        // Whether to use CUDA
    bool use_nccl;        // Whether to use NCCL for GPU communication
    int num_threads;      // Number of threads per process
} DistributedConfig;

// Mixed precision configuration
typedef struct {
    PrecisionType compute_precision;    // Precision for computation
    PrecisionType storage_precision;    // Precision for storage
    bool dynamic_loss_scaling;          // Whether to use dynamic loss scaling
    float initial_loss_scale;           // Initial loss scale
    int loss_scale_window;              // Window for loss scale updates
} PrecisionConfig;

// Function declarations for mixed precision
bool convert_precision(Matrix* dst, const Matrix* src, PrecisionType dst_precision);
bool convert_precision_inplace(Matrix* matrix, PrecisionType new_precision);
float get_loss_scale(const PrecisionConfig* config);
void update_loss_scale(PrecisionConfig* config, bool overflow);

// Function declarations for distributed training
bool distributed_init(const DistributedConfig* config);
void distributed_cleanup(void);
bool all_reduce(Matrix* matrix, const DistributedConfig* config);
bool broadcast(Matrix* matrix, int root, const DistributedConfig* config);
bool scatter(Matrix* dst, const Matrix* src, int root, const DistributedConfig* config);
bool gather(Matrix* dst, const Matrix* src, int root, const DistributedConfig* config);

// Advanced scheduler types
typedef enum {
    SCHEDULER_CONSTANT = 0,
    SCHEDULER_LINEAR = 1,
    SCHEDULER_COSINE = 2,
    SCHEDULER_WARMUP = 3,
    SCHEDULER_ONE_CYCLE = 4
} SchedulerType;

// Scheduler configuration
typedef struct {
    SchedulerType type;
    float initial_lr;
    float final_lr;
    int total_steps;
    int warmup_steps;
    float cycle_momentum;
    float max_momentum;
    float min_momentum;
} SchedulerConfig;

// Function declarations for schedulers
float get_learning_rate(const SchedulerConfig* config, int step);
void update_momentum(SchedulerConfig* config, int step);

#endif // TRANSFORMER_ADVANCED_H 