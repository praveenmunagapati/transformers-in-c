#ifndef TRANSFORMER_TRAINING_H
#define TRANSFORMER_TRAINING_H

#include "transformer_model.h"
#include "matrix_ops.h"
#include "config.h"
#include "utils.h"

// Loss functions
float cross_entropy_loss(const Matrix* predictions, const int* targets, int batch_size, int seq_len);
void cross_entropy_gradient(Matrix* grad_output, const Matrix* predictions, 
                          const int* targets, int batch_size, int seq_len);

// Optimizer
typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int step;
    Matrix** m;  // First moment
    Matrix** v;  // Second moment
    int num_params;
} AdamOptimizer;

AdamOptimizer* adam_create(Transformer* model);
void adam_free(AdamOptimizer* optimizer);
void adam_step(AdamOptimizer* optimizer, Transformer* model);

// Training utilities
void create_padding_mask(Matrix* mask, const int* lengths, int batch_size, int max_len);
void create_causal_mask(Matrix* mask, int seq_len);

// Training loop
typedef struct {
    float* losses;
    int num_steps;
    int current_step;
} TrainingStats;

TrainingStats* training_stats_create(int num_steps);
void training_stats_free(TrainingStats* stats);
void training_stats_update(TrainingStats* stats, float loss);

void train_epoch(Transformer* model, AdamOptimizer* optimizer,
                Batch** batches, int num_batches,
                TrainingStats* stats);

float validate_epoch(Transformer* model, Dataset* dataset,
                   int batch_size, int max_len);

// Learning rate scheduler
float get_learning_rate(int step, int warmup_steps, float d_model);

// Gradient clipping
void clip_gradients(Transformer* model, float max_norm);

// Model checkpointing
void save_checkpoint(Transformer* model, const char* path);
Transformer* load_checkpoint(const char* path);

#endif // TRANSFORMER_TRAINING_H 