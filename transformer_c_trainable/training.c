#include "training.h"
#include "error.h"
#include "advanced.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Loss functions
float cross_entropy_loss(const Matrix* predictions, const int* targets, 
                        int batch_size, int seq_len) {
    ERROR_CHECK_NULL(predictions);
    ERROR_CHECK_NULL(targets);
    ERROR_CHECK_RANGE(batch_size, 1, INT_MAX);
    ERROR_CHECK_RANGE(seq_len, 1, INT_MAX);
    
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int target_idx = targets[b * seq_len + t];
            float pred = predictions->data[b * seq_len * VOCAB_SIZE + t * VOCAB_SIZE + target_idx];
            loss -= logf(pred + EPSILON);
        }
    }
    return loss / (batch_size * seq_len);
}

void cross_entropy_gradient(Matrix* grad_output, const Matrix* predictions,
                          const int* targets, int batch_size, int seq_len) {
    ERROR_CHECK_NULL(grad_output);
    ERROR_CHECK_NULL(predictions);
    ERROR_CHECK_NULL(targets);
    ERROR_CHECK_RANGE(batch_size, 1, INT_MAX);
    ERROR_CHECK_RANGE(seq_len, 1, INT_MAX);
    
    matrix_fill(grad_output, 0.0f);
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int target_idx = targets[b * seq_len + t];
            int idx = b * seq_len * VOCAB_SIZE + t * VOCAB_SIZE + target_idx;
            grad_output->data[idx] = -1.0f / (predictions->data[idx] + EPSILON);
        }
    }
}

// Adam Optimizer
AdamOptimizer* adam_create(Transformer* model) {
    ERROR_CHECK_NULL(model);
    
    AdamOptimizer* optimizer = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    if (!optimizer) {
        error_set(ERROR_MEMORY, "Failed to allocate optimizer", __FILE__, __LINE__, __func__);
        return NULL;
    }

    optimizer->learning_rate = LEARNING_RATE;
    optimizer->beta1 = 0.9f;
    optimizer->beta2 = 0.999f;
    optimizer->epsilon = 1e-8f;
    optimizer->step = 0;
    optimizer->num_params = 0;

    // Count parameters
    // Source embedding
    optimizer->num_params += model->src_embedding->weights->rows * model->src_embedding->weights->cols;
    // Target embedding
    optimizer->num_params += model->tgt_embedding->weights->rows * model->tgt_embedding->weights->cols;
    // Output projection
    optimizer->num_params += model->output_proj->rows * model->output_proj->cols;
    // Encoder layers
    for (int i = 0; i < model->encoder->num_layers; i++) {
        EncoderLayer* layer = model->encoder->layers[i];
        optimizer->num_params += layer->self_attn->query_weights->rows * layer->self_attn->query_weights->cols;
        optimizer->num_params += layer->self_attn->key_weights->rows * layer->self_attn->key_weights->cols;
        optimizer->num_params += layer->self_attn->value_weights->rows * layer->self_attn->value_weights->cols;
        optimizer->num_params += layer->self_attn->output_weights->rows * layer->self_attn->output_weights->cols;
        optimizer->num_params += layer->ffn->weights1->rows * layer->ffn->weights1->cols;
        optimizer->num_params += layer->ffn->weights2->rows * layer->ffn->weights2->cols;
    }
    // Decoder layers
    for (int i = 0; i < model->decoder->num_layers; i++) {
        DecoderLayer* layer = model->decoder->layers[i];
        optimizer->num_params += layer->self_attn->query_weights->rows * layer->self_attn->query_weights->cols;
        optimizer->num_params += layer->self_attn->key_weights->rows * layer->self_attn->key_weights->cols;
        optimizer->num_params += layer->self_attn->value_weights->rows * layer->self_attn->value_weights->cols;
        optimizer->num_params += layer->self_attn->output_weights->rows * layer->self_attn->output_weights->cols;
        optimizer->num_params += layer->cross_attn->query_weights->rows * layer->cross_attn->query_weights->cols;
        optimizer->num_params += layer->cross_attn->key_weights->rows * layer->cross_attn->key_weights->cols;
        optimizer->num_params += layer->cross_attn->value_weights->rows * layer->cross_attn->value_weights->cols;
        optimizer->num_params += layer->cross_attn->output_weights->rows * layer->cross_attn->output_weights->cols;
        optimizer->num_params += layer->ffn->weights1->rows * layer->ffn->weights1->cols;
        optimizer->num_params += layer->ffn->weights2->rows * layer->ffn->weights2->cols;
    }

    // Allocate moment matrices
    optimizer->m = (Matrix**)malloc(optimizer->num_params * sizeof(Matrix*));
    optimizer->v = (Matrix**)malloc(optimizer->num_params * sizeof(Matrix*));
    
    if (!optimizer->m || !optimizer->v) {
        error_set(ERROR_MEMORY, "Failed to allocate moment matrices", __FILE__, __LINE__, __func__);
        adam_free(optimizer);
        return NULL;
    }
    
    for (int i = 0; i < optimizer->num_params; i++) {
        optimizer->m[i] = matrix_create(1, 1);
        optimizer->v[i] = matrix_create(1, 1);
        if (!optimizer->m[i] || !optimizer->v[i]) {
            error_set(ERROR_MEMORY, "Failed to allocate moment matrix", __FILE__, __LINE__, __func__);
            adam_free(optimizer);
            return NULL;
        }
        matrix_fill(optimizer->m[i], 0.0f);
        matrix_fill(optimizer->v[i], 0.0f);
    }

    LOG_INFO("Created Adam optimizer with %d parameters", optimizer->num_params);
    return optimizer;
}

void adam_free(AdamOptimizer* optimizer) {
    if (!optimizer) return;
    
    for (int i = 0; i < optimizer->num_params; i++) {
        matrix_free(optimizer->m[i]);
        matrix_free(optimizer->v[i]);
    }
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
    
    LOG_DEBUG("Freed Adam optimizer");
}

void adam_step(AdamOptimizer* optimizer, Transformer* model) {
    ERROR_CHECK_NULL(optimizer);
    ERROR_CHECK_NULL(model);
    
    optimizer->step++;
    
    // Get learning rate from scheduler
    SchedulerConfig scheduler_config = {
        .type = SCHEDULER_WARMUP,
        .initial_lr = LEARNING_RATE,
        .final_lr = LEARNING_RATE * 0.1f,
        .total_steps = 100000,
        .warmup_steps = 4000,
        .cycle_momentum = true,
        .max_momentum = 0.9f,
        .min_momentum = 0.8f
    };
    
    float lr = get_learning_rate(&scheduler_config, optimizer->step);
    float beta1_t = powf(optimizer->beta1, optimizer->step);
    float beta2_t = powf(optimizer->beta2, optimizer->step);

    // Update source embedding weights
    for (int i = 0; i < model->src_embedding->weights->rows * model->src_embedding->weights->cols; i++) {
        float g = model->src_embedding->weights->data[i];
        float m = optimizer->m[0]->data[0] * optimizer->beta1 + g * (1 - optimizer->beta1);
        float v = optimizer->v[0]->data[0] * optimizer->beta2 + g * g * (1 - optimizer->beta2);
        float m_hat = m / (1 - beta1_t);
        float v_hat = v / (1 - beta2_t);
        model->src_embedding->weights->data[i] -= lr * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
        optimizer->m[0]->data[0] = m;
        optimizer->v[0]->data[0] = v;
    }

    // Update momentum
    update_momentum(&scheduler_config, optimizer->step);
    
    LOG_DEBUG("Adam step %d completed with learning rate %f", optimizer->step, lr);
}

// Training utilities
Batch* batch_create(int batch_size, int max_src_len, int max_tgt_len) {
    ERROR_CHECK_RANGE(batch_size, 1, INT_MAX);
    ERROR_CHECK_RANGE(max_src_len, 1, INT_MAX);
    ERROR_CHECK_RANGE(max_tgt_len, 1, INT_MAX);
    
    Batch* batch = (Batch*)malloc(sizeof(Batch));
    if (!batch) {
        error_set(ERROR_MEMORY, "Failed to allocate batch", __FILE__, __LINE__, __func__);
        return NULL;
    }
    
    batch->src_ids = (int*)malloc(batch_size * max_src_len * sizeof(int));
    batch->tgt_ids = (int*)malloc(batch_size * max_tgt_len * sizeof(int));
    
    if (!batch->src_ids || !batch->tgt_ids) {
        error_set(ERROR_MEMORY, "Failed to allocate batch data", __FILE__, __LINE__, __func__);
        batch_free(batch);
        return NULL;
    }
    
    batch->src_len = max_src_len;
    batch->tgt_len = max_tgt_len;
    batch->batch_size = batch_size;
    
    LOG_DEBUG("Created batch with size %d, src_len %d, tgt_len %d", 
              batch_size, max_src_len, max_tgt_len);
    return batch;
}

void batch_free(Batch* batch) {
    if (!batch) return;
    
    free(batch->src_ids);
    free(batch->tgt_ids);
    free(batch);
    
    LOG_DEBUG("Freed batch");
}

void create_padding_mask(Matrix* mask, const int* lengths, int batch_size, int max_len) {
    ERROR_CHECK_NULL(mask);
    ERROR_CHECK_NULL(lengths);
    ERROR_CHECK_RANGE(batch_size, 1, INT_MAX);
    ERROR_CHECK_RANGE(max_len, 1, INT_MAX);
    
    matrix_fill(mask, 1.0f);
    for (int b = 0; b < batch_size; b++) {
        for (int t = lengths[b]; t < max_len; t++) {
            mask->data[b * max_len + t] = 0.0f;
        }
    }
}

void create_causal_mask(Matrix* mask, int seq_len) {
    ERROR_CHECK_NULL(mask);
    ERROR_CHECK_RANGE(seq_len, 1, INT_MAX);
    
    matrix_fill(mask, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j <= i; j++) {
            mask->data[i * seq_len + j] = 1.0f;
        }
    }
}

// Training loop
TrainingStats* training_stats_create(int num_steps) {
    ERROR_CHECK_RANGE(num_steps, 1, INT_MAX);
    
    TrainingStats* stats = (TrainingStats*)malloc(sizeof(TrainingStats));
    if (!stats) {
        error_set(ERROR_MEMORY, "Failed to allocate training stats", __FILE__, __LINE__, __func__);
        return NULL;
    }
    
    stats->losses = (float*)malloc(num_steps * sizeof(float));
    if (!stats->losses) {
        error_set(ERROR_MEMORY, "Failed to allocate losses array", __FILE__, __LINE__, __func__);
        free(stats);
        return NULL;
    }
    
    stats->num_steps = num_steps;
    stats->current_step = 0;
    
    LOG_DEBUG("Created training stats with %d steps", num_steps);
    return stats;
}

void training_stats_free(TrainingStats* stats) {
    if (!stats) return;
    
    free(stats->losses);
    free(stats);
    
    LOG_DEBUG("Freed training stats");
}

void training_stats_update(TrainingStats* stats, float loss) {
    ERROR_CHECK_NULL(stats);
    
    if (stats->current_step < stats->num_steps) {
        stats->losses[stats->current_step++] = loss;
        LOG_DEBUG("Updated training stats: step %d, loss %f", 
                  stats->current_step - 1, loss);
    }
}

void clip_gradients(Transformer* model, float max_norm) {
    ERROR_CHECK_NULL(model);
    ERROR_CHECK_RANGE(max_norm, 0.0f, INFINITY);
    
    float total_norm = 0.0f;
    
    // Calculate total norm
    // Source embedding
    for (int i = 0; i < model->src_embedding->weights->rows * model->src_embedding->weights->cols; i++) {
        total_norm += model->src_embedding->weights->data[i] * model->src_embedding->weights->data[i];
    }
    // Target embedding
    for (int i = 0; i < model->tgt_embedding->weights->rows * model->tgt_embedding->weights->cols; i++) {
        total_norm += model->tgt_embedding->weights->data[i] * model->tgt_embedding->weights->data[i];
    }
    // Output projection
    for (int i = 0; i < model->output_proj->rows * model->output_proj->cols; i++) {
        total_norm += model->output_proj->data[i] * model->output_proj->data[i];
    }
    // Encoder layers
    for (int i = 0; i < model->encoder->num_layers; i++) {
        EncoderLayer* layer = model->encoder->layers[i];
        // Self-attention weights
        for (int j = 0; j < layer->self_attn->query_weights->rows * layer->self_attn->query_weights->cols; j++) {
            total_norm += layer->self_attn->query_weights->data[j] * layer->self_attn->query_weights->data[j];
        }
        for (int j = 0; j < layer->self_attn->key_weights->rows * layer->self_attn->key_weights->cols; j++) {
            total_norm += layer->self_attn->key_weights->data[j] * layer->self_attn->key_weights->data[j];
        }
        for (int j = 0; j < layer->self_attn->value_weights->rows * layer->self_attn->value_weights->cols; j++) {
            total_norm += layer->self_attn->value_weights->data[j] * layer->self_attn->value_weights->data[j];
        }
        for (int j = 0; j < layer->self_attn->output_weights->rows * layer->self_attn->output_weights->cols; j++) {
            total_norm += layer->self_attn->output_weights->data[j] * layer->self_attn->output_weights->data[j];
        }
        // Feed-forward network weights
        for (int j = 0; j < layer->ffn->weights1->rows * layer->ffn->weights1->cols; j++) {
            total_norm += layer->ffn->weights1->data[j] * layer->ffn->weights1->data[j];
        }
        for (int j = 0; j < layer->ffn->weights2->rows * layer->ffn->weights2->cols; j++) {
            total_norm += layer->ffn->weights2->data[j] * layer->ffn->weights2->data[j];
        }
    }
    // Decoder layers
    for (int i = 0; i < model->decoder->num_layers; i++) {
        DecoderLayer* layer = model->decoder->layers[i];
        // Self-attention weights
        for (int j = 0; j < layer->self_attn->query_weights->rows * layer->self_attn->query_weights->cols; j++) {
            total_norm += layer->self_attn->query_weights->data[j] * layer->self_attn->query_weights->data[j];
        }
        for (int j = 0; j < layer->self_attn->key_weights->rows * layer->self_attn->key_weights->cols; j++) {
            total_norm += layer->self_attn->key_weights->data[j] * layer->self_attn->key_weights->data[j];
        }
        for (int j = 0; j < layer->self_attn->value_weights->rows * layer->self_attn->value_weights->cols; j++) {
            total_norm += layer->self_attn->value_weights->data[j] * layer->self_attn->value_weights->data[j];
        }
        for (int j = 0; j < layer->self_attn->output_weights->rows * layer->self_attn->output_weights->cols; j++) {
            total_norm += layer->self_attn->output_weights->data[j] * layer->self_attn->output_weights->data[j];
        }
        // Cross-attention weights
        for (int j = 0; j < layer->cross_attn->query_weights->rows * layer->cross_attn->query_weights->cols; j++) {
            total_norm += layer->cross_attn->query_weights->data[j] * layer->cross_attn->query_weights->data[j];
        }
        for (int j = 0; j < layer->cross_attn->key_weights->rows * layer->cross_attn->key_weights->cols; j++) {
            total_norm += layer->cross_attn->key_weights->data[j] * layer->cross_attn->key_weights->data[j];
        }
        for (int j = 0; j < layer->cross_attn->value_weights->rows * layer->cross_attn->value_weights->cols; j++) {
            total_norm += layer->cross_attn->value_weights->data[j] * layer->cross_attn->value_weights->data[j];
        }
        for (int j = 0; j < layer->cross_attn->output_weights->rows * layer->cross_attn->output_weights->cols; j++) {
            total_norm += layer->cross_attn->output_weights->data[j] * layer->cross_attn->output_weights->data[j];
        }
        // Feed-forward network weights
        for (int j = 0; j < layer->ffn->weights1->rows * layer->ffn->weights1->cols; j++) {
            total_norm += layer->ffn->weights1->data[j] * layer->ffn->weights1->data[j];
        }
        for (int j = 0; j < layer->ffn->weights2->rows * layer->ffn->weights2->cols; j++) {
            total_norm += layer->ffn->weights2->data[j] * layer->ffn->weights2->data[j];
        }
    }
    
    total_norm = sqrtf(total_norm);
    
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        LOG_DEBUG("Gradient clipping: norm %f, scale %f", total_norm, scale);
        
        // Scale gradients
        // Source embedding
        for (int i = 0; i < model->src_embedding->weights->rows * model->src_embedding->weights->cols; i++) {
            model->src_embedding->weights->data[i] *= scale;
        }
        // Target embedding
        for (int i = 0; i < model->tgt_embedding->weights->rows * model->tgt_embedding->weights->cols; i++) {
            model->tgt_embedding->weights->data[i] *= scale;
        }
        // Output projection
        for (int i = 0; i < model->output_proj->rows * model->output_proj->cols; i++) {
            model->output_proj->data[i] *= scale;
        }
        // Encoder layers
        for (int i = 0; i < model->encoder->num_layers; i++) {
            EncoderLayer* layer = model->encoder->layers[i];
            // Self-attention weights
            for (int j = 0; j < layer->self_attn->query_weights->rows * layer->self_attn->query_weights->cols; j++) {
                layer->self_attn->query_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->self_attn->key_weights->rows * layer->self_attn->key_weights->cols; j++) {
                layer->self_attn->key_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->self_attn->value_weights->rows * layer->self_attn->value_weights->cols; j++) {
                layer->self_attn->value_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->self_attn->output_weights->rows * layer->self_attn->output_weights->cols; j++) {
                layer->self_attn->output_weights->data[j] *= scale;
            }
            // Feed-forward network weights
            for (int j = 0; j < layer->ffn->weights1->rows * layer->ffn->weights1->cols; j++) {
                layer->ffn->weights1->data[j] *= scale;
            }
            for (int j = 0; j < layer->ffn->weights2->rows * layer->ffn->weights2->cols; j++) {
                layer->ffn->weights2->data[j] *= scale;
            }
        }
        // Decoder layers
        for (int i = 0; i < model->decoder->num_layers; i++) {
            DecoderLayer* layer = model->decoder->layers[i];
            // Self-attention weights
            for (int j = 0; j < layer->self_attn->query_weights->rows * layer->self_attn->query_weights->cols; j++) {
                layer->self_attn->query_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->self_attn->key_weights->rows * layer->self_attn->key_weights->cols; j++) {
                layer->self_attn->key_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->self_attn->value_weights->rows * layer->self_attn->value_weights->cols; j++) {
                layer->self_attn->value_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->self_attn->output_weights->rows * layer->self_attn->output_weights->cols; j++) {
                layer->self_attn->output_weights->data[j] *= scale;
            }
            // Cross-attention weights
            for (int j = 0; j < layer->cross_attn->query_weights->rows * layer->cross_attn->query_weights->cols; j++) {
                layer->cross_attn->query_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->cross_attn->key_weights->rows * layer->cross_attn->key_weights->cols; j++) {
                layer->cross_attn->key_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->cross_attn->value_weights->rows * layer->cross_attn->value_weights->cols; j++) {
                layer->cross_attn->value_weights->data[j] *= scale;
            }
            for (int j = 0; j < layer->cross_attn->output_weights->rows * layer->cross_attn->output_weights->cols; j++) {
                layer->cross_attn->output_weights->data[j] *= scale;
            }
            // Feed-forward network weights
            for (int j = 0; j < layer->ffn->weights1->rows * layer->ffn->weights1->cols; j++) {
                layer->ffn->weights1->data[j] *= scale;
            }
            for (int j = 0; j < layer->ffn->weights2->rows * layer->ffn->weights2->cols; j++) {
                layer->ffn->weights2->data[j] *= scale;
            }
        }
    }
}

void train_epoch(Transformer* model, AdamOptimizer* optimizer,
                Batch** batches, int num_batches,
                TrainingStats* stats) {
    ERROR_CHECK_NULL(model);
    ERROR_CHECK_NULL(optimizer);
    ERROR_CHECK_NULL(batches);
    ERROR_CHECK_NULL(stats);
    ERROR_CHECK_RANGE(num_batches, 1, INT_MAX);
    
    // Initialize mixed precision if enabled
    PrecisionConfig precision_config = {
        .compute_precision = PRECISION_FP32,
        .storage_precision = PRECISION_FP32,
        .dynamic_loss_scaling = true,
        .initial_loss_scale = 2.0f,
        .loss_scale_window = 2000
    };
    
    // Initialize distributed training if enabled
    DistributedConfig dist_config = {
        .world_size = 1,
        .rank = 0,
        .local_rank = 0,
        .use_cuda = false,
        .use_nccl = false,
        .num_threads = 4
    };
    
    if (!distributed_init(&dist_config)) {
        LOG_ERROR("Failed to initialize distributed training");
        return;
    }
    
    for (int b = 0; b < num_batches; b++) {
        Batch* batch = batches[b];
        LOG_INFO("Processing batch %d/%d", b + 1, num_batches);
        
        // Forward pass
        Matrix* output = matrix_create(batch->tgt_len, VOCAB_SIZE);
        if (!output) {
            LOG_ERROR("Failed to create output matrix");
            continue;
        }
        
        transformer_forward(model, batch->src_ids, batch->src_len,
                          batch->tgt_ids, batch->tgt_len, output);
        
        // Calculate loss
        float loss = cross_entropy_loss(output, batch->tgt_ids, 1, batch->tgt_len);
        training_stats_update(stats, loss);
        
        // Apply loss scaling for mixed precision
        float loss_scale = get_loss_scale(&precision_config);
        loss *= loss_scale;
        
        // Backward pass (gradient calculation)
        Matrix* grad_output = matrix_create(batch->tgt_len, VOCAB_SIZE);
        if (!grad_output) {
            LOG_ERROR("Failed to create gradient matrix");
            matrix_free(output);
            continue;
        }
        
        cross_entropy_gradient(grad_output, output, batch->tgt_ids, 1, batch->tgt_len);
        
        // Scale gradients for mixed precision
        for (int i = 0; i < grad_output->rows * grad_output->cols; i++) {
            grad_output->data[i] *= loss_scale;
        }
        
        // Clip gradients
        clip_gradients(model, MAX_GRAD_NORM);
        
        // All-reduce gradients in distributed training
        if (!all_reduce(grad_output, &dist_config)) {
            LOG_ERROR("Failed to perform all-reduce");
            matrix_free(output);
            matrix_free(grad_output);
            continue;
        }
        
        // Update weights
        adam_step(optimizer, model);
        
        // Update loss scale
        bool overflow = false;  // In practice, check for NaN/Inf
        update_loss_scale(&precision_config, overflow);
        
        // Cleanup
        matrix_free(output);
        matrix_free(grad_output);
        
        LOG_INFO("Completed batch %d/%d with loss %f", b + 1, num_batches, loss);
    }
    
    distributed_cleanup();
}

float validate_epoch(Transformer* model, Dataset* dataset,
                   int batch_size, int max_len) {
    ERROR_CHECK_NULL(model);
    ERROR_CHECK_NULL(dataset);
    ERROR_CHECK_RANGE(batch_size, 1, INT_MAX);
    ERROR_CHECK_RANGE(max_len, 1, INT_MAX);
    
    int num_batches = (dataset->size + batch_size - 1) / batch_size;
    float total_loss = 0.0f;
    int total_samples = 0;
    
    LOG_INFO("Starting validation with %d batches", num_batches);
    
    // Create a single batch for validation
    Batch* batch = batch_create(batch_size, max_len, max_len);
    if (!batch) {
        LOG_ERROR("Failed to create validation batch");
        return 0.0f;
    }
    
    // Create output matrix
    Matrix* output = matrix_create(batch_size, max_len);
    if (!output) {
        LOG_ERROR("Failed to create output matrix");
        batch_free(batch);
        return 0.0f;
    }
    
    // Process each batch
    for (int i = 0; i < num_batches; i++) {
        LOG_DEBUG("Processing validation batch %d/%d", i + 1, num_batches);
        
        generate_batch(batch, dataset, i * batch_size);
        
        // Forward pass
        transformer_forward(model, batch->src_ids, batch->src_len,
                          batch->tgt_ids, batch->tgt_len, output);
        
        // Compute loss
        float batch_loss = cross_entropy_loss(output, batch->tgt_ids,
                                            batch->batch_size, batch->tgt_len);
        
        total_loss += batch_loss * batch->batch_size;
        total_samples += batch->batch_size;
        
        LOG_DEBUG("Validation batch %d/%d loss: %f", i + 1, num_batches, batch_loss);
    }
    
    // Cleanup
    batch_free(batch);
    matrix_free(output);
    
    float avg_loss = total_samples > 0 ? total_loss / total_samples : 0.0f;
    LOG_INFO("Validation completed with average loss: %f", avg_loss);
    return avg_loss;
} 