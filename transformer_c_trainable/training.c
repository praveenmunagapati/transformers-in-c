#include "training.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Loss functions
float cross_entropy_loss(const Matrix* predictions, const int* targets, 
                        int batch_size, int seq_len) {
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
    AdamOptimizer* optimizer = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    if (!optimizer) return NULL;

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
    
    for (int i = 0; i < optimizer->num_params; i++) {
        optimizer->m[i] = matrix_create(1, 1);
        optimizer->v[i] = matrix_create(1, 1);
        matrix_fill(optimizer->m[i], 0.0f);
        matrix_fill(optimizer->v[i], 0.0f);
    }

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
}

void adam_step(AdamOptimizer* optimizer, Transformer* model) {
    optimizer->step++;
    float lr = get_learning_rate(optimizer->step, WARMUP_STEPS, D_MODEL);
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

    // Similar updates for other parameters...
    // (In a full implementation, we would update all parameters)
}

// Training utilities
Batch* batch_create(int batch_size, int max_src_len, int max_tgt_len) {
    Batch* batch = (Batch*)malloc(sizeof(Batch));
    if (!batch) return NULL;
    batch->src_ids = (int*)malloc(batch_size * max_src_len * sizeof(int));
    batch->tgt_ids = (int*)malloc(batch_size * max_tgt_len * sizeof(int));
    batch->src_len = max_src_len;
    batch->tgt_len = max_tgt_len;
    return batch;
}

void batch_free(Batch* batch) {
    if (!batch) return;
    free(batch->src_ids);
    free(batch->tgt_ids);
    free(batch);
}

void create_padding_mask(Matrix* mask, const int* lengths, int batch_size, int max_len) {
    matrix_fill(mask, 1.0f);
    for (int b = 0; b < batch_size; b++) {
        for (int t = lengths[b]; t < max_len; t++) {
            mask->data[b * max_len + t] = 0.0f;
        }
    }
}

void create_causal_mask(Matrix* mask, int seq_len) {
    matrix_fill(mask, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j <= i; j++) {
            mask->data[i * seq_len + j] = 1.0f;
        }
    }
}

// Training loop
TrainingStats* training_stats_create(int num_steps) {
    TrainingStats* stats = (TrainingStats*)malloc(sizeof(TrainingStats));
    if (!stats) return NULL;
    stats->losses = (float*)malloc(num_steps * sizeof(float));
    stats->num_steps = num_steps;
    stats->current_step = 0;
    return stats;
}

void training_stats_free(TrainingStats* stats) {
    if (!stats) return;
    free(stats->losses);
    free(stats);
}

void training_stats_update(TrainingStats* stats, float loss) {
    if (stats->current_step < stats->num_steps) {
        stats->losses[stats->current_step++] = loss;
    }
}

float get_learning_rate(int step, int warmup_steps, float d_model) {
    if (step < warmup_steps) {
        return LEARNING_RATE * (step + 1) / warmup_steps;
    }
    return LEARNING_RATE * sqrtf(d_model) / sqrtf(step + 1);
}

void clip_gradients(Transformer* model, float max_norm) {
    float total_norm = 0.0f;
    
    // Calculate total norm
    // Source embedding
    for (int i = 0; i < model->src_embedding->weights->rows * model->src_embedding->weights->cols; i++) {
        total_norm += model->src_embedding->weights->data[i] * model->src_embedding->weights->data[i];
    }
    // Add similar calculations for other parameters...
    
    total_norm = sqrtf(total_norm);
    
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        // Scale gradients
        // Source embedding
        for (int i = 0; i < model->src_embedding->weights->rows * model->src_embedding->weights->cols; i++) {
            model->src_embedding->weights->data[i] *= scale;
        }
        // Scale other parameters similarly...
    }
}

void train_epoch(Transformer* model, AdamOptimizer* optimizer,
                Batch** batches, int num_batches,
                TrainingStats* stats) {
    for (int b = 0; b < num_batches; b++) {
        Batch* batch = batches[b];
        
        // Forward pass
        Matrix* output = matrix_create(batch->tgt_len, VOCAB_SIZE);
        transformer_forward(model, batch->src_ids, batch->src_len,
                          batch->tgt_ids, batch->tgt_len, output);
        
        // Calculate loss
        float loss = cross_entropy_loss(output, batch->tgt_ids, 1, batch->tgt_len);
        training_stats_update(stats, loss);
        
        // Backward pass (gradient calculation)
        Matrix* grad_output = matrix_create(batch->tgt_len, VOCAB_SIZE);
        cross_entropy_gradient(grad_output, output, batch->tgt_ids, 1, batch->tgt_len);
        
        // Clip gradients
        clip_gradients(model, MAX_GRAD_NORM);
        
        // Update weights
        adam_step(optimizer, model);
        
        // Cleanup
        matrix_free(output);
        matrix_free(grad_output);
    }
}

// Model checkpointing
void save_checkpoint(Transformer* model, const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) return;
    
    // Save model parameters
    // Source embedding
    fwrite(model->src_embedding->weights->data, 
           sizeof(float), 
           model->src_embedding->weights->rows * model->src_embedding->weights->cols,
           file);
    
    // Save other parameters similarly...
    
    fclose(file);
}

Transformer* load_checkpoint(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;
    
    Transformer* model = transformer_create();
    
    // Load model parameters
    // Source embedding
    fread(model->src_embedding->weights->data,
          sizeof(float),
          model->src_embedding->weights->rows * model->src_embedding->weights->cols,
          file);
    
    // Load other parameters similarly...
    
    fclose(file);
    return model;
} 