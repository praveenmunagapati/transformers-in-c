#include "attention.h"
#include "my_math.h"
#include <stdlib.h>
#include <math.h>

// Create a new attention layer
AttentionLayer* attention_create(int d_model, int num_heads) {
    AttentionLayer* layer = (AttentionLayer*)malloc(sizeof(AttentionLayer));
    if (!layer) return NULL;
    
    int head_dim = d_model / num_heads;
    
    // Initialize weight matrices
    layer->query_weights = matrix_create(d_model, d_model);
    layer->key_weights = matrix_create(d_model, d_model);
    layer->value_weights = matrix_create(d_model, d_model);
    layer->output_weights = matrix_create(d_model, d_model);
    
    // Initialize gradient matrices
    layer->grad_query_weights = matrix_create(d_model, d_model);
    layer->grad_key_weights = matrix_create(d_model, d_model);
    layer->grad_value_weights = matrix_create(d_model, d_model);
    layer->grad_output_weights = matrix_create(d_model, d_model);
    
    // Initialize intermediate matrices
    layer->query_proj = matrix_create(d_model, d_model);
    layer->key_proj = matrix_create(d_model, d_model);
    layer->value_proj = matrix_create(d_model, d_model);
    layer->attention_scores = matrix_create(d_model, d_model);
    layer->attention_weights = matrix_create(d_model, d_model);
    layer->attention_output = matrix_create(d_model, d_model);
    
    // Initialize weights with Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (d_model + d_model));
    initialize_weights(layer->query_weights->data, d_model * d_model, scale);
    initialize_weights(layer->key_weights->data, d_model * d_model, scale);
    initialize_weights(layer->value_weights->data, d_model * d_model, scale);
    initialize_weights(layer->output_weights->data, d_model * d_model, scale);
    
    layer->scale_factor = 1.0f / sqrtf((float)head_dim);
    
    return layer;
}

// Free attention layer resources
void attention_free(AttentionLayer* layer) {
    if (!layer) return;
    
    matrix_free(layer->query_weights);
    matrix_free(layer->key_weights);
    matrix_free(layer->value_weights);
    matrix_free(layer->output_weights);
    
    matrix_free(layer->grad_query_weights);
    matrix_free(layer->grad_key_weights);
    matrix_free(layer->grad_value_weights);
    matrix_free(layer->grad_output_weights);
    
    matrix_free(layer->query_proj);
    matrix_free(layer->key_proj);
    matrix_free(layer->value_proj);
    matrix_free(layer->attention_scores);
    matrix_free(layer->attention_weights);
    matrix_free(layer->attention_output);
    
    free(layer);
}

// Forward pass of the attention mechanism
void attention_forward(AttentionLayer* layer, const Matrix* input, 
                      const Matrix* mask, Matrix* output) {
    // Project input to queries, keys, and values
    matrix_multiply(layer->query_proj, input, layer->query_weights);
    matrix_multiply(layer->key_proj, input, layer->key_weights);
    matrix_multiply(layer->value_proj, input, layer->value_weights);
    
    // Compute attention scores
    matrix_attention_scores(layer->attention_scores, 
                          layer->query_proj, layer->key_proj, 
                          layer->scale_factor);
    
    // Apply mask if provided
    if (mask) {
        matrix_apply_mask(layer->attention_scores, mask);
    }
    
    // Compute attention weights using softmax
    matrix_attention_weights(layer->attention_weights, layer->attention_scores);
    
    // Compute attention output
    matrix_attention_output(layer->attention_output, 
                          layer->attention_weights, layer->value_proj);
    
    // Project to output
    matrix_multiply(output, layer->attention_output, layer->output_weights);
}

// Backward pass of the attention mechanism
void attention_backward(AttentionLayer* layer, const Matrix* grad_output,
                       const Matrix* input, Matrix* grad_input) {
    // Gradient of output projection
    matrix_gradient_multiply(layer->grad_output_weights, layer->grad_attention_output,
                           grad_output, layer->attention_output, layer->output_weights);
    
    // Gradient of attention output
    Matrix* grad_attention_output = matrix_create(layer->attention_output->rows,
                                                layer->attention_output->cols);
    matrix_multiply(grad_attention_output, grad_output, layer->output_weights);
    
    // Gradient of attention weights and values
    Matrix* grad_weights = matrix_create(layer->attention_weights->rows,
                                       layer->attention_weights->cols);
    Matrix* grad_values = matrix_create(layer->value_proj->rows,
                                      layer->value_proj->cols);
    
    matrix_gradient_multiply(grad_weights, grad_values, grad_attention_output,
                           layer->attention_weights, layer->value_proj);
    
    // Gradient of attention scores
    Matrix* grad_scores = matrix_create(layer->attention_scores->rows,
                                      layer->attention_scores->cols);
    
    // Compute gradient of softmax
    for (int i = 0; i < grad_scores->rows; i++) {
        float* row = grad_scores->data + i * grad_scores->cols;
        float* weights_row = layer->attention_weights->data + i * layer->attention_weights->cols;
        float* grad_weights_row = grad_weights->data + i * grad_weights->cols;
        
        for (int j = 0; j < grad_scores->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < grad_scores->cols; k++) {
                if (j == k) {
                    sum += weights_row[j] * (1.0f - weights_row[j]) * grad_weights_row[k];
                } else {
                    sum -= weights_row[j] * weights_row[k] * grad_weights_row[k];
                }
            }
            row[j] = sum;
        }
    }
    
    // Gradient of queries and keys
    matrix_gradient_multiply(layer->grad_query_weights, layer->grad_key_weights,
                           grad_scores, layer->query_proj, layer->key_proj);
    
    // Gradient of values
    matrix_gradient_multiply(layer->grad_value_weights, NULL,
                           grad_values, layer->value_proj, NULL);
    
    // Gradient of input
    matrix_multiply(grad_input, grad_attention_output, layer->output_weights);
    
    // Clean up temporary matrices
    matrix_free(grad_attention_output);
    matrix_free(grad_weights);
    matrix_free(grad_values);
    matrix_free(grad_scores);
}

// Reset gradients to zero
void attention_reset_gradients(AttentionLayer* layer) {
    matrix_fill(layer->grad_query_weights, 0.0f);
    matrix_fill(layer->grad_key_weights, 0.0f);
    matrix_fill(layer->grad_value_weights, 0.0f);
    matrix_fill(layer->grad_output_weights, 0.0f);
}

// Update weights using gradients
void attention_update_weights(AttentionLayer* layer, float learning_rate) {
    // Update query weights
    for (int i = 0; i < layer->query_weights->rows * layer->query_weights->cols; i++) {
        layer->query_weights->data[i] -= learning_rate * layer->grad_query_weights->data[i];
    }
    
    // Update key weights
    for (int i = 0; i < layer->key_weights->rows * layer->key_weights->cols; i++) {
        layer->key_weights->data[i] -= learning_rate * layer->grad_key_weights->data[i];
    }
    
    // Update value weights
    for (int i = 0; i < layer->value_weights->rows * layer->value_weights->cols; i++) {
        layer->value_weights->data[i] -= learning_rate * layer->grad_value_weights->data[i];
    }
    
    // Update output weights
    for (int i = 0; i < layer->output_weights->rows * layer->output_weights->cols; i++) {
        layer->output_weights->data[i] -= learning_rate * layer->grad_output_weights->data[i];
    }
}

// Print layer statistics
void attention_print_stats(const AttentionLayer* layer) {
    printf("Attention Layer Statistics:\n");
    printf("Query weights norm: %.4f\n", matrix_norm(layer->query_weights->data, 
                                                   layer->query_weights->rows * layer->query_weights->cols));
    printf("Key weights norm: %.4f\n", matrix_norm(layer->key_weights->data,
                                                 layer->key_weights->rows * layer->key_weights->cols));
    printf("Value weights norm: %.4f\n", matrix_norm(layer->value_weights->data,
                                                   layer->value_weights->rows * layer->value_weights->cols));
    printf("Output weights norm: %.4f\n", matrix_norm(layer->output_weights->data,
                                                    layer->output_weights->rows * layer->output_weights->cols));
    printf("\n");
} 