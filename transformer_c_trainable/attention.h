#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include "matrix_ops.h"
#include "config.h"

// Attention layer structure
typedef struct {
    Matrix* query_weights;    // Query projection weights
    Matrix* key_weights;      // Key projection weights
    Matrix* value_weights;    // Value projection weights
    Matrix* output_weights;   // Output projection weights
    
    // Gradients
    Matrix* grad_query_weights;
    Matrix* grad_key_weights;
    Matrix* grad_value_weights;
    Matrix* grad_output_weights;
    
    // Intermediate values for backpropagation
    Matrix* query_proj;
    Matrix* key_proj;
    Matrix* value_proj;
    Matrix* attention_scores;
    Matrix* attention_weights;
    Matrix* attention_output;
    
    float scale_factor;       // Scaling factor for attention scores
} AttentionLayer;

// Function declarations
AttentionLayer* attention_create(int d_model, int num_heads);
void attention_free(AttentionLayer* layer);

// Forward pass
void attention_forward(AttentionLayer* layer, const Matrix* input, 
                      const Matrix* mask, Matrix* output);

// Backward pass
void attention_backward(AttentionLayer* layer, const Matrix* grad_output,
                       const Matrix* input, Matrix* grad_input);

// Utility functions
void attention_reset_gradients(AttentionLayer* layer);
void attention_update_weights(AttentionLayer* layer, float learning_rate);
void attention_print_stats(const AttentionLayer* layer);

#endif // TRANSFORMER_ATTENTION_H 