#ifndef TRANSFORMER_LAYERS_H
#define TRANSFORMER_LAYERS_H

#include "matrix_ops.h"
#include "config.h"

// Layer Normalization structure
typedef struct {
    Matrix* gamma;           // Scale parameters
    Matrix* beta;            // Shift parameters
    Matrix* grad_gamma;      // Gradients for gamma
    Matrix* grad_beta;       // Gradients for beta
    Matrix* normalized;      // Normalized values
    Matrix* mean;           // Mean values
    Matrix* variance;       // Variance values
} LayerNorm;

// Feed-Forward Network structure
typedef struct {
    Matrix* weights1;        // First linear layer weights
    Matrix* weights2;        // Second linear layer weights
    Matrix* bias1;          // First linear layer bias
    Matrix* bias2;          // Second linear layer bias
    Matrix* grad_weights1;   // Gradients for weights1
    Matrix* grad_weights2;   // Gradients for weights2
    Matrix* grad_bias1;     // Gradients for bias1
    Matrix* grad_bias2;     // Gradients for bias2
    Matrix* hidden;         // Hidden layer activations
} FeedForward;

// Function declarations for Layer Normalization
LayerNorm* layer_norm_create(int size);
void layer_norm_free(LayerNorm* layer);
void layer_norm_forward(LayerNorm* layer, const Matrix* input, Matrix* output);
void layer_norm_backward(LayerNorm* layer, const Matrix* grad_output,
                        const Matrix* input, Matrix* grad_input);
void layer_norm_reset_gradients(LayerNorm* layer);
void layer_norm_update_weights(LayerNorm* layer, float learning_rate);

// Function declarations for Feed-Forward Network
FeedForward* feed_forward_create(int d_model, int d_ff);
void feed_forward_free(FeedForward* layer);
void feed_forward_forward(FeedForward* layer, const Matrix* input, Matrix* output);
void feed_forward_backward(FeedForward* layer, const Matrix* grad_output,
                          const Matrix* input, Matrix* grad_input);
void feed_forward_reset_gradients(FeedForward* layer);
void feed_forward_update_weights(FeedForward* layer, float learning_rate);

// Utility functions
void layer_norm_print_stats(const LayerNorm* layer);
void feed_forward_print_stats(const FeedForward* layer);

#endif // TRANSFORMER_LAYERS_H 