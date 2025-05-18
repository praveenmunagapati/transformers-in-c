#include "layers.h"
#include "my_math.h"
#include <stdlib.h>
#include <string.h>

// Layer Normalization implementation
LayerNorm* layer_norm_create(int size) {
    LayerNorm* layer = (LayerNorm*)malloc(sizeof(LayerNorm));
    if (!layer) return NULL;

    layer->gamma = matrix_create(1, size);
    layer->beta = matrix_create(1, size);
    layer->grad_gamma = matrix_create(1, size);
    layer->grad_beta = matrix_create(1, size);
    layer->normalized = matrix_create(1, size);
    layer->mean = matrix_create(1, 1);
    layer->variance = matrix_create(1, 1);

    if (!layer->gamma || !layer->beta || !layer->grad_gamma || !layer->grad_beta ||
        !layer->normalized || !layer->mean || !layer->variance) {
        layer_norm_free(layer);
        return NULL;
    }

    // Initialize gamma to ones and beta to zeros
    matrix_fill(layer->gamma, 1.0f);
    matrix_fill(layer->beta, 0.0f);
    matrix_fill(layer->grad_gamma, 0.0f);
    matrix_fill(layer->grad_beta, 0.0f);

    return layer;
}

void layer_norm_free(LayerNorm* layer) {
    if (!layer) return;
    matrix_free(layer->gamma);
    matrix_free(layer->beta);
    matrix_free(layer->grad_gamma);
    matrix_free(layer->grad_beta);
    matrix_free(layer->normalized);
    matrix_free(layer->mean);
    matrix_free(layer->variance);
    free(layer);
}

void layer_norm_forward(LayerNorm* layer, const Matrix* input, Matrix* output) {
    int size = input->cols;
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input->data[i];
    }
    mean /= size;
    layer->mean->data[0] = mean;

    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input->data[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    layer->variance->data[0] = variance;

    // Normalize and scale
    float std_dev = sqrtf(variance + EPSILON);
    for (int i = 0; i < size; i++) {
        layer->normalized->data[i] = (input->data[i] - mean) / std_dev;
        output->data[i] = layer->gamma->data[i] * layer->normalized->data[i] + layer->beta->data[i];
    }
}

void layer_norm_backward(LayerNorm* layer, const Matrix* grad_output,
                        const Matrix* input, Matrix* grad_input) {
    int size = input->cols;
    float std_dev = sqrtf(layer->variance->data[0] + EPSILON);
    
    // Calculate gradients for gamma and beta
    for (int i = 0; i < size; i++) {
        layer->grad_gamma->data[i] += grad_output->data[i] * layer->normalized->data[i];
        layer->grad_beta->data[i] += grad_output->data[i];
    }

    // Calculate gradient for input
    float grad_normalized_sum = 0.0f;
    float grad_normalized_dot_normalized = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float grad_normalized = grad_output->data[i] * layer->gamma->data[i];
        grad_normalized_sum += grad_normalized;
        grad_normalized_dot_normalized += grad_normalized * layer->normalized->data[i];
    }

    for (int i = 0; i < size; i++) {
        float grad_normalized = grad_output->data[i] * layer->gamma->data[i];
        grad_input->data[i] = (grad_normalized - grad_normalized_sum / size -
                              layer->normalized->data[i] * grad_normalized_dot_normalized / size) / std_dev;
    }
}

void layer_norm_reset_gradients(LayerNorm* layer) {
    matrix_fill(layer->grad_gamma, 0.0f);
    matrix_fill(layer->grad_beta, 0.0f);
}

void layer_norm_update_weights(LayerNorm* layer, float learning_rate) {
    for (int i = 0; i < layer->gamma->cols; i++) {
        layer->gamma->data[i] -= learning_rate * layer->grad_gamma->data[i];
        layer->beta->data[i] -= learning_rate * layer->grad_beta->data[i];
    }
}

// Feed-Forward Network implementation
FeedForward* feed_forward_create(int d_model, int d_ff) {
    FeedForward* layer = (FeedForward*)malloc(sizeof(FeedForward));
    if (!layer) return NULL;

    layer->weights1 = matrix_create(d_model, d_ff);
    layer->weights2 = matrix_create(d_ff, d_model);
    layer->bias1 = matrix_create(1, d_ff);
    layer->bias2 = matrix_create(1, d_model);
    layer->grad_weights1 = matrix_create(d_model, d_ff);
    layer->grad_weights2 = matrix_create(d_ff, d_model);
    layer->grad_bias1 = matrix_create(1, d_ff);
    layer->grad_bias2 = matrix_create(1, d_model);
    layer->hidden = matrix_create(1, d_ff);

    if (!layer->weights1 || !layer->weights2 || !layer->bias1 || !layer->bias2 ||
        !layer->grad_weights1 || !layer->grad_weights2 || !layer->grad_bias1 || !layer->grad_bias2 ||
        !layer->hidden) {
        feed_forward_free(layer);
        return NULL;
    }

    // Initialize weights using Xavier/Glorot initialization
    float scale1 = sqrtf(2.0f / (d_model + d_ff));
    float scale2 = sqrtf(2.0f / (d_ff + d_model));
    
    for (int i = 0; i < d_model * d_ff; i++) {
        layer->weights1->data[i] = random_normal(0.0f, scale1);
        layer->grad_weights1->data[i] = 0.0f;
    }
    
    for (int i = 0; i < d_ff * d_model; i++) {
        layer->weights2->data[i] = random_normal(0.0f, scale2);
        layer->grad_weights2->data[i] = 0.0f;
    }

    matrix_fill(layer->bias1, 0.0f);
    matrix_fill(layer->bias2, 0.0f);
    matrix_fill(layer->grad_bias1, 0.0f);
    matrix_fill(layer->grad_bias2, 0.0f);

    return layer;
}

void feed_forward_free(FeedForward* layer) {
    if (!layer) return;
    matrix_free(layer->weights1);
    matrix_free(layer->weights2);
    matrix_free(layer->bias1);
    matrix_free(layer->bias2);
    matrix_free(layer->grad_weights1);
    matrix_free(layer->grad_weights2);
    matrix_free(layer->grad_bias1);
    matrix_free(layer->grad_bias2);
    matrix_free(layer->hidden);
    free(layer);
}

void feed_forward_forward(FeedForward* layer, const Matrix* input, Matrix* output) {
    // First linear layer with ReLU activation
    matrix_multiply(layer->hidden->data, input->data, layer->weights1->data, 
                   1, input->cols, layer->weights1->cols);
    for (int i = 0; i < layer->hidden->cols; i++) {
        layer->hidden->data[i] += layer->bias1->data[i];
        layer->hidden->data[i] = relu(layer->hidden->data[i]);
    }

    // Second linear layer
    matrix_multiply(output->data, layer->hidden->data, layer->weights2->data, 
                   1, layer->hidden->cols, layer->weights2->cols);
    for (int i = 0; i < output->cols; i++) {
        output->data[i] += layer->bias2->data[i];
    }
}

void feed_forward_backward(FeedForward* layer, const Matrix* grad_output,
                          const Matrix* input, Matrix* grad_input) {
    // Gradient of second linear layer
    matrix_multiply(layer->grad_weights2->data, layer->hidden->data, grad_output->data, 
                   layer->hidden->cols, 1, grad_output->cols);
    for (int i = 0; i < layer->grad_bias2->cols; i++) {
        layer->grad_bias2->data[i] += grad_output->data[i];
    }

    // Gradient through ReLU
    Matrix* grad_hidden = matrix_create(1, layer->hidden->cols);
    matrix_multiply(grad_hidden->data, grad_output->data, layer->weights2->data, 
                   1, grad_output->cols, layer->weights2->rows);
    for (int i = 0; i < grad_hidden->cols; i++) {
        grad_hidden->data[i] *= relu_derivative(layer->hidden->data[i]);
    }

    // Gradient of first linear layer
    matrix_multiply(layer->grad_weights1->data, input->data, grad_hidden->data, 
                   input->cols, 1, grad_hidden->cols);
    for (int i = 0; i < layer->grad_bias1->cols; i++) {
        layer->grad_bias1->data[i] += grad_hidden->data[i];
    }

    // Gradient for input
    matrix_multiply(grad_input->data, grad_hidden->data, layer->weights1->data, 
                   1, grad_hidden->cols, layer->weights1->rows);

    matrix_free(grad_hidden);
}

void feed_forward_reset_gradients(FeedForward* layer) {
    matrix_fill(layer->grad_weights1, 0.0f);
    matrix_fill(layer->grad_weights2, 0.0f);
    matrix_fill(layer->grad_bias1, 0.0f);
    matrix_fill(layer->grad_bias2, 0.0f);
}

void feed_forward_update_weights(FeedForward* layer, float learning_rate) {
    for (int i = 0; i < layer->weights1->rows * layer->weights1->cols; i++) {
        layer->weights1->data[i] -= learning_rate * layer->grad_weights1->data[i];
    }
    for (int i = 0; i < layer->weights2->rows * layer->weights2->cols; i++) {
        layer->weights2->data[i] -= learning_rate * layer->grad_weights2->data[i];
    }
    for (int i = 0; i < layer->bias1->cols; i++) {
        layer->bias1->data[i] -= learning_rate * layer->grad_bias1->data[i];
    }
    for (int i = 0; i < layer->bias2->cols; i++) {
        layer->bias2->data[i] -= learning_rate * layer->grad_bias2->data[i];
    }
}

// Utility functions
void layer_norm_print_stats(const LayerNorm* layer) {
    printf("Layer Normalization Stats:\n");
    printf("Mean: %f\n", layer->mean->data[0]);
    printf("Variance: %f\n", layer->variance->data[0]);
}

void feed_forward_print_stats(const FeedForward* layer) {
    printf("Feed-Forward Network Stats:\n");
    printf("Hidden layer size: %d\n", layer->hidden->cols);
    printf("Input size: %d\n", layer->weights1->rows);
    printf("Output size: %d\n", layer->weights2->cols);
} 