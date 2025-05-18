#include "my_math.h"
#include <stdlib.h>
#include <string.h>

// Basic math operations
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float tanh_derivative(float x) {
    float th = tanhf(x);
    return 1.0f - th * th;
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}

float softplus_derivative(float x) {
    return sigmoid(x);
}

// Vector operations
void vector_add(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_subtract(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

void vector_multiply(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

void vector_divide(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / (b[i] + EPSILON);
    }
}

float vector_dot_product(const float* a, const float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_scale(float* result, const float* a, float scale, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * scale;
    }
}

float vector_norm(const float* a, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * a[i];
    }
    return sqrtf(sum);
}

// Matrix operations
void matrix_multiply(float* result, const float* a, const float* b, 
                    int m, int n, int p) {
    memset(result, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}

void matrix_transpose(float* result, const float* a, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j * m + i] = a[i * n + j];
        }
    }
}

void matrix_add(float* result, const float* a, const float* b, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        result[i] = a[i] + b[i];
    }
}

void matrix_scale(float* result, const float* a, float scale, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        result[i] = a[i] * scale;
    }
}

// Activation functions
void softmax(float* result, const float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        result[i] = expf(input[i] - max_val);
        sum += result[i];
    }

    for (int i = 0; i < size; i++) {
        result[i] /= sum;
    }
}

void layer_norm(float* result, const float* input, const float* gamma, 
                const float* beta, int size) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;

    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    // Normalize and scale
    float std_dev = sqrtf(variance + EPSILON);
    for (int i = 0; i < size; i++) {
        result[i] = gamma[i] * ((input[i] - mean) / std_dev) + beta[i];
    }
}

// Random number generation
float random_uniform(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

void initialize_weights(float* weights, int size, float scale) {
    for (int i = 0; i < size; i++) {
        weights[i] = random_uniform(-scale, scale);
    }
} 