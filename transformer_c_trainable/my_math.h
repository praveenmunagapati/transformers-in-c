#ifndef TRANSFORMER_MATH_H
#define TRANSFORMER_MATH_H

#include <math.h>
#include "config.h"

// Basic math operations
float sigmoid(float x);
float tanh_derivative(float x);
float relu(float x);
float relu_derivative(float x);
float softplus(float x);
float softplus_derivative(float x);

// Vector operations
void vector_add(float* result, const float* a, const float* b, int size);
void vector_subtract(float* result, const float* a, const float* b, int size);
void vector_multiply(float* result, const float* a, const float* b, int size);
void vector_divide(float* result, const float* a, const float* b, int size);
float vector_dot_product(const float* a, const float* b, int size);
void vector_scale(float* result, const float* a, float scale, int size);
float vector_norm(const float* a, int size);

// Matrix operations
void matrix_multiply(float* result, const float* a, const float* b, 
                    int m, int n, int p);
void matrix_transpose(float* result, const float* a, int m, int n);
void matrix_add(float* result, const float* a, const float* b, int m, int n);
void matrix_scale(float* result, const float* a, float scale, int m, int n);

// Activation functions
void softmax(float* result, const float* input, int size);
void layer_norm(float* result, const float* input, const float* gamma, 
                const float* beta, int size);

// Random number generation
float random_uniform(float min, float max);
void initialize_weights(float* weights, int size, float scale);

#endif // TRANSFORMER_MATH_H 