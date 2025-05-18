#ifndef TRANSFORMER_MATRIX_OPS_H
#define TRANSFORMER_MATRIX_OPS_H

#include "config.h"

// Matrix structure for easier handling
typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

// Matrix creation and destruction
Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* matrix);
Matrix* matrix_clone(const Matrix* matrix);

// Basic matrix operations
void matrix_fill(Matrix* matrix, float value);
void matrix_copy(Matrix* dest, const Matrix* src);
void matrix_add_inplace(Matrix* dest, const Matrix* src);
void matrix_scale_inplace(Matrix* matrix, float scale);

// Matrix multiplication and transformations
void matrix_multiply_transpose(Matrix* result, const Matrix* a, const Matrix* b);
void matrix_transpose_inplace(Matrix* matrix);
void matrix_reshape(Matrix* matrix, int new_rows, int new_cols);

// Attention-specific operations
void matrix_attention_scores(Matrix* scores, const Matrix* queries, 
                           const Matrix* keys, float scale);
void matrix_attention_weights(Matrix* weights, const Matrix* scores);
void matrix_attention_output(Matrix* output, const Matrix* weights, 
                           const Matrix* values);

// Gradient computations
void matrix_gradient_multiply(Matrix* grad_a, Matrix* grad_b, 
                            const Matrix* grad_output,
                            const Matrix* a, const Matrix* b);
void matrix_gradient_add(Matrix* grad_a, Matrix* grad_b, 
                        const Matrix* grad_output);

// Utility functions
void matrix_print(const Matrix* matrix, const char* name);
float matrix_max(const Matrix* matrix);
float matrix_sum(const Matrix* matrix);
void matrix_apply_mask(Matrix* matrix, const Matrix* mask);

#endif // TRANSFORMER_MATRIX_OPS_H 