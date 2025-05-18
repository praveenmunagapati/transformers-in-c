#include "matrix_ops.h"
#include "my_math.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Matrix creation and destruction
Matrix* matrix_create(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (!matrix) return NULL;
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (float*)calloc(rows * cols, sizeof(float));
    
    if (!matrix->data) {
        free(matrix);
        return NULL;
    }
    
    return matrix;
}

void matrix_free(Matrix* matrix) {
    if (matrix) {
        free(matrix->data);
        free(matrix);
    }
}

Matrix* matrix_clone(const Matrix* matrix) {
    Matrix* clone = matrix_create(matrix->rows, matrix->cols);
    if (!clone) return NULL;
    
    memcpy(clone->data, matrix->data, matrix->rows * matrix->cols * sizeof(float));
    return clone;
}

// Basic matrix operations
void matrix_fill(Matrix* matrix, float value) {
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = value;
    }
}

void matrix_copy(Matrix* dest, const Matrix* src) {
    if (dest->rows != src->rows || dest->cols != src->cols) return;
    memcpy(dest->data, src->data, src->rows * src->cols * sizeof(float));
}

void matrix_add_inplace(Matrix* dest, const Matrix* src) {
    if (dest->rows != src->rows || dest->cols != src->cols) return;
    for (int i = 0; i < dest->rows * dest->cols; i++) {
        dest->data[i] += src->data[i];
    }
}

void matrix_scale_inplace(Matrix* matrix, float scale) {
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] *= scale;
    }
}

// Matrix multiplication and transformations
void matrix_multiply_transpose(Matrix* result, const Matrix* a, const Matrix* b) {
    if (a->cols != b->cols || result->rows != a->rows || result->cols != b->rows) return;
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->rows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[j * b->cols + k];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
}

void matrix_transpose_inplace(Matrix* matrix) {
    Matrix* temp = matrix_clone(matrix);
    if (!temp) return;
    
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[j * matrix->rows + i] = temp->data[i * matrix->cols + j];
        }
    }
    
    int temp_rows = matrix->rows;
    matrix->rows = matrix->cols;
    matrix->cols = temp_rows;
    
    matrix_free(temp);
}

void matrix_reshape(Matrix* matrix, int new_rows, int new_cols) {
    if (new_rows * new_cols != matrix->rows * matrix->cols) return;
    matrix->rows = new_rows;
    matrix->cols = new_cols;
}

// Attention-specific operations
void matrix_attention_scores(Matrix* scores, const Matrix* queries, 
                           const Matrix* keys, float scale) {
    matrix_multiply_transpose(scores, queries, keys);
    matrix_scale_inplace(scores, scale);
}

void matrix_attention_weights(Matrix* weights, const Matrix* scores) {
    for (int i = 0; i < weights->rows; i++) {
        float* row = weights->data + i * weights->cols;
        softmax(row, scores->data + i * scores->cols, weights->cols);
    }
}

void matrix_attention_output(Matrix* output, const Matrix* weights, 
                           const Matrix* values) {
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < weights->cols; k++) {
                sum += weights->data[i * weights->cols + k] * 
                       values->data[k * values->cols + j];
            }
            output->data[i * output->cols + j] = sum;
        }
    }
}

// Gradient computations
void matrix_gradient_multiply(Matrix* grad_a, Matrix* grad_b, 
                            const Matrix* grad_output,
                            const Matrix* a, const Matrix* b) {
    // Gradient for matrix multiplication: dL/dA = dL/dC * B^T, dL/dB = A^T * dL/dC
    Matrix* b_transposed = matrix_create(b->cols, b->rows);
    matrix_transpose_inplace(b_transposed);
    
    matrix_multiply(grad_a->data, grad_output->data, b_transposed->data,
                   grad_output->rows, grad_output->cols, b_transposed->cols);
    
    Matrix* a_transposed = matrix_create(a->cols, a->rows);
    matrix_transpose_inplace(a_transposed);
    
    matrix_multiply(grad_b->data, a_transposed->data, grad_output->data,
                   a_transposed->rows, a_transposed->cols, grad_output->cols);
    
    matrix_free(b_transposed);
    matrix_free(a_transposed);
}

void matrix_gradient_add(Matrix* grad_a, Matrix* grad_b, 
                        const Matrix* grad_output) {
    matrix_copy(grad_a, grad_output);
    matrix_copy(grad_b, grad_output);
}

// Utility functions
void matrix_print(const Matrix* matrix, const char* name) {
    printf("%s (%dx%d):\n", name, matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%.4f ", matrix->data[i * matrix->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

float matrix_max(const Matrix* matrix) {
    float max_val = matrix->data[0];
    for (int i = 1; i < matrix->rows * matrix->cols; i++) {
        if (matrix->data[i] > max_val) {
            max_val = matrix->data[i];
        }
    }
    return max_val;
}

float matrix_sum(const Matrix* matrix) {
    float sum = 0.0f;
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        sum += matrix->data[i];
    }
    return sum;
}

void matrix_apply_mask(Matrix* matrix, const Matrix* mask) {
    if (matrix->rows != mask->rows || matrix->cols != mask->cols) return;
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] *= mask->data[i];
    }
} 