#include "transformer.h"
#include "error.h"
#include "advanced.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Test utilities
#define TEST_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "Test failed: %s at %s:%d\n", #expr, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define TEST_RUN(test_func) do { \
    printf("Running test: %s\n", #test_func); \
    if (!test_func()) { \
        printf("Test failed: %s\n", #test_func); \
        return false; \
    } \
    printf("Test passed: %s\n", #test_func); \
} while(0)

// Test functions
static bool test_matrix_operations(void) {
    // Test matrix creation
    Matrix* m1 = matrix_create(2, 3);
    TEST_ASSERT(m1 != NULL);
    TEST_ASSERT(m1->rows == 2);
    TEST_ASSERT(m1->cols == 3);
    
    // Test matrix fill
    matrix_fill(m1, 1.0f);
    for (int i = 0; i < m1->rows * m1->cols; i++) {
        TEST_ASSERT(m1->data[i] == 1.0f);
    }
    
    // Test matrix addition
    Matrix* m2 = matrix_create(2, 3);
    matrix_fill(m2, 2.0f);
    Matrix* m3 = matrix_add(m1, m2);
    TEST_ASSERT(m3 != NULL);
    for (int i = 0; i < m3->rows * m3->cols; i++) {
        TEST_ASSERT(m3->data[i] == 3.0f);
    }
    
    // Cleanup
    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    return true;
}

static bool test_transformer_creation(void) {
    Transformer* model = transformer_create(1000, 512, 8, 2048, 6, 6, 0.1f);
    TEST_ASSERT(model != NULL);
    TEST_ASSERT(model->vocab_size == 1000);
    TEST_ASSERT(model->d_model == 512);
    TEST_ASSERT(model->num_heads == 8);
    TEST_ASSERT(model->d_ff == 2048);
    TEST_ASSERT(model->num_encoder_layers == 6);
    TEST_ASSERT(model->num_decoder_layers == 6);
    TEST_ASSERT(model->dropout == 0.1f);
    
    transformer_free(model);
    return true;
}

static bool test_mixed_precision(void) {
    // Create test matrix
    Matrix* m1 = matrix_create(2, 2);
    matrix_fill(m1, 1.5f);
    
    // Test FP16 conversion
    Matrix* m2 = matrix_create(2, 2);
    TEST_ASSERT(convert_precision(m2, m1, PRECISION_FP16));
    
    // Test BF16 conversion
    Matrix* m3 = matrix_create(2, 2);
    TEST_ASSERT(convert_precision(m3, m1, PRECISION_BF16));
    
    // Test inplace conversion
    TEST_ASSERT(convert_precision_inplace(m1, PRECISION_FP16));
    
    // Cleanup
    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    return true;
}

static bool test_distributed_training(void) {
    DistributedConfig config = {
        .world_size = 2,
        .rank = 0,
        .local_rank = 0,
        .use_cuda = false,
        .use_nccl = false,
        .num_threads = 4
    };
    
    // Test initialization
    TEST_ASSERT(distributed_init(&config));
    
    // Test all-reduce
    Matrix* m = matrix_create(2, 2);
    matrix_fill(m, 1.0f);
    TEST_ASSERT(all_reduce(m, &config));
    
    // Cleanup
    matrix_free(m);
    distributed_cleanup();
    return true;
}

static bool test_scheduler(void) {
    SchedulerConfig config = {
        .type = SCHEDULER_ONE_CYCLE,
        .initial_lr = 0.1f,
        .final_lr = 0.001f,
        .total_steps = 1000,
        .warmup_steps = 100,
        .cycle_momentum = true,
        .max_momentum = 0.9f,
        .min_momentum = 0.8f
    };
    
    // Test learning rate calculation
    float lr = get_learning_rate(&config, 50);
    TEST_ASSERT(lr > 0.0f);
    TEST_ASSERT(lr <= config.initial_lr);
    
    // Test momentum update
    update_momentum(&config, 50);
    
    return true;
}

static bool test_error_handling(void) {
    // Test error setting
    error_set(ERROR_MEMORY, "Test error", __FILE__, __LINE__, __func__);
    TEST_ASSERT(error_get() == ERROR_MEMORY);
    
    // Test error clearing
    error_clear();
    TEST_ASSERT(error_get() == ERROR_NONE);
    
    return true;
}

static bool test_logging(void) {
    LogConfig config = {
        .log_file = "test.log",
        .min_level = LOG_DEBUG,
        .console_output = true,
        .max_file_size = 1024 * 1024,
        .max_files = 3,
        .use_timestamps = true
    };
    
    // Test logging initialization
    TEST_ASSERT(log_init(&config));
    
    // Test different log levels
    LOG_DEBUG("Debug message");
    LOG_INFO("Info message");
    LOG_WARNING("Warning message");
    LOG_ERROR("Error message");
    LOG_FATAL("Fatal message");
    
    // Test log rotation
    log_rotate();
    
    log_cleanup();
    return true;
}

static bool test_checkpointing(void) {
    // Create a model
    Transformer* model = transformer_create(1000, 512, 8, 2048, 6, 6, 0.1f);
    TEST_ASSERT(model != NULL);
    
    // Save checkpoint
    TEST_ASSERT(save_checkpoint(model, "test_checkpoint.bin"));
    
    // Load checkpoint
    Transformer* loaded_model = load_checkpoint("test_checkpoint.bin");
    TEST_ASSERT(loaded_model != NULL);
    
    // Verify model parameters
    TEST_ASSERT(loaded_model->vocab_size == model->vocab_size);
    TEST_ASSERT(loaded_model->d_model == model->d_model);
    TEST_ASSERT(loaded_model->num_heads == model->num_heads);
    TEST_ASSERT(loaded_model->d_ff == model->d_ff);
    TEST_ASSERT(loaded_model->num_encoder_layers == model->num_encoder_layers);
    TEST_ASSERT(loaded_model->num_decoder_layers == model->num_decoder_layers);
    TEST_ASSERT(loaded_model->dropout == model->dropout);
    
    // Cleanup
    transformer_free(model);
    transformer_free(loaded_model);
    return true;
}

int main(void) {
    // Initialize error handling and logging
    error_init();
    LogConfig log_config = {
        .log_file = "test_suite.log",
        .min_level = LOG_DEBUG,
        .console_output = true,
        .max_file_size = 1024 * 1024,
        .max_files = 3,
        .use_timestamps = true
    };
    log_init(&log_config);
    
    // Run all tests
    TEST_RUN(test_matrix_operations);
    TEST_RUN(test_transformer_creation);
    TEST_RUN(test_mixed_precision);
    TEST_RUN(test_distributed_training);
    TEST_RUN(test_scheduler);
    TEST_RUN(test_error_handling);
    TEST_RUN(test_logging);
    TEST_RUN(test_checkpointing);
    
    // Cleanup
    log_cleanup();
    error_cleanup();
    
    printf("All tests passed!\n");
    return 0;
} 