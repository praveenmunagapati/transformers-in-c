#ifndef TRANSFORMER_UTILS_H
#define TRANSFORMER_UTILS_H

#include "matrix_ops.h"
#include "config.h"
#include <stdio.h>

// Data loading and preprocessing
typedef struct {
    int* data;
    int size;
    int max_len;
} Dataset;

Dataset* dataset_create(const char* filename, int max_len);
void dataset_free(Dataset* dataset);
void dataset_shuffle(Dataset* dataset);
void dataset_split(Dataset* dataset, float train_ratio, 
                  Dataset** train, Dataset** val);

// Tokenization (simple character-level for now)
typedef struct {
    char* vocab;
    int vocab_size;
    int* char_to_idx;
    int* idx_to_char;
} Tokenizer;

Tokenizer* tokenizer_create(const char* vocab_file);
void tokenizer_free(Tokenizer* tokenizer);
void tokenizer_encode(Tokenizer* tokenizer, const char* text, int* ids, int max_len);
void tokenizer_decode(Tokenizer* tokenizer, const int* ids, int len, char* text);

// Batch generation
typedef struct {
    int* src_ids;
    int* tgt_ids;
    int src_len;
    int tgt_len;
    int batch_size;
} Batch;

Batch* batch_create(int batch_size, int max_src_len, int max_tgt_len);
void batch_free(Batch* batch);
void generate_batch(Batch* batch, Dataset* dataset, int start_idx);

// Logging and monitoring
typedef struct {
    FILE* log_file;
    int log_level;
} Logger;

Logger* logger_create(const char* log_file, int log_level);
void logger_free(Logger* logger);
void logger_info(Logger* logger, const char* format, ...);
void logger_error(Logger* logger, const char* format, ...);
void logger_debug(Logger* logger, const char* format, ...);

// Memory management
void* safe_malloc(size_t size);
void* safe_calloc(size_t nmemb, size_t size);
void* safe_realloc(void* ptr, size_t size);

// Error handling
typedef enum {
    ERROR_NONE = 0,
    ERROR_MEMORY = -1,
    ERROR_FILE = -2,
    ERROR_INVALID_INPUT = -3,
    ERROR_DIMENSION_MISMATCH = -4
} ErrorCode;

const char* error_to_string(ErrorCode code);
void handle_error(ErrorCode code, const char* message);

// Progress bar
typedef struct {
    int total;
    int current;
    int width;
} ProgressBar;

ProgressBar* progress_bar_create(int total, int width);
void progress_bar_free(ProgressBar* bar);
void progress_bar_update(ProgressBar* bar, int current);
void progress_bar_display(ProgressBar* bar);

// Timing utilities
typedef struct {
    double start_time;
    double total_time;
    int num_calls;
} Timer;

Timer* timer_create();
void timer_free(Timer* timer);
void timer_start(Timer* timer);
void timer_stop(Timer* timer);
double timer_get_average(Timer* timer);

// Random number generation
void set_random_seed(unsigned int seed);
float random_uniform(float min, float max);
float random_normal(float mean, float stddev);

// File operations
int file_exists(const char* path);
int create_directory(const char* path);
int remove_file(const char* path);
long get_file_size(const char* path);

// String utilities
char* str_join(const char** strings, int count, const char* delimiter);
char** str_split(const char* string, const char* delimiter, int* count);
void str_free_array(char** array, int count);

#endif // TRANSFORMER_UTILS_H 