#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir _mkdir
#else
#include <sys/stat.h>
#endif

// Data loading and preprocessing
Dataset* dataset_create(const char* filename, int max_len) {
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    if (!dataset) return NULL;

    FILE* file = fopen(filename, "r");
    if (!file) {
        handle_error(ERROR_FILE, "Failed to open dataset file");
        free(dataset);
        return NULL;
    }

    // Count lines and max length
    int count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        count++;
    }
    rewind(file);

    dataset->data = (int*)safe_malloc(count * max_len * sizeof(int));
    dataset->size = count;
    dataset->max_len = max_len;

    // Read data
    int idx = 0;
    while (fgets(line, sizeof(line), file) && idx < count) {
        char* token = strtok(line, " \t\n");
        int pos = 0;
        while (token && pos < max_len) {
            dataset->data[idx * max_len + pos] = atoi(token);
            token = strtok(NULL, " \t\n");
            pos++;
        }
        // Pad with zeros
        while (pos < max_len) {
            dataset->data[idx * max_len + pos] = 0;
            pos++;
        }
        idx++;
    }

    fclose(file);
    return dataset;
}

void dataset_free(Dataset* dataset) {
    if (!dataset) return;
    free(dataset->data);
    free(dataset);
}

void dataset_shuffle(Dataset* dataset) {
    for (int i = dataset->size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap rows
        for (int k = 0; k < dataset->max_len; k++) {
            int temp = dataset->data[i * dataset->max_len + k];
            dataset->data[i * dataset->max_len + k] = dataset->data[j * dataset->max_len + k];
            dataset->data[j * dataset->max_len + k] = temp;
        }
    }
}

void dataset_split(Dataset* dataset, float train_ratio, 
                  Dataset** train, Dataset** val) {
    int train_size = (int)(dataset->size * train_ratio);
    int val_size = dataset->size - train_size;

    *train = (Dataset*)safe_malloc(sizeof(Dataset));
    *val = (Dataset*)safe_malloc(sizeof(Dataset));

    (*train)->data = (int*)safe_malloc(train_size * dataset->max_len * sizeof(int));
    (*val)->data = (int*)safe_malloc(val_size * dataset->max_len * sizeof(int));

    (*train)->size = train_size;
    (*train)->max_len = dataset->max_len;
    (*val)->size = val_size;
    (*val)->max_len = dataset->max_len;

    // Copy data
    memcpy((*train)->data, dataset->data, 
           train_size * dataset->max_len * sizeof(int));
    memcpy((*val)->data, dataset->data + train_size * dataset->max_len,
           val_size * dataset->max_len * sizeof(int));
}

// Tokenization
Tokenizer* tokenizer_create(const char* vocab_file) {
    Tokenizer* tokenizer = (Tokenizer*)safe_malloc(sizeof(Tokenizer));
    if (!tokenizer) return NULL;

    FILE* file = fopen(vocab_file, "r");
    if (!file) {
        handle_error(ERROR_FILE, "Failed to open vocabulary file");
        free(tokenizer);
        return NULL;
    }

    // Read vocabulary
    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), file)) count++;
    rewind(file);

    tokenizer->vocab = (char*)safe_malloc(count * sizeof(char));
    tokenizer->char_to_idx = (int*)safe_malloc(256 * sizeof(int));
    tokenizer->idx_to_char = (int*)safe_malloc(count * sizeof(int));
    tokenizer->vocab_size = count;

    // Initialize mapping arrays
    memset(tokenizer->char_to_idx, -1, 256 * sizeof(int));
    memset(tokenizer->idx_to_char, -1, count * sizeof(int));

    // Read characters
    int idx = 0;
    while (fgets(line, sizeof(line), file) && idx < count) {
        char c = line[0];
        tokenizer->vocab[idx] = c;
        tokenizer->char_to_idx[(unsigned char)c] = idx;
        tokenizer->idx_to_char[idx] = c;
        idx++;
    }

    fclose(file);
    return tokenizer;
}

void tokenizer_free(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    free(tokenizer->vocab);
    free(tokenizer->char_to_idx);
    free(tokenizer->idx_to_char);
    free(tokenizer);
}

void tokenizer_encode(Tokenizer* tokenizer, const char* text, int* ids, int max_len) {
    int len = strlen(text);
    int pos = 0;
    for (int i = 0; i < len && pos < max_len; i++) {
        int idx = tokenizer->char_to_idx[(unsigned char)text[i]];
        if (idx != -1) {
            ids[pos++] = idx;
        }
    }
    // Pad with zeros
    while (pos < max_len) {
        ids[pos++] = 0;
    }
}

void tokenizer_decode(Tokenizer* tokenizer, const int* ids, int len, char* text) {
    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (ids[i] >= 0 && ids[i] < tokenizer->vocab_size) {
            text[pos++] = tokenizer->idx_to_char[ids[i]];
        }
    }
    text[pos] = '\0';
}

// Logging
Logger* logger_create(const char* log_file, int log_level) {
    Logger* logger = (Logger*)safe_malloc(sizeof(Logger));
    if (!logger) return NULL;

    logger->log_file = fopen(log_file, "a");
    if (!logger->log_file) {
        handle_error(ERROR_FILE, "Failed to open log file");
        free(logger);
        return NULL;
    }

    logger->log_level = log_level;
    return logger;
}

void logger_free(Logger* logger) {
    if (!logger) return;
    if (logger->log_file) fclose(logger->log_file);
    free(logger);
}

void logger_info(Logger* logger, const char* format, ...) {
    if (!logger || logger->log_level < 1) return;
    va_list args;
    va_start(args, format);
    fprintf(logger->log_file, "[INFO] ");
    vfprintf(logger->log_file, format, args);
    fprintf(logger->log_file, "\n");
    va_end(args);
    fflush(logger->log_file);
}

void logger_error(Logger* logger, const char* format, ...) {
    if (!logger || logger->log_level < 2) return;
    va_list args;
    va_start(args, format);
    fprintf(logger->log_file, "[ERROR] ");
    vfprintf(logger->log_file, format, args);
    fprintf(logger->log_file, "\n");
    va_end(args);
    fflush(logger->log_file);
}

void logger_debug(Logger* logger, const char* format, ...) {
    if (!logger || logger->log_level < 0) return;
    va_list args;
    va_start(args, format);
    fprintf(logger->log_file, "[DEBUG] ");
    vfprintf(logger->log_file, format, args);
    fprintf(logger->log_file, "\n");
    va_end(args);
    fflush(logger->log_file);
}

// Memory management
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        handle_error(ERROR_MEMORY, "Memory allocation failed");
        return NULL;
    }
    return ptr;
}

void* safe_calloc(size_t nmemb, size_t size) {
    void* ptr = calloc(nmemb, size);
    if (!ptr) {
        handle_error(ERROR_MEMORY, "Memory allocation failed");
        return NULL;
    }
    return ptr;
}

void* safe_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        handle_error(ERROR_MEMORY, "Memory reallocation failed");
        return NULL;
    }
    return new_ptr;
}

// Error handling
const char* error_to_string(ErrorCode code) {
    switch (code) {
        case ERROR_NONE: return "No error";
        case ERROR_MEMORY: return "Memory allocation failed";
        case ERROR_FILE: return "File operation failed";
        case ERROR_INVALID_INPUT: return "Invalid input";
        case ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        default: return "Unknown error";
    }
}

void handle_error(ErrorCode code, const char* message) {
    fprintf(stderr, "Error %d: %s - %s\n", code, error_to_string(code), message);
}

// Progress bar
ProgressBar* progress_bar_create(int total, int width) {
    ProgressBar* bar = (ProgressBar*)safe_malloc(sizeof(ProgressBar));
    if (!bar) return NULL;
    bar->total = total;
    bar->current = 0;
    bar->width = width;
    return bar;
}

void progress_bar_free(ProgressBar* bar) {
    free(bar);
}

void progress_bar_update(ProgressBar* bar, int current) {
    bar->current = current;
}

void progress_bar_display(ProgressBar* bar) {
    float progress = (float)bar->current / bar->total;
    int filled = (int)(progress * bar->width);
    
    printf("\r[");
    for (int i = 0; i < bar->width; i++) {
        if (i < filled) printf("=");
        else printf(" ");
    }
    printf("] %.1f%%", progress * 100);
    fflush(stdout);
}

// Timing utilities
Timer* timer_create() {
    Timer* timer = (Timer*)safe_malloc(sizeof(Timer));
    if (!timer) return NULL;
    timer->total_time = 0.0;
    timer->num_calls = 0;
    return timer;
}

void timer_free(Timer* timer) {
    free(timer);
}

void timer_start(Timer* timer) {
    timer->start_time = (double)clock() / CLOCKS_PER_SEC;
}

void timer_stop(Timer* timer) {
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    timer->total_time += end_time - timer->start_time;
    timer->num_calls++;
}

double timer_get_average(Timer* timer) {
    return timer->num_calls > 0 ? timer->total_time / timer->num_calls : 0.0;
}

// Random number generation
void set_random_seed(unsigned int seed) {
    srand(seed);
}

float random_uniform(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

float random_normal(float mean, float stddev) {
    // Box-Muller transform
    float u1 = random_uniform(0, 1);
    float u2 = random_uniform(0, 1);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + stddev * z0;
}

// File operations
int file_exists(const char* path) {
    FILE* file = fopen(path, "r");
    if (file) {
        fclose(file);
        return 1;
    }
    return 0;
}

int create_directory(const char* path) {
    return mkdir(path);
}

int remove_file(const char* path) {
    return remove(path);
}

long get_file_size(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) return -1;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    return size;
}

// String utilities
char* str_join(const char** strings, int count, const char* delimiter) {
    if (count <= 0) return NULL;
    
    // Calculate total length
    int total_len = 0;
    int delim_len = strlen(delimiter);
    for (int i = 0; i < count; i++) {
        total_len += strlen(strings[i]);
        if (i < count - 1) total_len += delim_len;
    }
    
    // Allocate and join
    char* result = (char*)safe_malloc(total_len + 1);
    if (!result) return NULL;
    
    int pos = 0;
    for (int i = 0; i < count; i++) {
        strcpy(result + pos, strings[i]);
        pos += strlen(strings[i]);
        if (i < count - 1) {
            strcpy(result + pos, delimiter);
            pos += delim_len;
        }
    }
    result[total_len] = '\0';
    
    return result;
}

char** str_split(const char* string, const char* delimiter, int* count) {
    if (!string || !delimiter || !count) return NULL;
    
    // Count tokens
    char* str = strdup(string);
    char* token = strtok(str, delimiter);
    *count = 0;
    while (token) {
        (*count)++;
        token = strtok(NULL, delimiter);
    }
    free(str);
    
    // Allocate array
    char** result = (char**)safe_malloc(*count * sizeof(char*));
    if (!result) return NULL;
    
    // Split string
    str = strdup(string);
    token = strtok(str, delimiter);
    for (int i = 0; i < *count; i++) {
        result[i] = strdup(token);
        token = strtok(NULL, delimiter);
    }
    free(str);
    
    return result;
}

void str_free_array(char** array, int count) {
    if (!array) return;
    for (int i = 0; i < count; i++) {
        free(array[i]);
    }
    free(array);
} 