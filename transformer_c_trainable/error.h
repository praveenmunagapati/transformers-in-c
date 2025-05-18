#ifndef TRANSFORMER_ERROR_H
#define TRANSFORMER_ERROR_H

#include <stdint.h>
#include <stdbool.h>

// Error codes
typedef enum {
    ERROR_NONE = 0,
    ERROR_MEMORY = 1,
    ERROR_FILE = 2,
    ERROR_INVALID_INPUT = 3,
    ERROR_INVALID_STATE = 4,
    ERROR_IO = 5,
    ERROR_MATH = 6,
    ERROR_CHECKPOINT = 7,
    ERROR_TRAINING = 8,
    ERROR_VALIDATION = 9,
    ERROR_DISTRIBUTED = 10,
    ERROR_PRECISION = 11,
    ERROR_SCHEDULER = 12,
    ERROR_MAX
} ErrorCode;

// Log levels
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3,
    LOG_FATAL = 4
} LogLevel;

// Log configuration
typedef struct {
    const char* log_file;
    LogLevel min_level;
    bool console_output;
    size_t max_file_size;
    int max_files;
    bool use_timestamps;
} LogConfig;

// Error context
typedef struct {
    ErrorCode code;
    const char* message;
    const char* file;
    int line;
    const char* function;
} ErrorContext;

// Function declarations
void error_init(void);
void error_cleanup(void);
void error_set(ErrorCode code, const char* message, const char* file, int line, const char* function);
ErrorCode error_get(void);
const char* error_get_message(void);
void error_clear(void);

// Logging functions
bool log_init(const LogConfig* config);
void log_cleanup(void);
void log_message(LogLevel level, const char* format, ...);
void log_error(ErrorCode code, const char* format, ...);
void log_rotate(void);

// Error checking macros
#define ERROR_CHECK(expr) do { \
    if (!(expr)) { \
        error_set(ERROR_INVALID_STATE, #expr, __FILE__, __LINE__, __func__); \
        return false; \
    } \
} while(0)

#define ERROR_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        error_set(ERROR_MEMORY, "NULL pointer", __FILE__, __LINE__, __func__); \
        return false; \
    } \
} while(0)

#define ERROR_CHECK_RANGE(val, min, max) do { \
    if ((val) < (min) || (val) > (max)) { \
        error_set(ERROR_INVALID_INPUT, "Value out of range", __FILE__, __LINE__, __func__); \
        return false; \
    } \
} while(0)

// Logging macros
#define LOG_DEBUG(...) log_message(LOG_DEBUG, __VA_ARGS__)
#define LOG_INFO(...) log_message(LOG_INFO, __VA_ARGS__)
#define LOG_WARNING(...) log_message(LOG_WARNING, __VA_ARGS__)
#define LOG_ERROR(...) log_message(LOG_ERROR, __VA_ARGS__)
#define LOG_FATAL(...) log_message(LOG_FATAL, __VA_ARGS__)

#endif // TRANSFORMER_ERROR_H 