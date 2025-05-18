#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <pthread.h>
#include <sys/stat.h>
#include <errno.h>

#define MAX_ERROR_MESSAGE 1024
#define MAX_LOG_FILES 10
#define DEFAULT_MAX_FILE_SIZE (10 * 1024 * 1024) // 10MB

static ErrorContext g_error = {0};
static LogConfig g_log_config = {0};
static FILE* g_log_file = NULL;
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_error_mutex = PTHREAD_MUTEX_INITIALIZER;

void error_init(void) {
    pthread_mutex_init(&g_error_mutex, NULL);
    error_clear();
}

void error_cleanup(void) {
    pthread_mutex_destroy(&g_error_mutex);
}

void error_set(ErrorCode code, const char* message, const char* file, int line, const char* function) {
    pthread_mutex_lock(&g_error_mutex);
    g_error.code = code;
    g_error.message = message;
    g_error.file = file;
    g_error.line = line;
    g_error.function = function;
    pthread_mutex_unlock(&g_error_mutex);
    
    log_error(code, "Error in %s:%d (%s): %s", file, line, function, message);
}

ErrorCode error_get(void) {
    pthread_mutex_lock(&g_error_mutex);
    ErrorCode code = g_error.code;
    pthread_mutex_unlock(&g_error_mutex);
    return code;
}

const char* error_get_message(void) {
    pthread_mutex_lock(&g_error_mutex);
    const char* msg = g_error.message;
    pthread_mutex_unlock(&g_error_mutex);
    return msg;
}

void error_clear(void) {
    pthread_mutex_lock(&g_error_mutex);
    memset(&g_error, 0, sizeof(g_error));
    pthread_mutex_unlock(&g_error_mutex);
}

static void rotate_log_file(void) {
    if (!g_log_file) return;
    
    fclose(g_log_file);
    
    // Rotate existing log files
    char old_name[256];
    char new_name[256];
    for (int i = g_log_config.max_files - 1; i > 0; i--) {
        snprintf(old_name, sizeof(old_name), "%s.%d", g_log_config.log_file, i);
        snprintf(new_name, sizeof(new_name), "%s.%d", g_log_config.log_file, i + 1);
        rename(old_name, new_name);
    }
    
    // Rename current log file
    snprintf(old_name, sizeof(old_name), "%s", g_log_config.log_file);
    snprintf(new_name, sizeof(new_name), "%s.1", g_log_config.log_file);
    rename(old_name, new_name);
    
    // Open new log file
    g_log_file = fopen(g_log_config.log_file, "w");
    if (!g_log_file) {
        fprintf(stderr, "Failed to open new log file: %s\n", g_log_config.log_file);
    }
}

bool log_init(const LogConfig* config) {
    if (!config || !config->log_file) {
        return false;
    }
    
    pthread_mutex_lock(&g_log_mutex);
    
    // Copy configuration
    memcpy(&g_log_config, config, sizeof(LogConfig));
    
    // Set defaults if not specified
    if (g_log_config.max_file_size == 0) {
        g_log_config.max_file_size = DEFAULT_MAX_FILE_SIZE;
    }
    if (g_log_config.max_files == 0) {
        g_log_config.max_files = MAX_LOG_FILES;
    }
    
    // Open log file
    g_log_file = fopen(g_log_config.log_file, "a");
    if (!g_log_file) {
        pthread_mutex_unlock(&g_log_mutex);
        return false;
    }
    
    pthread_mutex_unlock(&g_log_mutex);
    return true;
}

void log_cleanup(void) {
    pthread_mutex_lock(&g_log_mutex);
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
    pthread_mutex_unlock(&g_log_mutex);
}

static const char* get_log_level_string(LogLevel level) {
    switch (level) {
        case LOG_DEBUG: return "DEBUG";
        case LOG_INFO: return "INFO";
        case LOG_WARNING: return "WARNING";
        case LOG_ERROR: return "ERROR";
        case LOG_FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

void log_message(LogLevel level, const char* format, ...) {
    if (level < g_log_config.min_level) return;
    
    pthread_mutex_lock(&g_log_mutex);
    
    // Check if we need to rotate the log file
    if (g_log_file) {
        struct stat st;
        if (fstat(fileno(g_log_file), &st) == 0 && st.st_size >= g_log_config.max_file_size) {
            rotate_log_file();
        }
    }
    
    // Get current time
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Format the message
    va_list args;
    va_start(args, format);
    char message[MAX_ERROR_MESSAGE];
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    
    // Write to log file
    if (g_log_file) {
        fprintf(g_log_file, "[%s] [%s] %s\n", 
                timestamp, get_log_level_string(level), message);
        fflush(g_log_file);
    }
    
    // Write to console if enabled
    if (g_log_config.console_output) {
        fprintf(stderr, "[%s] [%s] %s\n", 
                timestamp, get_log_level_string(level), message);
    }
    
    pthread_mutex_unlock(&g_log_mutex);
}

void log_error(ErrorCode code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    char message[MAX_ERROR_MESSAGE];
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    
    log_message(LOG_ERROR, "Error %d: %s", code, message);
}

void log_rotate(void) {
    pthread_mutex_lock(&g_log_mutex);
    rotate_log_file();
    pthread_mutex_unlock(&g_log_mutex);
} 