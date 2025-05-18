#include "transformer_model.h"
#include "training.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --train <data_file>     Train the model on specified data file\n");
    printf("  --vocab <vocab_file>    Vocabulary file for tokenization\n");
    printf("  --model <model_file>    Save/load model from specified file\n");
    printf("  --epochs <num>          Number of training epochs (default: 10)\n");
    printf("  --batch-size <size>     Batch size for training (default: 32)\n");
    printf("  --lr <rate>             Learning rate (default: 0.0001)\n");
    printf("  --max-len <length>      Maximum sequence length (default: 512)\n");
    printf("  --help                  Show this help message\n");
}

int main(int argc, char** argv) {
    // Default parameters
    const char* train_file = NULL;
    const char* vocab_file = NULL;
    const char* model_file = NULL;
    int num_epochs = 10;
    int batch_size = 32;
    float learning_rate = 0.0001f;
    int max_len = 512;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_file = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_file = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_file = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-len") == 0 && i + 1 < argc) {
            max_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate required arguments
    if (!train_file || !vocab_file) {
        printf("Error: Training data file and vocabulary file are required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Initialize random seed
    set_random_seed(time(NULL));

    // Create logger
    Logger* logger = logger_create("transformer.log", 1);
    if (!logger) {
        printf("Error: Failed to create logger\n");
        return 1;
    }

    // Load vocabulary
    Tokenizer* tokenizer = tokenizer_create(vocab_file);
    if (!tokenizer) {
        logger_error(logger, "Failed to create tokenizer");
        logger_free(logger);
        return 1;
    }

    // Load dataset
    Dataset* dataset = dataset_create(train_file, max_len);
    if (!dataset) {
        logger_error(logger, "Failed to load dataset");
        tokenizer_free(tokenizer);
        logger_free(logger);
        return 1;
    }

    // Split dataset
    Dataset* train_data = NULL;
    Dataset* val_data = NULL;
    dataset_split(dataset, 0.9f, &train_data, &val_data);
    dataset_free(dataset);

    // Create model
    Transformer* model = transformer_create(
        tokenizer->vocab_size,  // vocab_size
        512,                    // d_model
        8,                      // num_heads
        2048,                   // d_ff
        6,                      // num_encoder_layers
        6,                      // num_decoder_layers
        0.1f                    // dropout
    );

    if (!model) {
        logger_error(logger, "Failed to create transformer model");
        dataset_free(train_data);
        dataset_free(val_data);
        tokenizer_free(tokenizer);
        logger_free(logger);
        return 1;
    }

    // Create optimizer
    AdamOptimizer* optimizer = adam_create(model);

    if (!optimizer) {
        logger_error(logger, "Failed to create optimizer");
        transformer_free(model);
        dataset_free(train_data);
        dataset_free(val_data);
        tokenizer_free(tokenizer);
        logger_free(logger);
        return 1;
    }

    // Training loop
    logger_info(logger, "Starting training...");
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        logger_info(logger, "Epoch %d/%d", epoch + 1, num_epochs);
        
        // Create batches for training
        int num_train_batches = (train_data->size + batch_size - 1) / batch_size;
        Batch** train_batches = (Batch**)malloc(num_train_batches * sizeof(Batch*));
        for (int i = 0; i < num_train_batches; i++) {
            train_batches[i] = batch_create(batch_size, max_len, max_len);
            generate_batch(train_batches[i], train_data, i * batch_size);
        }
        
        // Train for one epoch
        TrainingStats* stats = training_stats_create(num_train_batches);
        train_epoch(model, optimizer, train_batches, num_train_batches, stats);
        float train_loss = stats->losses[stats->current_step - 1];
        training_stats_free(stats);
        
        // Free training batches
        for (int i = 0; i < num_train_batches; i++) {
            batch_free(train_batches[i]);
        }
        free(train_batches);

        // Validate
        float val_loss = validate_epoch(model, val_data, batch_size, max_len);
        logger_info(logger, "Train loss: %.4f, Val loss: %.4f", train_loss, val_loss);

        // Save checkpoint
        if (model_file) {
            char checkpoint[256];
            snprintf(checkpoint, sizeof(checkpoint), "%s.epoch%d", model_file, epoch + 1);
            save_checkpoint(model, checkpoint);
        }
    }

    // Save final model
    if (model_file) {
        save_checkpoint(model, model_file);
    }

    // Cleanup
    adam_free(optimizer);
    transformer_free(model);
    dataset_free(train_data);
    dataset_free(val_data);
    tokenizer_free(tokenizer);
    logger_free(logger);

    return 0;
} 