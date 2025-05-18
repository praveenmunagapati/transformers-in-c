#ifndef TRANSFORMER_CONFIG_H
#define TRANSFORMER_CONFIG_H

// Model Architecture Parameters
#define VOCAB_SIZE 32000        // Size of vocabulary
#define MAX_SEQ_LENGTH 512      // Maximum sequence length
#define D_MODEL 512            // Dimension of model embeddings
#define NUM_HEADS 8            // Number of attention heads
#define NUM_ENCODER_LAYERS 6   // Number of encoder layers
#define NUM_DECODER_LAYERS 6   // Number of decoder layers
#define D_FF 2048             // Dimension of feed-forward network
#define DROPOUT_RATE 0.1       // Dropout rate

// Training Parameters
#define BATCH_SIZE 32          // Batch size for training
#define LEARNING_RATE 0.0001   // Learning rate
#define WARMUP_STEPS 4000      // Number of warmup steps
#define MAX_EPOCHS 100         // Maximum number of training epochs

// Mathematical Constants
#define EPSILON 1e-6          // Small constant for numerical stability
#define SCALE_FACTOR 0.1       // Scaling factor for initialization

// Memory Management
#define MAX_BATCH_SIZE 64      // Maximum allowed batch size
#define MAX_GRAD_NORM 1.0      // Maximum gradient norm for clipping

// Error Codes
#define SUCCESS 0
#define ERROR_MEMORY_ALLOCATION -1
#define ERROR_INVALID_INPUT -2
#define ERROR_DIMENSION_MISMATCH -3

#endif // TRANSFORMER_CONFIG_H 