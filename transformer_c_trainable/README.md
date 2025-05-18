# C Implementation of Transformer Model

This is a production-ready implementation of the Transformer model in C, following the architecture described in the paper "Attention Is All You Need" by Vaswani et al. The implementation is optimized for performance and includes all necessary components for training and inference.

## Features

- Full implementation of the Transformer architecture
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Positional encoding
- Dropout for regularization
- Adam optimizer
- Cross-entropy loss
- Batch processing
- Gradient clipping
- Model checkpointing
- Comprehensive logging
- Memory-efficient implementation
- OpenMP parallelization

## Requirements

- C compiler (GCC 7.0 or later recommended)
- Make
- OpenMP support
- 64-bit system (for large model support)

## Building

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-c.git
cd transformer-c

# Build the project
make

# For debug build
make debug

# For profiling
make profile
```

## Usage

### Training

```bash
./bin/transformer --train data/train.txt \
                 --vocab data/vocab.txt \
                 --model model.bin \
                 --epochs 10 \
                 --batch-size 32 \
                 --lr 0.0001 \
                 --max-len 512
```

### Command-line Arguments

- `--train <file>`: Training data file (required)
- `--vocab <file>`: Vocabulary file (required)
- `--model <file>`: Model file for saving/loading (optional)
- `--epochs <num>`: Number of training epochs (default: 10)
- `--batch-size <size>`: Batch size (default: 32)
- `--lr <rate>`: Learning rate (default: 0.0001)
- `--max-len <length>`: Maximum sequence length (default: 512)
- `--help`: Show help message

## Project Structure

```
transformer-c/
├── bin/                    # Compiled binaries
├── obj/                    # Object files
├── src/                    # Source files
│   ├── attention.c        # Attention mechanism
│   ├── attention.h
│   ├── config.h           # Configuration
│   ├── layers.c           # Layer implementations
│   ├── layers.h
│   ├── main.c             # Entry point
│   ├── matrix_ops.c       # Matrix operations
│   ├── matrix_ops.h
│   ├── my_math.c          # Math utilities
│   ├── my_math.h
│   ├── training.c         # Training loop
│   ├── training.h
│   ├── transformer_model.c # Transformer model
│   ├── transformer_model.h
│   ├── utils.c            # Utilities
│   └── utils.h
├── Makefile               # Build configuration
└── README.md             # This file
```

## Implementation Details

### Model Architecture

- Embedding layer with learned positional encodings
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Dropout for regularization

### Optimization

- Adam optimizer with learning rate scheduling
- Gradient clipping
- Batch processing
- OpenMP parallelization for matrix operations
- Memory-efficient implementation

### Training

- Cross-entropy loss
- Batch processing
- Validation split
- Model checkpointing
- Comprehensive logging

## Performance

The implementation is optimized for performance:

- OpenMP parallelization for matrix operations
- Memory-efficient implementation
- Optimized attention mechanism
- Efficient batch processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Transformer paper: "Attention Is All You Need" by Vaswani et al.
- PyTorch implementation for reference
- TensorFlow implementation for reference 