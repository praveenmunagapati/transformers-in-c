CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -fopenmp
LDFLAGS = -lm -fopenmp

SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/transformer

# Create directories
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# Compile
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run
run: $(TARGET)
	$(TARGET) --train data/train.txt --vocab data/vocab.txt --model model.bin

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: clean all

# Profile build
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: clean all

.PHONY: all clean run debug profile 