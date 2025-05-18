#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include "matrix_ops.h"
#include "attention.h"
#include "layers.h"
#include "config.h"

// Embedding layer
typedef struct {
    Matrix* weights; // [vocab_size, d_model]
} Embedding;

Embedding* embedding_create(int vocab_size, int d_model);
void embedding_free(Embedding* emb);
void embedding_forward(Embedding* emb, const int* input_ids, int seq_len, Matrix* output);

// Positional encoding
void positional_encoding(Matrix* pos_enc, int max_seq_len, int d_model);

// Encoder layer
typedef struct {
    AttentionLayer* self_attn;
    LayerNorm* norm1;
    FeedForward* ffn;
    LayerNorm* norm2;
} EncoderLayer;

EncoderLayer* encoder_layer_create(int d_model, int num_heads, int d_ff);
void encoder_layer_free(EncoderLayer* layer);
void encoder_layer_forward(EncoderLayer* layer, Matrix* input, Matrix* mask, Matrix* output);

// Encoder stack
typedef struct {
    EncoderLayer** layers;
    int num_layers;
} Encoder;

Encoder* encoder_create(int num_layers, int d_model, int num_heads, int d_ff);
void encoder_free(Encoder* enc);
void encoder_forward(Encoder* enc, Matrix* input, Matrix* mask, Matrix* output);

// Decoder layer (stub for now)
typedef struct {
    AttentionLayer* self_attn;
    AttentionLayer* cross_attn;
    LayerNorm* norm1;
    LayerNorm* norm2;
    FeedForward* ffn;
    LayerNorm* norm3;
} DecoderLayer;

DecoderLayer* decoder_layer_create(int d_model, int num_heads, int d_ff);
void decoder_layer_free(DecoderLayer* layer);
void decoder_layer_forward(DecoderLayer* layer, Matrix* input, Matrix* enc_output, Matrix* self_mask, Matrix* enc_dec_mask, Matrix* output);

// Decoder stack
typedef struct {
    DecoderLayer** layers;
    int num_layers;
} Decoder;

Decoder* decoder_create(int num_layers, int d_model, int num_heads, int d_ff);
void decoder_free(Decoder* dec);
void decoder_forward(Decoder* dec, Matrix* input, Matrix* enc_output, Matrix* self_mask, Matrix* enc_dec_mask, Matrix* output);

// Full transformer model
typedef struct {
    Embedding* src_embedding;
    Embedding* tgt_embedding;
    Matrix* src_pos_encoding;
    Matrix* tgt_pos_encoding;
    Encoder* encoder;
    Decoder* decoder;
    Matrix* output_proj; // [d_model, vocab_size]
} Transformer;

Transformer* transformer_create();
void transformer_free(Transformer* model);
void transformer_forward(Transformer* model, const int* src_ids, int src_len, const int* tgt_ids, int tgt_len, Matrix* output);

#endif // TRANSFORMER_MODEL_H 