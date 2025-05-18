#include "transformer_model.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Embedding layer
Embedding* embedding_create(int vocab_size, int d_model) {
    Embedding* emb = (Embedding*)malloc(sizeof(Embedding));
    if (!emb) return NULL;
    emb->weights = matrix_create(vocab_size, d_model);
    initialize_weights(emb->weights->data, vocab_size * d_model, SCALE_FACTOR);
    return emb;
}

void embedding_free(Embedding* emb) {
    if (!emb) return;
    matrix_free(emb->weights);
    free(emb);
}

void embedding_forward(Embedding* emb, const int* input_ids, int seq_len, Matrix* output) {
    // output: [seq_len, d_model]
    for (int i = 0; i < seq_len; i++) {
        int idx = input_ids[i];
        memcpy(output->data + i * emb->weights->cols,
               emb->weights->data + idx * emb->weights->cols,
               emb->weights->cols * sizeof(float));
    }
}

// Sinusoidal positional encoding
void positional_encoding(Matrix* pos_enc, int max_seq_len, int d_model) {
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = pos / powf(10000, 2 * (i / 2) / (float)d_model);
            if (i % 2 == 0)
                pos_enc->data[pos * d_model + i] = sinf(angle);
            else
                pos_enc->data[pos * d_model + i] = cosf(angle);
        }
    }
}

// Encoder layer
EncoderLayer* encoder_layer_create(int d_model, int num_heads, int d_ff) {
    EncoderLayer* layer = (EncoderLayer*)malloc(sizeof(EncoderLayer));
    if (!layer) return NULL;
    layer->self_attn = attention_create(d_model, num_heads);
    layer->norm1 = layer_norm_create(d_model);
    layer->ffn = feed_forward_create(d_model, d_ff);
    layer->norm2 = layer_norm_create(d_model);
    return layer;
}

void encoder_layer_free(EncoderLayer* layer) {
    if (!layer) return;
    attention_free(layer->self_attn);
    layer_norm_free(layer->norm1);
    feed_forward_free(layer->ffn);
    layer_norm_free(layer->norm2);
    free(layer);
}

void encoder_layer_forward(EncoderLayer* layer, Matrix* input, Matrix* mask, Matrix* output) {
    // Self-attention + Add & Norm
    Matrix* attn_out = matrix_create(input->rows, input->cols);
    attention_forward(layer->self_attn, input, mask, attn_out);
    matrix_add_inplace(attn_out, input); // Residual
    layer_norm_forward(layer->norm1, attn_out, attn_out);
    // Feed-forward + Add & Norm
    Matrix* ffn_out = matrix_create(input->rows, input->cols);
    feed_forward_forward(layer->ffn, attn_out, ffn_out);
    matrix_add_inplace(ffn_out, attn_out); // Residual
    layer_norm_forward(layer->norm2, ffn_out, output);
    matrix_free(attn_out);
    matrix_free(ffn_out);
}

// Encoder stack
Encoder* encoder_create(int num_layers, int d_model, int num_heads, int d_ff) {
    Encoder* enc = (Encoder*)malloc(sizeof(Encoder));
    if (!enc) return NULL;
    enc->num_layers = num_layers;
    enc->layers = (EncoderLayer**)malloc(num_layers * sizeof(EncoderLayer*));
    for (int i = 0; i < num_layers; i++)
        enc->layers[i] = encoder_layer_create(d_model, num_heads, d_ff);
    return enc;
}

void encoder_free(Encoder* enc) {
    if (!enc) return;
    for (int i = 0; i < enc->num_layers; i++)
        encoder_layer_free(enc->layers[i]);
    free(enc->layers);
    free(enc);
}

void encoder_forward(Encoder* enc, Matrix* input, Matrix* mask, Matrix* output) {
    Matrix* x = matrix_clone(input);
    Matrix* tmp = matrix_create(input->rows, input->cols);
    for (int i = 0; i < enc->num_layers; i++) {
        encoder_layer_forward(enc->layers[i], x, mask, tmp);
        matrix_copy(x, tmp);
    }
    matrix_copy(output, x);
    matrix_free(x);
    matrix_free(tmp);
}

// Decoder layer
DecoderLayer* decoder_layer_create(int d_model, int num_heads, int d_ff) {
    DecoderLayer* layer = (DecoderLayer*)malloc(sizeof(DecoderLayer));
    if (!layer) return NULL;
    layer->self_attn = attention_create(d_model, num_heads);
    layer->cross_attn = attention_create(d_model, num_heads);
    layer->norm1 = layer_norm_create(d_model);
    layer->norm2 = layer_norm_create(d_model);
    layer->ffn = feed_forward_create(d_model, d_ff);
    layer->norm3 = layer_norm_create(d_model);
    return layer;
}

void decoder_layer_free(DecoderLayer* layer) {
    if (!layer) return;
    attention_free(layer->self_attn);
    attention_free(layer->cross_attn);
    layer_norm_free(layer->norm1);
    layer_norm_free(layer->norm2);
    feed_forward_free(layer->ffn);
    layer_norm_free(layer->norm3);
    free(layer);
}

void decoder_layer_forward(DecoderLayer* layer, Matrix* input, Matrix* enc_output, Matrix* self_mask, Matrix* enc_dec_mask, Matrix* output) {
    // Self-attention + Add & Norm
    Matrix* attn_out = matrix_create(input->rows, input->cols);
    attention_forward(layer->self_attn, input, self_mask, attn_out);
    matrix_add_inplace(attn_out, input);
    layer_norm_forward(layer->norm1, attn_out, attn_out);
    // Cross-attention + Add & Norm
    attention_forward(layer->cross_attn, attn_out, enc_dec_mask, attn_out);
    matrix_add_inplace(attn_out, attn_out);
    layer_norm_forward(layer->norm2, attn_out, attn_out);
    // Feed-forward + Add & Norm
    Matrix* ffn_out = matrix_create(input->rows, input->cols);
    feed_forward_forward(layer->ffn, attn_out, ffn_out);
    matrix_add_inplace(ffn_out, attn_out);
    layer_norm_forward(layer->norm3, ffn_out, output);
    matrix_free(attn_out);
    matrix_free(ffn_out);
}

// Decoder stack
Decoder* decoder_create(int num_layers, int d_model, int num_heads, int d_ff) {
    Decoder* dec = (Decoder*)malloc(sizeof(Decoder));
    if (!dec) return NULL;
    dec->num_layers = num_layers;
    dec->layers = (DecoderLayer**)malloc(num_layers * sizeof(DecoderLayer*));
    for (int i = 0; i < num_layers; i++)
        dec->layers[i] = decoder_layer_create(d_model, num_heads, d_ff);
    return dec;
}

void decoder_free(Decoder* dec) {
    if (!dec) return;
    for (int i = 0; i < dec->num_layers; i++)
        decoder_layer_free(dec->layers[i]);
    free(dec->layers);
    free(dec);
}

void decoder_forward(Decoder* dec, Matrix* input, Matrix* enc_output, Matrix* self_mask, Matrix* enc_dec_mask, Matrix* output) {
    Matrix* x = matrix_clone(input);
    Matrix* tmp = matrix_create(input->rows, input->cols);
    for (int i = 0; i < dec->num_layers; i++) {
        decoder_layer_forward(dec->layers[i], x, enc_output, self_mask, enc_dec_mask, tmp);
        matrix_copy(x, tmp);
    }
    matrix_copy(output, x);
    matrix_free(x);
    matrix_free(tmp);
}

// Full transformer
Transformer* transformer_create() {
    Transformer* model = (Transformer*)malloc(sizeof(Transformer));
    model->src_embedding = embedding_create(VOCAB_SIZE, D_MODEL);
    model->tgt_embedding = embedding_create(VOCAB_SIZE, D_MODEL);
    model->src_pos_encoding = matrix_create(MAX_SEQ_LENGTH, D_MODEL);
    model->tgt_pos_encoding = matrix_create(MAX_SEQ_LENGTH, D_MODEL);
    positional_encoding(model->src_pos_encoding, MAX_SEQ_LENGTH, D_MODEL);
    positional_encoding(model->tgt_pos_encoding, MAX_SEQ_LENGTH, D_MODEL);
    model->encoder = encoder_create(NUM_ENCODER_LAYERS, D_MODEL, NUM_HEADS, D_FF);
    model->decoder = decoder_create(NUM_DECODER_LAYERS, D_MODEL, NUM_HEADS, D_FF);
    model->output_proj = matrix_create(D_MODEL, VOCAB_SIZE);
    initialize_weights(model->output_proj->data, D_MODEL * VOCAB_SIZE, SCALE_FACTOR);
    return model;
}

void transformer_free(Transformer* model) {
    if (!model) return;
    embedding_free(model->src_embedding);
    embedding_free(model->tgt_embedding);
    matrix_free(model->src_pos_encoding);
    matrix_free(model->tgt_pos_encoding);
    encoder_free(model->encoder);
    decoder_free(model->decoder);
    matrix_free(model->output_proj);
    free(model);
}

void transformer_forward(Transformer* model, const int* src_ids, int src_len, const int* tgt_ids, int tgt_len, Matrix* output) {
    // Embedding + Positional encoding for source
    Matrix* src_emb = matrix_create(src_len, D_MODEL);
    embedding_forward(model->src_embedding, src_ids, src_len, src_emb);
    for (int i = 0; i < src_len * D_MODEL; i++)
        src_emb->data[i] += model->src_pos_encoding->data[i];
    // Embedding + Positional encoding for target
    Matrix* tgt_emb = matrix_create(tgt_len, D_MODEL);
    embedding_forward(model->tgt_embedding, tgt_ids, tgt_len, tgt_emb);
    for (int i = 0; i < tgt_len * D_MODEL; i++)
        tgt_emb->data[i] += model->tgt_pos_encoding->data[i];
    // Encoder forward
    Matrix* enc_out = matrix_create(src_len, D_MODEL);
    encoder_forward(model->encoder, src_emb, NULL, enc_out);
    // Decoder forward
    Matrix* dec_out = matrix_create(tgt_len, D_MODEL);
    decoder_forward(model->decoder, tgt_emb, enc_out, NULL, NULL, dec_out);
    // Output projection
    matrix_multiply(output->data, dec_out->data, model->output_proj->data, tgt_len, D_MODEL, VOCAB_SIZE);
    matrix_free(src_emb);
    matrix_free(tgt_emb);
    matrix_free(enc_out);
    matrix_free(dec_out);
} 