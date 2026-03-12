#pragma once

#include <stddef.h>

#include "tensor.h"
#include "model.h"

struct gguf;
struct gpt2;

void *gpt2_load(struct gguf *g);
void gpt2_prefill(struct gpt2 *model, int *tok, int *pos, size_t T, tensor_t *output);
void gpt2_decode(struct gpt2 *model, int tok, int pos, tensor_t *output);
void gpt2_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx);
void gpt2_close(void *ctx);
