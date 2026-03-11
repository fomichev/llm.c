#pragma once

#include <stddef.h>

#include "tensor.h"

struct snapshot;
struct gpt2;

typedef size_t (*pick_token_t)(void *ctx, tensor_t *logits);

struct gpt2 *gpt2_load(struct snapshot *ss);
void gpt2_test_no_cache(struct gpt2 *model);
void gpt2_test_cache(struct gpt2 *model);
void gpt2_prefill(struct gpt2 *model, int *tok, int *pos, size_t T, tensor_t *output);
void gpt2_decode(struct gpt2 *model, int tok, int pos, tensor_t *output);
void gpt2_generate(struct gpt2 *model, const char *text, int num, pick_token_t f, void *ctx);
void gpt2_close(struct gpt2 *model);
