#pragma once

#include <stddef.h>

#include "tensor.h"

struct gguf;

typedef size_t (*pick_token_t)(void *ctx, tensor_t *logits);

struct model {
	const char *name;
	void *(*load)(struct gguf *g);
	void (*generate)(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx);
	void (*close)(void *ctx);
};

void register_model(const struct model *m);
const struct model *find_model(const char *name);
