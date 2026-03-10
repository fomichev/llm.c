#pragma once

#include "tensor.h"

#include <stddef.h>

enum kv_mode { KV_PREFILL, KV_DECODE };

struct kvcache {
	struct {
		tensor_t *k;
		tensor_t *v;
	} *hl;
	size_t size;
	size_t context;
	size_t embeddings;
	size_t layers;
};

struct kvcache *kvcache_alloc(size_t layers, size_t context, size_t embeddings);
void kvcache_rotate(struct kvcache *kv);
void kvcache_free(struct kvcache *kv);

void kvcache_get_k(struct kvcache *kv, size_t l, size_t t_idx, tensor_t *out);
void kvcache_get_v(struct kvcache *kv, size_t l, size_t t_idx, tensor_t *out);
void kvcache_set_k(struct kvcache *kv, size_t l, size_t t_idx, const tensor_t *src);
void kvcache_set_v(struct kvcache *kv, size_t l, size_t t_idx, const tensor_t *src);
