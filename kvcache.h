#pragma once

#include "tensor.h"

#include <stddef.h>

enum kv_mode { KV_PREFILL, KV_DECODE };

struct kvcache {
	struct {
		tensor_t *k;	/* (H, C, HLEN) */
		tensor_t *v;
	} *hl;
	size_t size;
	size_t context;
	size_t heads;
	size_t head_len;
	size_t layers;
};

struct kvcache *kvcache_alloc(size_t layers, size_t context, size_t heads, size_t head_len);
void kvcache_rotate(struct kvcache *kv);
void kvcache_free(struct kvcache *kv);

void kvcache_get_k(struct kvcache *kv, size_t l, size_t h_idx, tensor_t *out);
void kvcache_get_v(struct kvcache *kv, size_t l, size_t h_idx, tensor_t *out);
