#include "kvcache.h"

#include <stdlib.h>
#include <string.h>

struct kvcache *kvcache_alloc(size_t layers, size_t context, size_t heads, size_t head_len)
{
	struct kvcache *kv;

	kv = calloc(1, sizeof(*kv));
	if (!kv)
		return NULL;

	kv->layers = layers;
	kv->context = context;
	kv->heads = heads;
	kv->head_len = head_len;

	kv->hl = calloc(layers, sizeof(*kv->hl));
	assert(kv->hl);

	for (size_t i = 0; i < layers; i++) {
		kv->hl[i].k = tensor_new_zero(3, heads, context, head_len);
		kv->hl[i].v = tensor_new_zero(3, heads, context, head_len);
	}

	return kv;
}

void kvcache_rotate(struct kvcache *kv)
{
	size_t H = kv->heads;
	size_t C = kv->context;
	size_t HLEN = kv->head_len;
	size_t half = C / 2;

	for (size_t i = 0; i < kv->layers; i++) {
		scalar_t *kd = kv->hl[i].k->data;
		scalar_t *vd = kv->hl[i].v->data;

		for (size_t h = 0; h < H; h++) {
			scalar_t *kb = kd + h * C * HLEN;
			scalar_t *vb = vd + h * C * HLEN;

			memmove(kb, kb + half * HLEN, (C - half) * HLEN * sizeof(scalar_t));
			memmove(vb, vb + half * HLEN, (C - half) * HLEN * sizeof(scalar_t));
		}
	}

	kv->size -= half;
}

void kvcache_free(struct kvcache *kv)
{
	for (size_t i = 0; i < kv->layers; i++) {
		tensor_free(kv->hl[i].k);
		tensor_free(kv->hl[i].v);
	}
	free(kv->hl);
	free(kv);
}

void kvcache_get_k(struct kvcache *kv, size_t l, size_t h_idx, tensor_t *out)
{
	tensor_at(kv->hl[l].k, h_idx, out);
}

void kvcache_get_v(struct kvcache *kv, size_t l, size_t h_idx, tensor_t *out)
{
	tensor_at(kv->hl[l].v, h_idx, out);
}
