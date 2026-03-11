#include "kvcache.h"

#include <stdlib.h>
#include <string.h>

struct kvcache *kvcache_alloc(size_t layers, size_t context, size_t embeddings)
{
	struct kvcache *kv;

	kv = calloc(1, sizeof(*kv));
	if (!kv)
		return NULL;

	kv->layers = layers;
	kv->context = context;
	kv->embeddings = embeddings;

	kv->hl = calloc(layers, sizeof(*kv->hl));
	assert(kv->hl);

	for (size_t i = 0; i < layers; i++) {
		kv->hl[i].k = tensor_new_zero(2, context, embeddings);
		kv->hl[i].v = tensor_new_zero(2, context, embeddings);
	}

	return kv;
}

void kvcache_rotate(struct kvcache *kv)
{
	size_t C = kv->context;
	size_t E = kv->embeddings;
	size_t half = C / 2;

	for (size_t i = 0; i < kv->layers; i++) {
		scalar_t *kd = kv->hl[i].k->data;
		scalar_t *vd = kv->hl[i].v->data;

		memmove(kd, kd + half * E, (C - half) * E * sizeof(scalar_t));
		memmove(vd, vd + half * E, (C - half) * E * sizeof(scalar_t));
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

void kvcache_get_k(struct kvcache *kv, size_t l, size_t t_idx, tensor_t *out)
{
	tensor_at(kv->hl[l].k, t_idx, out);
}

void kvcache_get_v(struct kvcache *kv, size_t l, size_t t_idx, tensor_t *out)
{
	tensor_at(kv->hl[l].v, t_idx, out);
}

void kvcache_set_k(struct kvcache *kv, size_t l, size_t t_idx, const tensor_t *src)
{
	tensor_t row;
	tensor_at(kv->hl[l].k, t_idx, &row);
	tensor_copy(&row, src);
}

void kvcache_set_v(struct kvcache *kv, size_t l, size_t t_idx, const tensor_t *src)
{
	tensor_t row;
	tensor_at(kv->hl[l].v, t_idx, &row);
	tensor_copy(&row, src);
}
