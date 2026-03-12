#include "gpt2.h"
#include "nn.h"
#include "gguf.h"
#include "vocab.h"
#include "kvcache.h"
#include "simd.h"
#include "profiler.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct gpt2 {
	struct gguf *gguf;

	struct {
		size_t context; /* context size */
		size_t heads; /* number of heads */
		size_t head_len; /* head size */
		size_t layers; /* number of hidden layers */
		size_t embeddings; /* embeddings size */
		size_t vocab_len; /* vocabulary size */
	};

	scalar_t hlen_sq;

	const tensor_t *wte;
	const tensor_t *wpe;
	struct gpt2h {
		const tensor_t *ln_1_weight;
		const tensor_t *ln_1_bias;

		struct gpt2attn {
			const tensor_t *c_attn_weight;
			const tensor_t *c_attn_bias;

			const tensor_t *c_proj_weight;
			const tensor_t *c_proj_bias;
		} attn;

		const tensor_t *ln_2_weight;
		const tensor_t *ln_2_bias;

		struct gpt2mlp {
			const tensor_t *c_fc_weight;
			const tensor_t *c_fc_bias;
			const tensor_t *c_proj_weight;
			const tensor_t *c_proj_bias;
		} mlp;
	} *hl;
	const tensor_t *ln_f_weight;
	const tensor_t *ln_f_bias;

	struct {
		tensor_t *output;
		tensor_t *tokens;
		tensor_t *positions;
		tensor_t *tokens_attn;
		tensor_t *qh; /* heads X from all tokens */
		tensor_t *kh; /* heads X from all tokens */
		tensor_t *vh; /* heads X from all tokens */
		tensor_t *masked_attn;
		tensor_t *attn;
		tensor_t *attn_residual;
		tensor_t *mlp_fc;
		tensor_t *logits;
	} state;

	struct kvcache *cache;
};

void *gpt2_load(struct gguf *g)
{
	struct gpt2 *model;

	model = calloc(1, sizeof(*model));
	if (!model)
		return NULL;

	model->gguf = g;

	model->context = gguf_get_uint32(g, "gpt2.context_length"); /* 1024 */
	model->embeddings = gguf_get_uint32(g, "gpt2.embedding_length"); /* 768 */
	model->heads = gguf_get_uint32(g, "gpt2.attention.head_count"); /* 12 */
	model->layers = gguf_get_uint32(g, "gpt2.block_count"); /* 12 */
	model->head_len = model->embeddings / model->heads; /* 64 */
	model->vocab_len = gguf_get_arr_n(g, "tokenizer.ggml.tokens"); /* 50257 */

	size_t C = model->context;
	size_t HLEN = model->head_len;
	size_t H = model->heads;
	size_t E = model->embeddings;

	model->hlen_sq = sqrt((scalar_t)HLEN);

	assert(H * HLEN == E);

	model->hl = calloc(model->layers, sizeof(*model->hl));
	assert(model->hl);

	model->wte = gguf_tensor_2d(g, model->vocab_len, E, "token_embd.weight");
	model->wpe = gguf_tensor_2d(g, C, E, "position_embd.weight");

	for (size_t i = 0; i < model->layers; i++) {
		model->hl[i].ln_1_weight = gguf_tensor_1d(g, E, "blk.%zu.attn_norm.weight", i);
		model->hl[i].ln_1_bias = gguf_tensor_1d(g, E, "blk.%zu.attn_norm.bias", i);

		model->hl[i].attn.c_attn_weight = gguf_tensor_2d(g, E * 3, E, "blk.%zu.attn_qkv.weight", i);
		model->hl[i].attn.c_attn_bias = gguf_tensor_1d(g, E * 3, "blk.%zu.attn_qkv.bias", i);

		model->hl[i].attn.c_proj_weight = gguf_tensor_2d(g, E, E, "blk.%zu.attn_output.weight", i);
		model->hl[i].attn.c_proj_bias = gguf_tensor_1d(g, E, "blk.%zu.attn_output.bias", i);

		model->hl[i].ln_2_weight = gguf_tensor_1d(g, E, "blk.%zu.ffn_norm.weight", i);
		model->hl[i].ln_2_bias = gguf_tensor_1d(g, E, "blk.%zu.ffn_norm.bias", i);

		model->hl[i].mlp.c_fc_weight = gguf_tensor_2d(g, E * 4, E, "blk.%zu.ffn_up.weight", i);
		model->hl[i].mlp.c_fc_bias = gguf_tensor_1d(g, E * 4, "blk.%zu.ffn_up.bias", i);

		model->hl[i].mlp.c_proj_weight = gguf_tensor_2d(g, E, E * 4, "blk.%zu.ffn_down.weight", i);
		model->hl[i].mlp.c_proj_bias = gguf_tensor_1d(g, E, "blk.%zu.ffn_down.bias", i);
	}

	model->ln_f_weight = gguf_tensor_1d(g, E, "output_norm.weight");
	model->ln_f_bias = gguf_tensor_1d(g, E, "output_norm.bias");

	model->state.output = tensor_new_zero(2, C, E);
	model->state.tokens = tensor_new_zero(2, C, E);
	model->state.positions = tensor_new_zero(2, C, E);
	model->state.tokens_attn = tensor_new_zero(2, C, E * 3);
	model->state.qh = tensor_new_zero(2, C, HLEN);
	model->state.kh = tensor_new_zero(2, C, HLEN);
	model->state.vh = tensor_new_zero(2, C, HLEN);
	model->state.masked_attn = tensor_new_zero(2, C, C);
	model->state.attn = tensor_new_zero(2, C, E);
	model->state.attn_residual = tensor_new_zero(2, C, E);
	model->state.mlp_fc = tensor_new_zero(2, C, E * 4);
	model->state.logits = tensor_new_zero(1, model->vocab_len);

	model->cache = kvcache_alloc(model->layers, C, H, HLEN);
	assert(model->cache);

	uint64_t totmem = 0;
	totmem += model->state.output->maxcap;
	totmem += model->state.tokens->maxcap;
	totmem += model->state.positions->maxcap;
	totmem += model->state.tokens_attn->maxcap;
	totmem += model->state.qh->maxcap;
	totmem += model->state.kh->maxcap;
	totmem += model->state.vh->maxcap;
	totmem += model->state.masked_attn->maxcap;
	totmem += model->state.attn->maxcap;
	totmem += model->state.attn_residual->maxcap;
	totmem += model->state.mlp_fc->maxcap;
	totmem += model->state.logits->maxcap;

	uint64_t cachemem = 0;
	for (size_t i = 0; i < model->layers; i++) {
		cachemem += model->cache->hl[i].k->maxcap;
		cachemem += model->cache->hl[i].v->maxcap;
	}

	fprintf(stderr, "runtime memory: %luMB + %luMB KV cache\n",
	        totmem / 1024 / 1024,
	        cachemem / 1024 / 1024);

	return model;
}

static void transformer(struct gpt2 *model, tensor_t *input, tensor_t *output, size_t l, enum kv_mode mode)
{
	tensor_t *qh = model->state.qh;
	tensor_t *kh = model->state.kh;
	tensor_t *vh = model->state.vh;
	tensor_t *masked_attn = model->state.masked_attn;

	struct kvcache *cache = model->cache;

	bool decode = cache && mode == KV_DECODE;
	bool prefill = !decode;

	size_t T = tensor_len(input);
	size_t AT = T; /* attention dimensionality for KV cache */
	size_t H = model->heads;
	size_t HLEN = model->head_len;
	size_t E = model->embeddings;

	if (decode) {
		assert(T == 1);
		AT = cache->size + 1;
	}

	tensor_resize(qh, T);
	tensor_resize(kh, AT);
	tensor_resize(vh, AT);
	tensor_resize_2d(masked_attn, AT, AT);

	/*
	 * Tensor views: tensor_at(t, i, &view) returns a zero-copy view
	 * pointing into t's data with one fewer dimension (like numpy's
	 * t[i]). tensor_reshape reinterprets dimensions without copying.
	 * Chaining the two drills into flat row-major data:
	 *
	 *   input                             (T, 3*E)
	 *   tensor_at(input, t_idx, &tok)  -> (3*E)      select token
	 *   tensor_reshape_2d(&tok, 3, E)  -> (3, E)     split Q/K/V
	 *   tensor_at(&tok, 1, &k)         -> (E)        select K
	 *   tensor_reshape_2d(&k, H, HLEN) -> (H, HLEN)  split heads
	 *   tensor_at(&k, h_idx, &k)       -> (HLEN)     select head
	 *
	 * KV cache is (H, C, HLEN) per layer: heads are the outer
	 * dimension so each head's tokens are contiguous. tensor_at on
	 * the cache gives a (C, HLEN) view that BLAS reads directly -
	 * no per-token extraction loop needed.
	 */
	for (size_t h_idx = 0; h_idx < H; h_idx++) {
		tensor_t cache_k, cache_v;
		tensor_t *kh_src = kh;
		tensor_t *vh_src = vh;

		if (decode) {
			tensor_t tok;
			tensor_at(input, 0, &tok);
			tensor_reshape_2d(&tok, 3, E);

			/* Q: extract from the single input token */
			tensor_t q;
			tensor_at(&tok, 0, &q);
			tensor_reshape_2d(&q, H, HLEN);
			tensor_at(&q, h_idx, &q);
			tensor_set_inner(qh, 0, &q);

			/* K: get contiguous (C, HLEN) view, write new token */
			kvcache_get_k(cache, l, h_idx, &cache_k);

			tensor_t k;
			tensor_at(&tok, 1, &k);
			tensor_reshape_2d(&k, H, HLEN);
			tensor_at(&k, h_idx, &k);
			tensor_set_inner(&cache_k, cache->size, &k);

			tensor_resize(&cache_k, AT);
			kh_src = &cache_k;

			/* V: same pattern */
			kvcache_get_v(cache, l, h_idx, &cache_v);

			tensor_t v;
			tensor_at(&tok, 2, &v);
			tensor_reshape_2d(&v, H, HLEN);
			tensor_at(&v, h_idx, &v);
			tensor_set_inner(&cache_v, cache->size, &v);

			tensor_resize(&cache_v, AT);
			vh_src = &cache_v;
		} else {
			/* Prefill path */
			if (cache) {
				kvcache_get_k(cache, l, h_idx, &cache_k);
				kvcache_get_v(cache, l, h_idx, &cache_v);
			}

			for (size_t t_idx = 0; t_idx < T; t_idx++) {
				tensor_t tok;
				tensor_at(input, t_idx, &tok);
				tensor_reshape_2d(&tok, 3, E);

				tensor_t q;
				tensor_at(&tok, 0, &q);
				tensor_reshape_2d(&q, H, HLEN);
				tensor_at(&q, h_idx, &q);
				tensor_set_inner(qh, t_idx, &q);

				tensor_t k;
				tensor_at(&tok, 1, &k);
				tensor_reshape_2d(&k, H, HLEN);
				tensor_at(&k, h_idx, &k);
				if (cache)
					tensor_set_inner(&cache_k, t_idx, &k);
				else
					tensor_set_inner(kh, t_idx, &k);

				tensor_t v;
				tensor_at(&tok, 2, &v);
				tensor_reshape_2d(&v, H, HLEN);
				tensor_at(&v, h_idx, &v);
				if (cache)
					tensor_set_inner(&cache_v, t_idx, &v);
				else
					tensor_set_inner(vh, t_idx, &v);
			}

			if (cache) {
				tensor_resize(&cache_k, AT);
				kh_src = &cache_k;

				tensor_resize(&cache_v, AT);
				vh_src = &cache_v;
			}
		}

		tensor_assert_2d(qh, T, HLEN);
		tensor_assert_2d(kh_src, AT, HLEN);
		tensor_assert_2d(vh_src, AT, HLEN);
		tensor_mma_transposed_2x2(masked_attn, qh, kh_src, NULL);
		tensor_assert_2d(masked_attn, T, AT);
		tensor_div_scalar(masked_attn, masked_attn, model->hlen_sq);

		if (prefill) {
			/* causal mask: when processing multiple tokens at once,
			 * prevent each position from attending to future tokens.
			 * Not needed in decode since T==1 (single query row). */
			for (size_t i = 0; i < AT; i++)
				for (size_t j = 0; j < AT; j++)
					if (j > i)
						masked_attn->data[i * AT + j] = -1.0000e+04;
		}

		softmax_2d(masked_attn);

		tensor_assert_2d(masked_attn, T, AT);
		tensor_assert_2d(vh_src, AT, HLEN);

		tensor_mma_2x2(qh, masked_attn, vh_src, NULL);
		tensor_assert_2d(qh, T, HLEN);

		for (size_t t_idx = prefill ? 0 : AT - 1; t_idx < AT; t_idx++) {
			tensor_t row;
			tensor_at(qh, prefill ? t_idx : 0, &row);
			tensor_assert_1d(&row, HLEN);

			tensor_t attn_tok;
			tensor_at(output, prefill ? t_idx : 0, &attn_tok);
			tensor_reshape_2d(&attn_tok, H, HLEN);
			tensor_set_inner(&attn_tok, h_idx, &row);
		}
	}
}

static void gpt2_eval_inner(struct gpt2 *model, int *tok, int *pos, size_t T, tensor_t *output, enum kv_mode mode)
{
	size_t E = model->embeddings;
	size_t C = model->context;

	tensor_t *tokens = model->state.tokens;
	tensor_t *positions = model->state.positions;
	tensor_t *tokens_attn = model->state.tokens_attn;

	tensor_t *attn = model->state.attn;
	tensor_t *attn_residual = model->state.attn_residual;

	tensor_t *mlp_fc = model->state.mlp_fc;

	/* reuse some memory */
	/* {{{ */
	tensor_t *hidden_state = tokens;
	tensor_t *tmp_mat = positions;
	/* }}} */

	tensor_pick_rows(tokens, model->wte, tok, T);
	tensor_assert_2d(tokens, T, E);

	tensor_pick_rows(positions, model->wpe, pos, T);
	tensor_assert_2d(positions, T, E);

	tensor_add(tokens, tokens, positions);

	tensor_resize(attn, T);
	tensor_resize(attn_residual, T);
	tensor_resize(mlp_fc, T);
	tensor_resize(output, T);

	profiler_record(0, "pick");

	for (size_t l = 0; l < model->layers; l++) {
		struct gpt2h *hl = &model->hl[l];
		struct gpt2attn *a = &hl->attn;
		struct gpt2mlp *mlp = &hl->mlp;

		tensor_copy(tmp_mat, hidden_state);
		layer_norm(output, tmp_mat, hl->ln_1_weight, hl->ln_1_bias);
		profiler_record(1, "ln1");

		tensor_mma_transposed_2x2(tokens_attn, output, a->c_attn_weight, a->c_attn_bias);
		tensor_assert_2d(tokens_attn, T, 3 * E);
		profiler_record(2, "c_attn");

		transformer(model, tokens_attn, attn, l, mode);
		profiler_record(3, "xformer");

		tensor_mma_transposed_2x2(tmp_mat, attn, a->c_proj_weight, a->c_proj_bias);
		profiler_record(4, "c_proj");

		tensor_assert_2d(tmp_mat, T, E);
		tensor_assert_2d(hidden_state, T, E);
		tensor_add(tmp_mat, tmp_mat, hidden_state);
		tensor_copy(attn_residual, tmp_mat);
		profiler_record(5, "c_proj residual");

		layer_norm(output, tmp_mat, hl->ln_2_weight, hl->ln_2_bias);
		tensor_assert_2d(output, T, E);
		profiler_record(6, "ln2");

		tensor_mma_transposed_2x2(mlp_fc, output, mlp->c_fc_weight, mlp->c_fc_bias);
		tensor_assert_2d(mlp_fc, T, E * 4);
		profiler_record(7, "c_fc");

		gelua(mlp_fc);
		profiler_record(8, "gelua");

		tensor_mma_transposed_2x2(hidden_state, mlp_fc, mlp->c_proj_weight, mlp->c_proj_bias);
		profiler_record(9, "c_proj");
		tensor_add(hidden_state, hidden_state, attn_residual);
		profiler_record(10, "c_proj residual");
	}

	layer_norm(output, hidden_state, model->ln_f_weight, model->ln_f_bias);
	profiler_record(11, "ln2 residual");
}

void gpt2_prefill(struct gpt2 *model, int *tok, int *pos, size_t T, tensor_t *output)
{
	gpt2_eval_inner(model, tok, pos, T, output, KV_PREFILL);
	model->cache->size = T;
}

void gpt2_decode(struct gpt2 *model, int tok, int pos, tensor_t *output)
{
	if (model->cache->size >= model->cache->context)
		kvcache_rotate(model->cache);

	/* After rotation pos may exceed the context window; clamp it to
	 * the current cache position so the position embedding stays in
	 * bounds. */
	if (pos >= (int)model->cache->context)
		pos = model->cache->size;

	gpt2_eval_inner(model, &tok, &pos, 1, output, KV_DECODE);
	model->cache->size++;
}

void gpt2_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx)
{
	struct gpt2 *model = ctx;
	int tok_sz;

	size_t E = model->embeddings;
	size_t C = model->context;

	uint64_t total_begin = profiler_now();

	tensor_t *output = model->state.output;
	tensor_t *logits = model->state.logits;

	/* tokenize the entire prompt first */
	int *toks = malloc(C * sizeof(int));
	int *poss = malloc(C * sizeof(int));
	assert(toks && poss);

	int T = 0;
	int tok;
	while ((tok = vocab_decode(model->gguf, text, &tok_sz)) != -1) {
		printf("%.*s", tok_sz, text);
		text += tok_sz;
		assert(T < (int)C);
		toks[T] = tok;
		poss[T] = T;
		T++;
	}
	fflush(stdout);

	/* batch prefill: run the entire prompt in one forward pass */
	uint64_t prefill_begin = profiler_now();
	gpt2_prefill(model, toks, poss, T, output);
	uint64_t prefill_end = profiler_now();

	int pos = T;

	tensor_t last_row;
	tensor_at(output, T - 1, &last_row);
	tensor_assert_1d(logits, model->vocab_len);
	tensor_assert_1d(&last_row, E);
	tensor_assert_2d(model->wte, model->vocab_len, E);
	tensor_reshape_2d(&last_row, 1, E);
	tensor_mma_transposed_2x2(logits, &last_row, model->wte, NULL);
	tensor_reshape_1d(logits, model->vocab_len);
	/* pass raw logits; the sampling callback handles its own softmax */
	tok = f(cb_ctx, logits);

	uint64_t decode_begin = profiler_now();
	uint64_t batch_begin = decode_begin;
	while (num--) {
		gpt2_decode(model, tok, pos, output);
		pos++;

		tensor_at(output, 0, &last_row);
		tensor_assert_1d(logits, model->vocab_len);
		tensor_assert_1d(&last_row, E);
		tensor_assert_2d(model->wte, model->vocab_len, E);
		tensor_reshape_2d(&last_row, 1, E);
		tensor_mma_transposed_2x2(logits, &last_row, model->wte, NULL);
		tensor_reshape_1d(logits, model->vocab_len);
		/* pass raw logits; the sampling callback handles its own softmax */
		tok = f(cb_ctx, logits);

		if (pos && pos % 100 == 0) {
			uint64_t end = profiler_now();
			fprintf(stderr, "[%d tokens, %.9f tok/sec]", pos, (100/profiler_to_sec(end-batch_begin)));
			batch_begin = end;
		}
	}
	uint64_t decode_end = profiler_now();
	fprintf(stderr, "\n");

	fprintf(stderr, "prefill=%fs (%d tokens) decode=%fs total=%fs\n",
	        profiler_to_sec(prefill_end - prefill_begin), T,
	        profiler_to_sec(decode_end - decode_begin),
	        profiler_to_sec(decode_end - total_begin));

	free(toks);
	free(poss);
}

void gpt2_close(void *ctx)
{
	struct gpt2 *model = ctx;

	tensor_free_mapped(model->wte);
	tensor_free_mapped(model->wpe);
	for (size_t i = 0; i < model->layers; i++) {
		tensor_free_mapped(model->hl[i].ln_1_weight);
		tensor_free_mapped(model->hl[i].ln_1_bias);
		tensor_free_mapped(model->hl[i].attn.c_attn_weight);
		tensor_free_mapped(model->hl[i].attn.c_attn_bias);
		tensor_free_mapped(model->hl[i].attn.c_proj_weight);
		tensor_free_mapped(model->hl[i].attn.c_proj_bias);
		tensor_free_mapped(model->hl[i].ln_2_weight);
		tensor_free_mapped(model->hl[i].ln_2_bias);
		tensor_free_mapped(model->hl[i].mlp.c_fc_weight);
		tensor_free_mapped(model->hl[i].mlp.c_fc_bias);
		tensor_free_mapped(model->hl[i].mlp.c_proj_weight);
		tensor_free_mapped(model->hl[i].mlp.c_proj_bias);
	}
	free(model->hl);
	tensor_free_mapped(model->ln_f_weight);
	tensor_free_mapped(model->ln_f_bias);

	tensor_free(model->state.output);
	tensor_free(model->state.tokens);
	tensor_free(model->state.positions);
	tensor_free(model->state.tokens_attn);
	tensor_free(model->state.qh);
	tensor_free(model->state.kh);
	tensor_free(model->state.vh);
	tensor_free(model->state.masked_attn);
	tensor_free(model->state.attn);
	tensor_free(model->state.attn_residual);
	tensor_free(model->state.mlp_fc);
	tensor_free(model->state.logits);

	kvcache_free(model->cache);

	free(model);
}

static const struct model gpt2_model = {
	.name = "gpt2",
	.load = gpt2_load,
	.generate = gpt2_generate,
	.close = gpt2_close,
};

__attribute__((constructor))
static void gpt2_register(void)
{
	register_model(&gpt2_model);
}
