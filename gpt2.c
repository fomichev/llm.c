#include "llm.h"
#include "profiler.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct gpt2 {
	struct snapshot *ss;

	struct {
		size_t context; /* context size */
		size_t heads; /* number of heads */
		size_t head_len; /* head size */
		size_t layers; /* number of hidden layers */
		size_t embeddings; /* embeddings size */
		size_t vocab_len; /* vocabulary size */
		bool transposed; /* whether the weights were pre-transposed */
	};

	FT_TYPE hlen_sq;

	const ft_t *wte;
	const ft_t *wpe;
	struct gpt2h {
		const ft_t *ln_1_weight;
		const ft_t *ln_1_bias;

		struct gpt2attn {
			const ft_t *bias;
			const ft_t *masked_bias;

			const ft_t *c_attn_weight;
			const ft_t *c_attn_bias;

			const ft_t *c_proj_weight;
			const ft_t *c_proj_bias;
		} attn;

		const ft_t *ln_2_weight;
		const ft_t *ln_2_bias;

		struct gpt2mlp {
			const ft_t *c_fc_weight;
			const ft_t *c_fc_bias;
			const ft_t *c_proj_weight;
			const ft_t *c_proj_bias;
		} mlp;
	} hl[48]; /* to fit gpt2-xl */
	const ft_t *ln_f_weight;
	const ft_t *ln_f_bias;

	struct {
		ft_t *output;
		ft_t *tokens;
		ft_t *positions;
		ft_t *tokens_attn;
		ft_t *qh; /* heads X from all tokens */
		ft_t *kh; /* heads X from all tokens */
		ft_t *vh; /* heads X from all tokens */
		ft_t *masked_attn;
		ft_t *attn;
		ft_t *attn_residual;
		ft_t *mlp_fc;
		ft_t *logits;
	} state;

	struct {
		struct {
			ft_t *k;
			ft_t *v;
		} hl[48];
		size_t size;
		bool use;
	} cache;
};

struct gpt2 *gpt2_load(struct snapshot *ss)
{
	struct gpt2 *model;
	struct file *f;
	size_t idx = 0;

	model = calloc(1, sizeof(*model));
	if (!model)
		return NULL;

	model->ss = ss;
	f = snapshot_param(ss);

	model->context = snapshot_config_int(ss, "context"); /* 1024 */
	model->head_len = snapshot_config_int(ss, "head_len"); /* 64 */
	model->heads = snapshot_config_int(ss, "heads"); /* 12 */
	model->layers = snapshot_config_int(ss, "layers"); /* 12 */
	model->embeddings = snapshot_config_int(ss, "embeddings"); /* 768 */
	model->vocab_len = snapshot_config_int(ss, "vocab_len"); /* 50257 */
	model->transposed = snapshot_config_int(ss, "transposed"); /* false */

	size_t C = model->context;
	size_t HLEN = model->head_len;
	size_t H = model->heads;
	size_t E = model->embeddings;

	model->hlen_sq = sqrt((FT_TYPE)HLEN);

	assert(H * HLEN == E);

	model->wte = file_ft(f, 2, model->vocab_len, E);
	model->wpe = file_ft(f, 2, C, E);

	for (size_t i = 0; i < model->layers; i++) {
		model->hl[i].ln_1_weight = file_ft(f, 1, E);
		model->hl[i].ln_1_bias = file_ft(f, 1, E);
		model->hl[i].attn.bias = file_ft(f, 4, 1, 1, C, C);
		model->hl[i].attn.masked_bias = file_ft(f, 1, 1);
		if (model->transposed)
			model->hl[i].attn.c_attn_weight = file_ft(f, 2, E * 3, E);
		else
			model->hl[i].attn.c_attn_weight = file_ft(f, 2, E, E * 3);
		model->hl[i].attn.c_attn_bias = file_ft(f, 1, E * 3);
		if (model->transposed)
			model->hl[i].attn.c_proj_weight = file_ft(f, 2, E, E);
		else
			model->hl[i].attn.c_proj_weight = file_ft(f, 2, E, E);
		model->hl[i].attn.c_proj_bias = file_ft(f, 1, E);
		model->hl[i].ln_2_weight = file_ft(f, 1, E);
		model->hl[i].ln_2_bias = file_ft(f, 1, E);
		if (model->transposed)
			model->hl[i].mlp.c_fc_weight = file_ft(f, 2, E * 4, E);
		else
			model->hl[i].mlp.c_fc_weight = file_ft(f, 2, E, E * 4);
		model->hl[i].mlp.c_fc_bias = file_ft(f, 1, E * 4);
		if (model->transposed)
			model->hl[i].mlp.c_proj_weight = file_ft(f, 2, E, E * 4);
		else
			model->hl[i].mlp.c_proj_weight = file_ft(f, 2, E * 4, E);
		model->hl[i].mlp.c_proj_bias = file_ft(f, 1, E);
	}

	model->ln_f_weight = file_ft(f, 1, E);
	model->ln_f_bias = file_ft(f, 1, E);

	assert(file_is_eof(f));

	model->state.output = ft_new_zero(2, C, E);
	model->state.tokens = ft_new_zero(2, C, E);
	model->state.positions = ft_new_zero(2, C, E);
	model->state.tokens_attn = ft_new_zero(2, C, E * 3);
	model->state.qh = ft_new_zero(2, C, HLEN);
	model->state.kh = ft_new_zero(2, C, HLEN);
	model->state.vh = ft_new_zero(2, C, HLEN);
	model->state.masked_attn = ft_new_zero(2, C, C);
	model->state.attn = ft_new_zero(2, C, E);
	model->state.attn_residual = ft_new_zero(2, C, E);
	model->state.mlp_fc = ft_new_zero(2, C, E * 4);
	model->state.logits = ft_new_zero(1, model->vocab_len);

	for (size_t i = 0; i < model->layers; i++) {
		model->cache.hl[i].k = ft_new_zero(2, C, H * HLEN);
		model->cache.hl[i].v = ft_new_zero(2, C, H * HLEN);
		model->cache.use = true;
		model->cache.size = 0;
	}

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
		cachemem += model->cache.hl[i].k->maxcap;
		cachemem += model->cache.hl[i].v->maxcap;
	}

	printf("runtime memory: %luMB + %luMB KV cache\n",
	       totmem / 1024 / 1024,
	       cachemem / 1024 / 1024);

	return model;
}

static void layer_norm(
	ft_t *ln,
	ft_t *tmp_mat,
	const ft_t *weight,
	const ft_t *bias)
{
	for (size_t i = 0; i < ft_len(tmp_mat); i++) {
		ft_t row;

		ft_at(tmp_mat, i, &row);

		FT_TYPE row_mean = ft_mean(&row);

		fv_t s, e;

		fv_load1(&s, 0);
		fv_load1(&e, row_mean);

		size_t len = ft_len(&row);

		for (size_t j = 0; j < FT_LEN(len); j += FT_N) {
			fv_t tmp;

			fv_load(&tmp, &row.data[j]);
			fv_sub(&tmp, &tmp, &e);
			fv_mul(&tmp, &tmp, &tmp);
			fv_add(&s, &s, &tmp);
		}
		FT_TYPE sum = fv_reduce_sum(&s);
		for (size_t j = FT_LEN(len); j < len; j++) {
			FT_TYPE tmp = row.data[j] - row_mean;
			sum += tmp * tmp;
		}

		FT_TYPE var = sum / len;
		FT_TYPE var_sqrt = sqrtf(var + 1e-5);

		fv_t vsqrt;
		fv_load1(&vsqrt, var_sqrt);

		for (size_t j = 0; j < FT_LEN(len); j += FT_N) {
			fv_t tmp;

			fv_load(&tmp, &row.data[j]);
			fv_sub(&tmp, &tmp, &e);
			fv_div(&tmp, &tmp, &vsqrt);
			fv_store(&row.data[j], &tmp);
		}
		for (size_t j = FT_LEN(len); j < len; j++) {
			row.data[j] = (row.data[j] - row_mean) / var_sqrt;
		}

		ft_t ln_row;
		ft_at(ln, i, &ln_row);

		ft_mul(&ln_row, &row, weight);
		ft_add(&ln_row, &ln_row, bias);
	}
}

#define GELU_K1 0.7978845608028654 /* (sqrt(2.0 / M_PI)) */
#define GELU_K2 0.044715

static void gelua(ft_t *t)
{
	assert(t->totlen % FT_N == 0);

	fv_t vinp;
	fv_t va;

	fv_t k1;
	fv_load1(&k1, GELU_K1);

	fv_t k2;
	fv_load1(&k2, GELU_K2);

	fv_t one;
	fv_load1(&one, 1.0);

	fv_t half;
	fv_load1(&half, 0.5);

	for (size_t i = 0; i < FT_LEN(t->totlen); i += FT_N) {
		fv_load(&vinp, &t->data[i]);

		/* 1.0 + GELU_K2 * inp * inp */
		fv_mul(&va, &vinp, &vinp);
		fv_mul(&va, &va, &k2);
		fv_add(&va, &va, &one);

		/* tanh() */
		fv_mul(&va, &va, &vinp);
		fv_mul(&va, &va, &k1);
		fv_tanh(&va, &va);

		/* 1.0 + tanh() */
		fv_add(&va, &va, &one);

		/* 0.5 * (1.0 + tanh()) */
		fv_mul(&va, &va, &half);

		/* inp * 0.5 * (1.0 * tanh()) */
		fv_mul(&va, &va, &vinp);

		fv_store(&t->data[i], &va);
	}

	for (size_t i = FT_LEN(t->totlen); i < t->totlen; i++) {
		FT_TYPE inp;

		inp = t->data[i];
		t->data[i] = 0.5 * inp * (1.0 + tanhf(GELU_K1 * inp * (1.0 + GELU_K2 * inp * inp)));
	}
}

static void softmax_1d(ft_t *t)
{
	size_t len = ft_len(t);
	fv_t vsum, vmax;
	FT_TYPE max;

	assert(t->ndim == 1);

	/* https://discuss.pytorch.org/t/how-to-implement-the-exactly-same-softmax-as-f-softmax-by-pytorch/44263/2 */

	max = ft_max(t, NULL);

	fv_load1(&vsum, 0);
	fv_load1(&vmax, max);

	for (size_t i = 0; i < FT_LEN(len); i += FT_N) {
		fv_t vtmp;

		fv_load(&vtmp, &t->data[i]);
		fv_sub(&vtmp, &vtmp, &vmax);
		fv_exp(&vtmp, &vtmp);
		fv_store(&t->data[i], &vtmp);
		fv_add(&vsum, &vsum, &vtmp);
	}
	FT_TYPE sum = fv_reduce_sum(&vsum);
	for (size_t i = FT_LEN(len); i < len; i++) {
		t->data[i] = expf(t->data[i] - max);
		sum += t->data[i];
	}

	fv_load1(&vsum, sum);
	for (size_t i = 0; i < FT_LEN(len); i += FT_N) {
		fv_t tmp;

		fv_load(&tmp, &t->data[i]);
		fv_div(&tmp, &tmp, &vsum);
		fv_store(&t->data[i], &tmp);
	}
	for (size_t i = FT_LEN(len); i < len; i++) {
		t->data[i] = t->data[i] / sum;
	}
}

static void softmax_2d(ft_t *t)
{
	assert(t->ndim == 2);

	for (size_t i = 0; i < ft_len(t); i++) {
		ft_t row;

		ft_at(t, i, &row);
		softmax_1d(&row);
	}
}

static void transformer(struct gpt2 *model, ft_t *input, ft_t *output, size_t l)
{
	ft_t *qh = model->state.qh;
	ft_t *kh = model->state.kh;
	ft_t *vh = model->state.vh;
	ft_t *masked_attn = model->state.masked_attn;

	size_t C = model->context;
	size_t T = ft_len(input);
	size_t AT = T; /* attention dimensionality for KV cache */
	size_t H = model->heads;
	size_t HLEN = model->head_len;
	size_t E = model->embeddings;

	if (model->cache.use) {
		assert(T == 1);
		AT = model->cache.size + 1;
	}

	ft_resize(qh, T);
	ft_resize(kh, AT);
	ft_resize(vh, AT);
	ft_resize_2d(masked_attn, AT, AT);

	for (size_t h_idx = 0; h_idx < H; h_idx++) {
		if (model->cache.use) {
			ft_t tok, q;
			ft_at(input, 0, &tok);
			ft_reshape_2d(&tok, 3, E);
			ft_at(&tok, 0, &q);
			ft_reshape_2d(&q, H, HLEN);
			ft_at(&q, h_idx, &q);
			ft_set_inner(qh, 0, &q);
		}

		for (size_t t_idx = 0; t_idx < AT; t_idx++) {
			ft_t tok;
			ft_at(input, model->cache.use ? 0 : t_idx, &tok);
			ft_assert_1d(&tok, 3 * E);
			ft_reshape_2d(&tok, 3, E);

			if (!model->cache.use) {
				ft_t q;
				ft_at(&tok, 0, &q);
				ft_reshape_2d(&q, H, HLEN);
				ft_at(&q, h_idx, &q);
				ft_set_inner(qh, t_idx, &q);
			}

			ft_t k;
			if (model->cache.use && t_idx + 1 != AT) {
				ft_at(model->cache.hl[l].k, t_idx, &k);
				ft_reshape_2d(&k, H, HLEN);
				ft_at(&k, h_idx, &k);
			} else {
				ft_at(&tok, 1, &k);
				ft_reshape_2d(&k, H, HLEN);
				ft_at(&k, h_idx, &k);

				if (model->cache.use) {
					ft_t ck;
					assert(t_idx < C); /* TODO: handle overflow */
					ft_at(model->cache.hl[l].k, t_idx, &ck);
					ft_reshape_2d(&ck, H, HLEN);
					ft_set_inner(&ck, h_idx, &k);
				}
			}
			ft_set_inner(kh, t_idx, &k);

			ft_t v;
			if (model->cache.use && t_idx + 1 != AT) {
				ft_at(model->cache.hl[l].v, t_idx, &v);
				ft_reshape_2d(&v, H, HLEN);
				ft_at(&v, h_idx, &v);
				ft_assert_1d(&v, HLEN);
			} else {
				ft_at(&tok, 2, &v);
				ft_reshape_2d(&v, H, HLEN);
				ft_at(&v, h_idx, &v);
				ft_assert_1d(&v, HLEN);

				if (model->cache.use) {
					ft_t cv;
					assert(t_idx < C); /* TODO: handle overflow */
					ft_at(model->cache.hl[l].v, t_idx, &cv);
					ft_reshape_2d(&cv, H, HLEN);
					ft_set_inner(&cv, h_idx, &v);
				}
			}
			ft_set_inner(vh, t_idx, &v);
		}

		ft_assert_2d(qh, T, HLEN);
		ft_assert_2d(kh, AT, HLEN);
		ft_assert_2d(vh, AT, HLEN);
		ft_mma_transposed_2x2(masked_attn, qh, kh, NULL);
		ft_assert_2d(masked_attn, T, AT);
		ft_div_scalar(masked_attn, masked_attn, model->hlen_sq);

		if (!model->cache.use) {
			/* attention mask */
			for (size_t i = 0; i < AT; i++)
				for (size_t j = 0; j < AT; j++)
					if (j > i)
						masked_attn->data[i * AT + j] = -1.0000e+04;
		}

		softmax_2d(masked_attn);

		ft_assert_2d(masked_attn, T, AT);
		ft_assert_2d(vh, AT, HLEN);

		ft_mma_2x2(qh, masked_attn, vh, NULL);
		ft_assert_2d(qh, T, HLEN);

		for (size_t t_idx = model->cache.use ? AT - 1 : 0; t_idx < AT; t_idx++) {
			ft_t row;
			ft_at(qh, model->cache.use ? 0 : t_idx, &row);
			ft_assert_1d(&row, HLEN);

			ft_t attn_tok;
			ft_at(output, model->cache.use ? 0 : t_idx, &attn_tok);
			ft_reshape_2d(&attn_tok, H, HLEN);
			ft_set_inner(&attn_tok, h_idx, &row);
		}
	}
}

static void __gpt2_eval(struct gpt2 *model, int *tok, int *pos, size_t T, ft_t *output)
{
	size_t E = model->embeddings;
	size_t C = model->context;

	ft_t *tokens = model->state.tokens;
	ft_t *positions = model->state.positions;
	ft_t *tokens_attn = model->state.tokens_attn;

	ft_t *attn = model->state.attn;
	ft_t *attn_residual = model->state.attn_residual;

	ft_t *mlp_fc = model->state.mlp_fc;

	/* reuse some memory */
	/* {{{ */
	ft_t *hidden_state = tokens;
	ft_t *tmp_mat = positions;
	/* }}} */

	ft_pick_rows(tokens, model->wte, tok, T);
	ft_assert_2d(tokens, T, E);

	ft_pick_rows(positions, model->wpe, pos, T);
	ft_assert_2d(positions, T, E);

	ft_add(tokens, tokens, positions);

	ft_resize(attn, T);
	ft_resize(attn_residual, T);
	ft_resize(mlp_fc, T);
	ft_resize(output, T);

	profiler_record(0, "pick");

	for (size_t l = 0; l < model->layers; l++) {
		struct gpt2h *hl = &model->hl[l];
		struct gpt2attn *a = &hl->attn;
		struct gpt2mlp *mlp = &hl->mlp;

		ft_copy(tmp_mat, hidden_state);
		layer_norm(output, tmp_mat, hl->ln_1_weight, hl->ln_1_bias);
		profiler_record(1, "ln1");

		if (model->transposed)
			ft_mma_transposed_2x2(tokens_attn, output, a->c_attn_weight, a->c_attn_bias);
		else
			ft_mma_2x2(tokens_attn, output, a->c_attn_weight, a->c_attn_bias);
		ft_assert_2d(tokens_attn, T, 3 * E);
		profiler_record(2, "c_attn");

		transformer(model, tokens_attn, attn, l);
		profiler_record(3, "xformer");

		if (model->transposed)
			ft_mma_transposed_2x2(tmp_mat, attn, a->c_proj_weight, a->c_proj_bias);
		else
			ft_mma_2x2(tmp_mat, attn, a->c_proj_weight, a->c_proj_bias);
		profiler_record(4, "c_proj");

		ft_assert_2d(tmp_mat, T, E);
		ft_assert_2d(hidden_state, T, E);
		ft_add(tmp_mat, tmp_mat, hidden_state);
		ft_copy(attn_residual, tmp_mat);
		profiler_record(5, "c_proj residual");

		layer_norm(output, tmp_mat, hl->ln_2_weight, hl->ln_2_bias);
		ft_assert_2d(output, T, E);
		profiler_record(6, "ln2");

		if (model->transposed)
			ft_mma_transposed_2x2(mlp_fc, output, mlp->c_fc_weight, mlp->c_fc_bias);
		else
			ft_mma_2x2(mlp_fc, output, mlp->c_fc_weight, mlp->c_fc_bias);
		ft_assert_2d(mlp_fc, T, E * 4);
		profiler_record(7, "c_fc");

		gelua(mlp_fc);
		profiler_record(8, "gelua");

		if (model->transposed)
			ft_mma_transposed_2x2(hidden_state, mlp_fc, mlp->c_proj_weight, mlp->c_proj_bias);
		else
			ft_mma_2x2(hidden_state, mlp_fc, mlp->c_proj_weight, mlp->c_proj_bias);
		profiler_record(9, "c_proj");
		ft_add(hidden_state, hidden_state, attn_residual);
		profiler_record(10, "c_proj residual");
	}

	layer_norm(output, hidden_state, model->ln_f_weight, model->ln_f_bias);
	profiler_record(11, "ln2 residual");
}

static void gpt2_eval(struct gpt2 *model, int tok, int pos, ft_t *output)
{
	assert(model->cache.use);

	__gpt2_eval(model, &tok, &pos, 1, output);
	model->cache.size++;
}

static void gpt2_use_cache(struct gpt2 *model, bool use)
{
	model->cache.use = use;
	model->cache.size = 0;
}

void gpt2_test_no_cache(struct gpt2 *model)
{
	int *tok, *pos;

	size_t E = model->embeddings;

	int inp[] = { 818, 262, 3329, 314, 373, 1498, 284 };
	int exp[] = { 818, 262, 3329, 314, 373, 1498, 284, 651, 257, 922, 804, 379, 262, 2615, 290, 262, 2615 };
	int num = GPT2_EVAL_ROUNDS;

	tok = malloc(model->context * sizeof(int));
	pos = malloc(model->context * sizeof(int));
	assert(tok && pos);

	int T = ARRAY_SIZE(inp);

	memcpy(tok, inp, sizeof(inp));
	for (int i = 0; i < T; i++)
		pos[i] = i;

	ft_t *output = model->state.output;
	ft_t *logits = model->state.logits;

	gpt2_use_cache(model, false);

	profiler_start();
	while (num--) {
		__gpt2_eval(model, tok, pos, T, output);

		ft_t last_row;
		ft_at(output, T - 1, &last_row);

		size_t nextch = 0;
		ft_assert_1d(logits, model->vocab_len);
		ft_assert_1d(&last_row, E);
		ft_assert_2d(model->wte, model->vocab_len, E);
		ft_reshape_2d(&last_row, 1, E);
		ft_mma_transposed_2x2(logits, &last_row, model->wte, NULL);
		profiler_record(12, "wte");
		ft_reshape_1d(logits, model->vocab_len);
		softmax_1d(logits);
		FT_TYPE mx = ft_max(logits, &nextch);
		profiler_record(13, "max");

		tok[T] = nextch;
		pos[T] = T;
		T++;
	}
	profiler_report();

	if (E == 768 && GPT2_EVAL_ROUNDS == 10) {
		assert(memcmp(tok, exp, sizeof(int) * T) == 0);
		printf("verified output\n");
	}

	free(tok);
	free(pos);
}

void gpt2_test_cache(struct gpt2 *model)
{
	size_t E = model->embeddings;

	int inp[] = { 818, 262, 3329, 314, 373, 1498, 284 };
	int exp[] = { 651, 257, 922, 804, 379, 262, 2615, 290, 262, 2615, 2346 };
	int got[ARRAY_SIZE(exp)] = {};
	int num = GPT2_EVAL_ROUNDS + ARRAY_SIZE(inp);

	int tok = 0;
	int pos = 0;

	ft_t *output = model->state.output;
	ft_t *logits = model->state.logits;

	gpt2_use_cache(model, true);

	size_t nextch = 0;

	profiler_start();
	for (int i = 0; i < num; i++) {
		if (i < ARRAY_SIZE(inp))
			tok = inp[i];
		else
			tok = nextch;

		gpt2_eval(model, tok, pos, output);

		ft_t last_row;
		ft_at(output, 0, &last_row);

		ft_assert_1d(logits, model->vocab_len);
		ft_assert_1d(&last_row, E);
		ft_assert_2d(model->wte, model->vocab_len, E);
		ft_reshape_2d(&last_row, 1, E);
		ft_mma_transposed_2x2(logits, &last_row, model->wte, NULL);
		profiler_record(12, "wte");
		ft_reshape_1d(logits, model->vocab_len);
		softmax_1d(logits);
		FT_TYPE mx = ft_max(logits, &nextch);
		profiler_record(13, "max");

		if (i >= ARRAY_SIZE(inp) - 1)
			got[i - ARRAY_SIZE(inp) + 1] = nextch;

		pos++;
	}
	profiler_report();

	if (E == 768 && GPT2_EVAL_ROUNDS == 10) {
		assert(memcmp(got, exp, sizeof(int) * ARRAY_SIZE(got)) == 0);
		printf("verified output\n");
	}
}

void gpt2_generate(struct gpt2 *model, const char *text, int num, pick_token_t f, void *ctx)
{
	struct file *vocab;
	int tok_sz;

	size_t E = model->embeddings;
	size_t C = model->context;

	vocab = snapshot_vocab(model->ss);
	assert(vocab);

	int tok;
	int pos = 0;

	uint64_t total_begin = profiler_now();

	ft_t *output = model->state.output;
	ft_t *logits = model->state.logits;

	while ((tok = vocab_decode(vocab, text, &tok_sz)) != -1) {
		printf("%.*s", tok_sz, text);

		text += tok_sz;

		gpt2_eval(model, tok, pos, output);
		pos++;
	}

	size_t nextch = 0;
	ft_t last_row;

	ft_at(output, 0, &last_row);
	ft_assert_1d(logits, model->vocab_len);
	ft_assert_1d(&last_row, E);
	ft_assert_2d(model->wte, model->vocab_len, E);
	ft_reshape_2d(&last_row, 1, E);
	ft_mma_transposed_2x2(logits, &last_row, model->wte, NULL);
	ft_reshape_1d(logits, model->vocab_len);
	softmax_1d(logits);
	tok = f(ctx, logits);

	uint64_t begin = profiler_now();
	while (num--) {
		gpt2_eval(model, tok, pos, output);
		pos++;

		ft_at(output, 0, &last_row);
		ft_assert_1d(logits, model->vocab_len);
		ft_assert_1d(&last_row, E);
		ft_assert_2d(model->wte, model->vocab_len, E);
		ft_reshape_2d(&last_row, 1, E);
		ft_mma_transposed_2x2(logits, &last_row, model->wte, NULL);
		ft_reshape_1d(logits, model->vocab_len);
		softmax_1d(logits);
		tok = f(ctx, logits);

		if (pos && pos % 100 == 0) {
			uint64_t end = profiler_now();
			fprintf(stderr, "[%d tokens, %.9f sec/tok]", pos, (profiler_to_sec(end-begin))/100);
			begin = end;
		}
	}
	uint64_t total_end = profiler_now();
	printf("\n");

	printf("total=%fs\n", profiler_to_sec(total_end - total_begin));
}

void gpt2_close(struct gpt2 *model)
{
	ft_free_mapped(model->wte);
	ft_free_mapped(model->wpe);
	for (size_t i = 0; i < model->layers; i++) {
		ft_free_mapped(model->hl[i].ln_1_weight);
		ft_free_mapped(model->hl[i].ln_1_bias);
		ft_free_mapped(model->hl[i].attn.bias);
		ft_free_mapped(model->hl[i].attn.masked_bias);
		ft_free_mapped(model->hl[i].attn.c_attn_weight);
		ft_free_mapped(model->hl[i].attn.c_attn_bias);
		ft_free_mapped(model->hl[i].attn.c_proj_weight);
		ft_free_mapped(model->hl[i].attn.c_proj_bias);
		ft_free_mapped(model->hl[i].ln_2_weight);
		ft_free_mapped(model->hl[i].ln_2_bias);
		ft_free_mapped(model->hl[i].mlp.c_fc_weight);
		ft_free_mapped(model->hl[i].mlp.c_fc_bias);
		ft_free_mapped(model->hl[i].mlp.c_proj_weight);
		ft_free_mapped(model->hl[i].mlp.c_proj_bias);
	}
	ft_free_mapped(model->ln_f_weight);
	ft_free_mapped(model->ln_f_bias);

	ft_free(model->state.output);
	ft_free(model->state.tokens);
	ft_free(model->state.positions);
	ft_free(model->state.tokens_attn);
	ft_free(model->state.qh);
	ft_free(model->state.kh);
	ft_free(model->state.vh);
	ft_free(model->state.masked_attn);
	ft_free(model->state.attn);
	ft_free(model->state.attn_residual);
	ft_free(model->state.mlp_fc);
	ft_free(model->state.logits);

	file_close(snapshot_param(model->ss));
	file_close(snapshot_vocab(model->ss));
}
