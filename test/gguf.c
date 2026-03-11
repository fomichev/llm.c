#include "gguf.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

static void test_gguf_load(const char *path)
{
	struct gguf *g = gguf_load(path);
	assert(g);

	printf("tensors: %zu\n", gguf_tensor_count(g));
	printf("metadata: %zu\n", gguf_metadata_count(g));

	/* verify metadata */
	const char *arch = gguf_get_str(g, "general.architecture");
	assert(arch);
	assert(strcmp(arch, "gpt2") == 0);
	printf("arch: %s\n", arch);

	uint32_t ctx_len = gguf_get_uint32(g, "gpt2.context_length");
	uint32_t emb_len = gguf_get_uint32(g, "gpt2.embedding_length");
	uint32_t blocks = gguf_get_uint32(g, "gpt2.block_count");
	uint32_t heads = gguf_get_uint32(g, "gpt2.attention.head_count");

	printf("context_length: %u\n", ctx_len);
	printf("embedding_length: %u\n", emb_len);
	printf("block_count: %u\n", blocks);
	printf("head_count: %u\n", heads);

	assert(ctx_len == 1024);
	assert(emb_len == 768);
	assert(blocks == 12);
	assert(heads == 12);

	/* verify we have the right number of tensors:
	 * 2 (wte, wpe) + 12 per layer * 12 layers + 2 (ln_f) = 148 */
	assert(gguf_tensor_count(g) == 148);

	size_t V = 50257;
	size_t E = emb_len;
	size_t C = ctx_len;

	/* load a few tensors and verify shapes */
	tensor_t *wte = gguf_tensor(g, "token_embd.weight", 2, V, E);
	assert(wte);
	assert(wte->ndim == 2);
	assert(wte->dim[0] == V);
	assert(wte->dim[1] == E);
	printf("token_embd.weight: %lux%lu\n", wte->dim[0], wte->dim[1]);

	tensor_t *wpe = gguf_tensor(g, "position_embd.weight", 2, C, E);
	assert(wpe);
	assert(wpe->dim[0] == C);
	assert(wpe->dim[1] == E);
	printf("position_embd.weight: %lux%lu\n", wpe->dim[0], wpe->dim[1]);

	/* check layer 0 tensors */
	tensor_t *ln1_w = gguf_tensor(g, "blk.0.attn_norm.weight", 1, E);
	assert(ln1_w);
	assert(ln1_w->dim[0] == E);

	tensor_t *qkv_w = gguf_tensor(g, "blk.0.attn_qkv.weight", 2, E, E * 3);
	assert(qkv_w);
	assert(qkv_w->dim[0] == E);
	assert(qkv_w->dim[1] == E * 3);
	printf("blk.0.attn_qkv.weight: %lux%lu\n", qkv_w->dim[0], qkv_w->dim[1]);

	tensor_t *fc_w = gguf_tensor(g, "blk.0.ffn_up.weight", 2, E, E * 4);
	assert(fc_w);
	assert(fc_w->dim[0] == E);
	assert(fc_w->dim[1] == E * 4);

	tensor_t *proj_w = gguf_tensor(g, "blk.0.ffn_down.weight", 2, E * 4, E);
	assert(proj_w);
	assert(proj_w->dim[0] == E * 4);
	assert(proj_w->dim[1] == E);

	tensor_t *ln_f_w = gguf_tensor(g, "output_norm.weight", 1, E);
	assert(ln_f_w);
	assert(ln_f_w->dim[0] == E);

	tensor_t *ln_f_b = gguf_tensor(g, "output_norm.bias", 1, E);
	assert(ln_f_b);
	assert(ln_f_b->dim[0] == E);

	/* verify data is not all zeros (sanity check) */
	int nonzero = 0;
	for (size_t i = 0; i < 100 && i < wte->totlen; i++)
		if (wte->data[i] != 0.0f)
			nonzero++;
	assert(nonzero > 0);
	printf("wte first few values: %f %f %f\n",
	       wte->data[0], wte->data[1], wte->data[2]);

	/* verify all 12 layers can be loaded */
	for (int i = 0; i < 12; i++) {
		char name[64];

		snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
		assert(gguf_tensor(g, name, 1, E));

		snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
		assert(gguf_tensor(g, name, 2, E, E * 3));

		snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
		assert(gguf_tensor(g, name, 2, E * 4, E));
	}

	/* test missing tensor returns NULL */
	tensor_t *missing = gguf_tensor(g, "nonexistent.weight", 1, (size_t)1);
	assert(missing == NULL);

	printf("gguf: all tests passed\n");

	/* cleanup */
	tensor_free_mapped(wte);
	tensor_free_mapped(wpe);
	tensor_free_mapped(ln1_w);
	tensor_free_mapped(qkv_w);
	tensor_free_mapped(fc_w);
	tensor_free_mapped(proj_w);
	tensor_free_mapped(ln_f_w);
	tensor_free_mapped(ln_f_b);

	gguf_close(g);
}

int main(int argc, char *argv[])
{
	const char *path = "gpt2_124M.gguf";

	if (argc > 1)
		path = argv[1];

	test_gguf_load(path);
}
