#include "llm.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <execinfo.h>
#include <time.h>

static size_t pick_greedy(tensor_t *logits)
{
	size_t token;

	tensor_max(logits, &token);
	return token;
}

static void top_k(tensor_t *f, size_t *top_n, scalar_t *top_v, size_t k)
{
	assert(k <= f->totlen);

	for (size_t i = 0; i < k; i++) {
		top_n[i] = 0;
		top_v[i] = f->data[0];
	}

	for (size_t i = 1; i < f->totlen; i++) {
		scalar_t new_v = f->data[i];
		int new_p = -1;

		for (size_t j = 0; j < k; j++) {
			if (new_v > top_v[j])
				new_p = j;
		}

		if (new_p < 0)
			continue;

		for (size_t j = 0; j < k; j++) {
			if (j < new_p) {
				top_n[j] = top_n[j+1];
				top_v[j] = top_v[j+1];
			} else if (j == new_p) {
				top_n[j] = i;
				top_v[j] = new_v;
				break;
			}
		}
	}
}

static void top_k_test(void)
{
	tensor_t *x = tensor_new_1d(10,
			    0.0,  /* 0 */
			    1.0,  /* 1 */
			    7.0,  /* 2 */
			    2.0,  /* 3 */
			    9.0,  /* 4 */
			    3.0,  /* 5 */
			    6.0,  /* 6 */
			    4.0,  /* 7 */
			    5.0,  /* 8 */
			    8.0   /* 9 */);

	size_t top_n[5];
	scalar_t top_v[5];

	top_k(x, &top_n[0], &top_v[0], 5);
	assert(top_n[0] == 8);
	assert(top_n[1] == 6);
	assert(top_n[2] == 2);
	assert(top_n[3] == 9);
	assert(top_n[4] == 4);
	assert(top_v[0] == 5.0);
	assert(top_v[1] == 6.0);
	assert(top_v[2] == 7.0);
	assert(top_v[3] == 8.0);
	assert(top_v[4] == 9.0);
}

static size_t pick_top_k(tensor_t *logits, size_t k)
{
	size_t top_n[k];
	scalar_t top_v[k];

	top_k(logits, &top_n[0], &top_v[0], k);

	scalar_t max = top_v[0];
	for (size_t i = 0; i < k; i++) {
		if (top_v[i] > max)
			max = top_v[i];
	}

	scalar_t sum = 0;
	for (size_t i = 0; i < k; i++) {
		top_v[i] = expf(top_v[i] - max);
		sum += top_v[i];
	}

	for (size_t i = 0; i < k; i++)
		top_v[i] = top_v[i] / sum;

	scalar_t rem = drand48();

	size_t idx = 0;
	for (int i = k - 1; i >= 0; i--) {
		if (rem > top_v[i]) {
			rem -= top_v[i];
			continue;
		}

		idx = i;
		break;
	}

	return top_n[idx];
}

static size_t on_token(void *ctx, tensor_t *logits)
{
	struct snapshot *ss = ctx;
	struct file *vocab;
	const char *s;
	size_t token;

#if 0
	token = pick_greedy(logits);
#else
	token = pick_top_k(logits, 5);
#endif

	vocab = snapshot_vocab(ss);
	s = vocab_encode(vocab, token);
	printf("%s", s);
	fflush(stdout);

	return token;
}

void bt(int sig)
{
	void *buf[256];
	char **sym;
	int num;

	num = backtrace(buf, ARRAY_SIZE(buf));
	sym = backtrace_symbols(buf, num);
	if (!sym)
		return;

	printf("\n");
	printf("Ugh, I'm done, see the backtrace below.\n");
	printf("You can use the following command to find the faulty location in the source code:\n");
	printf("$ addr2line -e ./llmc -i <func>+<offset>\n");
	printf("\n");

	for (int i = 0; i < num; i++) {
		printf("%s\n", sym[i]);
	}
	free(sym);

}

int main(int argc, char *argv[])
{
	struct snapshot *ss;
	struct gpt2 *model;

	signal(SIGABRT, bt);

	if (argc < 1) {
		fprintf(stderr, "Usage: %s <path.llmc>\n", argv[0]);
		return EXIT_FAILURE;
	}

	assert(argc > 1);
	printf("loading model from %s\n", argv[1]);
	ss = snapshot_load(argv[1]);
	if (!ss) {
		fprintf(stderr, "failed to load model from '%s'\n", argv[1]);
		return EXIT_FAILURE;
	}

	assert(snapshot_config_int(ss, "version") == 1);

	if (getenv("SRAND48_SEED")) {
		srand48(atoi(getenv("SRAND48_SEED"))); /* reproducible output */
	} else {
		struct timespec ts = {};
		clock_gettime(CLOCK_MONOTONIC, &ts);
		srand48(ts.tv_sec * 1000000000 + ts.tv_nsec);
	}

	model = gpt2_load(ss);
	if (!model) {
		printf("failed to load model\n");
		return EXIT_FAILURE;
	}
	if (argc == 2) {
		top_k_test();
		gpt2_test_no_cache(model);
		gpt2_test_cache(model);
	} else {
		size_t sz = 0;
		for (int i = 2; i < argc; i++) {
			sz += strlen(argv[i]);
			sz += 1; /* space */
		}

		char *inp = malloc(sz + 1 /* \0 */);
		inp[0] = '\0';
		assert(inp);

		for (int i = 2; i < argc; i++) {
			if (i != 2)
				strcat(inp, " ");
			strcat(inp, argv[i]);
		}

		gpt2_generate(model, inp, 500, on_token, ss);
	}
	gpt2_close(model);
	snapshot_close(ss);
}
