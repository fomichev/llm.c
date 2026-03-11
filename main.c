#include "model.h"
#include "snapshot.h"
#include "vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <execinfo.h>
#include <time.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

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
	const struct model *m;
	struct snapshot *ss;
	char *model_name;
	void *ctx;

	signal(SIGABRT, bt);

	if (argc < 2) {
		fprintf(stderr, "Usage: %s <path.llmc> [prompt...]\n", argv[0]);
		return EXIT_FAILURE;
	}

	fprintf(stderr, "loading model from %s\n", argv[1]);
	ss = snapshot_load(argv[1]);
	if (!ss) {
		fprintf(stderr, "failed to load model from '%s'\n", argv[1]);
		return EXIT_FAILURE;
	}

	assert(snapshot_config_int(ss, "version") == 1);

	model_name = snapshot_config_str(ss, "model");
	m = find_model(model_name);
	if (!m) {
		fprintf(stderr, "unknown model '%s'\n", model_name);
		free(model_name);
		return EXIT_FAILURE;
	}
	free(model_name);

	if (getenv("SRAND48_SEED")) {
		srand48(atoi(getenv("SRAND48_SEED"))); /* reproducible output */
	} else {
		struct timespec ts = {};
		clock_gettime(CLOCK_MONOTONIC, &ts);
		srand48(ts.tv_sec * 1000000000 + ts.tv_nsec);
	}

	ctx = m->load(ss);
	if (!ctx) {
		fprintf(stderr, "failed to load model\n");
		return EXIT_FAILURE;
	}
	if (argc == 2 && !isatty(fileno(stdin))) {
		size_t cap = 4096, len = 0;
		char *inp = malloc(cap);
		assert(inp);
		size_t n;
		while ((n = fread(inp + len, 1, cap - len, stdin)) > 0) {
			len += n;
			if (len == cap) {
				cap *= 2;
				inp = realloc(inp, cap);
				assert(inp);
			}
		}
		inp[len] = '\0';

		m->generate(ctx, inp, 250, on_token, ss);
		free(inp);
	} else if (argc >= 3) {
		size_t sz = 0;
		for (int i = 2; i < argc; i++) {
			sz += strlen(argv[i]);
			sz += 1; /* space */
		}

		char *inp = malloc(sz + 1 /* \0 */);
		assert(inp);
		inp[0] = '\0';

		for (int i = 2; i < argc; i++) {
			if (i != 2)
				strcat(inp, " ");
			strcat(inp, argv[i]);
		}

		m->generate(ctx, inp, 500, on_token, ss);
	}
	m->close(ctx);
	snapshot_close(ss);
}
