#include <stdlib.h>
#include <string.h>
#include "snapshot.h"

struct vocab_entry {
	const char *str;
	int str_len;
	int token_idx;
};

struct vocab_ht {
	struct vocab_entry *buckets;
	size_t mask;
	int max_token_len;
};

static uint32_t fnv1a(const char *s, int len)
{
	uint32_t h = 0x811c9dc5;
	for (int i = 0; i < len; i++) {
		h ^= (unsigned char)s[i];
		h *= 0x01000193;
	}
	return h;
}

static void ht_insert(struct vocab_ht *ht, const char *str, int str_len,
		       int token_idx)
{
	uint32_t h = fnv1a(str, str_len) & ht->mask;

	while (ht->buckets[h].str) {
		struct vocab_entry *e = &ht->buckets[h];
		if (e->str_len == str_len && memcmp(e->str, str, str_len) == 0) {
			e->token_idx = token_idx;
			return;
		}
		h = (h + 1) & ht->mask;
	}

	ht->buckets[h].str = str;
	ht->buckets[h].str_len = str_len;
	ht->buckets[h].token_idx = token_idx;
}

static int ht_lookup(struct vocab_ht *ht, const char *s, int len)
{
	uint32_t h = fnv1a(s, len) & ht->mask;

	while (ht->buckets[h].str) {
		struct vocab_entry *e = &ht->buckets[h];
		if (e->str_len == len && memcmp(e->str, s, len) == 0)
			return e->token_idx;
		h = (h + 1) & ht->mask;
	}

	return -1;
}

static struct vocab_ht *vocab_ht_build(struct file *f)
{
	size_t n = file_len(f);
	size_t cap = 1;
	while (cap < 2 * n)
		cap <<= 1;

	struct vocab_ht *ht = malloc(sizeof(*ht));
	ht->buckets = calloc(cap, sizeof(struct vocab_entry));
	ht->mask = cap - 1;
	ht->max_token_len = 0;

	for (size_t i = 0; i < n; i++) {
		const char *tok = file_at(f, i);
		int slen = file_at_len(f, i) - 1;

		if (slen > ht->max_token_len)
			ht->max_token_len = slen;

		ht_insert(ht, tok, slen, i);
	}

	return ht;
}

int vocab_decode(struct file *f, const char *s, int *sz)
{
	static struct vocab_ht *ht;

	if (!ht)
		ht = vocab_ht_build(f);

	if (*s == 0)
		return -1;

	int remaining = strlen(s);
	int limit = ht->max_token_len < remaining ? ht->max_token_len : remaining;
	int best_len = 0;
	int best_idx = -1;

	for (int len = 1; len <= limit; len++) {
		int idx = ht_lookup(ht, s, len);
		if (idx >= 0) {
			best_len = len;
			best_idx = idx;
		}
	}

	*sz = best_len;
	return best_idx;
}

const char *vocab_encode(struct file *f, size_t token)
{
	assert(token < file_len(f));
	return file_at(f, token);
}
