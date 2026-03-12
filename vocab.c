#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "gguf.h"

#define VOCAB_KEY "tokenizer.ggml.tokens"

struct vocab_entry {
	const char *str;
	int str_len;
	int token_idx;
};

struct vocab_ht {
	struct vocab_entry *buckets;
	size_t mask;
	int max_token_len;
	char **decoded; /* decoded token strings, indexed by token_id */
	size_t n_tokens;
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

/*
 * GPT-2 byte-level BPE: byte values are mapped to Unicode code points
 * so that every token is a valid Unicode string. This function reverses
 * that mapping: Unicode code point -> original byte value.
 *
 * The mapping (from GPT-2's encoder.py bytes_to_unicode()):
 *   - Bytes 33-126, 161-172, 174-255 map to the same code point
 *   - Remaining 68 bytes (0-32, 127-160, 173) map to U+0100..U+0143
 */
static int bpe_char_to_byte(uint32_t cp)
{
	if (cp >= 33 && cp <= 126) return cp;
	if (cp >= 161 && cp <= 172) return cp;
	if (cp >= 174 && cp <= 255) return cp;
	if (cp >= 256 && cp <= 323) {
		int idx = cp - 256;
		if (idx < 33) return idx;	/* bytes 0-32 */
		idx -= 33;
		if (idx < 34) return 127 + idx;	/* bytes 127-160 */
		return 173;			/* byte 173 */
	}
	return -1;
}

/* Decode a GPT-2 BPE token (UTF-8 encoded) back to raw bytes. */
static char *bpe_decode(const char *token)
{
	size_t max_len = strlen(token) + 1;
	char *out = malloc(max_len);
	size_t pos = 0;

	const uint8_t *p = (const uint8_t *)token;
	while (*p) {
		uint32_t cp;
		int nbytes;

		if (*p < 0x80) {
			cp = *p;
			nbytes = 1;
		} else if ((*p & 0xE0) == 0xC0) {
			cp = (uint32_t)(*p & 0x1F) << 6 | (p[1] & 0x3F);
			nbytes = 2;
		} else if ((*p & 0xF0) == 0xE0) {
			cp = (uint32_t)(*p & 0x0F) << 12 |
			     (uint32_t)(p[1] & 0x3F) << 6 | (p[2] & 0x3F);
			nbytes = 3;
		} else {
			cp = (uint32_t)(*p & 0x07) << 18 |
			     (uint32_t)(p[1] & 0x3F) << 12 |
			     (uint32_t)(p[2] & 0x3F) << 6 | (p[3] & 0x3F);
			nbytes = 4;
		}

		int b = bpe_char_to_byte(cp);
		if (b >= 0)
			out[pos++] = (char)b;

		p += nbytes;
	}
	out[pos] = '\0';

	return out;
}

static struct vocab_ht *vocab_ht_build(struct gguf *g)
{
	size_t n = gguf_get_arr_n(g, VOCAB_KEY);
	size_t cap = 1;
	while (cap < 2 * n)
		cap <<= 1;

	struct vocab_ht *ht = malloc(sizeof(*ht));
	ht->buckets = calloc(cap, sizeof(struct vocab_entry));
	ht->mask = cap - 1;
	ht->max_token_len = 0;
	ht->decoded = calloc(n, sizeof(char *));
	ht->n_tokens = n;

	for (size_t i = 0; i < n; i++) {
		const char *tok = gguf_get_arr_str(g, VOCAB_KEY, i);
		ht->decoded[i] = bpe_decode(tok);
		int slen = strlen(ht->decoded[i]);

		if (slen > ht->max_token_len)
			ht->max_token_len = slen;

		ht_insert(ht, ht->decoded[i], slen, i);
	}

	return ht;
}

static struct vocab_ht *ht;

static void ensure_ht(struct gguf *g)
{
	if (!ht)
		ht = vocab_ht_build(g);
}

int vocab_decode(struct gguf *g, const char *s, int *sz)
{
	ensure_ht(g);

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

const char *vocab_encode(struct gguf *g, size_t token)
{
	ensure_ht(g);
	assert(token < ht->n_tokens);
	return ht->decoded[token];
}
