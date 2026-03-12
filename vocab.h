#pragma once

#include <stddef.h>

struct gguf;

const char *vocab_encode(struct gguf *g, size_t token);
int vocab_decode(struct gguf *g, const char *s, int *sz);
