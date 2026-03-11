#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

struct gguf;

struct gguf *gguf_load(const char *path);
void gguf_close(struct gguf *g);

size_t gguf_tensor_count(const struct gguf *g);
size_t gguf_metadata_count(const struct gguf *g);

/* metadata access by key */
const char *gguf_get_str(const struct gguf *g, const char *key);
uint64_t gguf_get_uint64(const struct gguf *g, const char *key);
uint32_t gguf_get_uint32(const struct gguf *g, const char *key);
float gguf_get_float32(const struct gguf *g, const char *key);

/* tensor access by name; returns mmap-backed tensor (zero-copy) */
tensor_t *gguf_tensor(const struct gguf *g, const char *name, size_t ndim, ...);
