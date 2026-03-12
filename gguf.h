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

/* array metadata access */
size_t gguf_get_arr_n(const struct gguf *g, const char *key);
const char *gguf_get_arr_str(const struct gguf *g, const char *key, size_t index);

tensor_t *gguf_tensor_1d(const struct gguf *g, size_t d1, const char *fmt, ...);
tensor_t *gguf_tensor_2d(const struct gguf *g, size_t d1, size_t d2, const char *fmt, ...);
tensor_t *gguf_tensor_3d(const struct gguf *g, size_t d1, size_t d2, size_t d3, const char *fmt, ...);
