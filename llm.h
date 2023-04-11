#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#include "simd.h"
#include "tensor.h"

struct file;

struct file *file_load(const char *meta, const char *vocab);
void file_close(struct file *f);
size_t file_len(const struct file *f);
void *file_at(const struct file *f, size_t pos);
size_t file_at_len(const struct file *f, size_t pos);
bool file_is_eof(struct file *f);
ft_t *file_ft(struct file *f, size_t ndim, ...);

struct snapshot;

struct snapshot *snapshot_load(const char *path);
void snapshot_close(struct snapshot *ss);
struct file *snapshot_param(struct snapshot *ss);
struct file *snapshot_vocab(struct snapshot *ss);
int snapshot_config_int(struct snapshot *ss, const char *k);

const char *vocab_encode(struct file *f, size_t token);
int vocab_decode(struct file *f, const char *s, int *sz);

struct gpt2;

typedef size_t (*pick_token_t)(void *ctx, ft_t *logits);

struct gpt2 *gpt2_load(struct snapshot *ss);
void gpt2_test_no_cache(struct gpt2 *model);
void gpt2_test_cache(struct gpt2 *model);
void gpt2_generate(struct gpt2 *model, const char *text, int num, pick_token_t f, void *ctx);
void gpt2_close(struct gpt2 *model);
