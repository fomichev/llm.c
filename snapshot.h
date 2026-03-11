#pragma once

#include <stddef.h>
#include <stdbool.h>

#include "tensor.h"

struct file;

struct file *file_load(const char *meta, const char *vocab);
void file_close(struct file *f);
size_t file_len(const struct file *f);
void *file_at(const struct file *f, size_t pos);
size_t file_at_len(const struct file *f, size_t pos);
bool file_is_eof(struct file *f);
tensor_t *file_tensor(struct file *f, size_t ndim, ...);

struct snapshot;

struct snapshot *snapshot_load(const char *path);
void snapshot_close(struct snapshot *ss);
struct file *snapshot_param(struct snapshot *ss);
struct file *snapshot_vocab(struct snapshot *ss);
int snapshot_config_int(struct snapshot *ss, const char *k);
