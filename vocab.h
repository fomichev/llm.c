#pragma once

#include <stddef.h>

struct file;

const char *vocab_encode(struct file *f, size_t token);
int vocab_decode(struct file *f, const char *s, int *sz);
