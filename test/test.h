#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline uint64_t now(void)
{
	struct timespec ts = {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static inline double to_sec(uint64_t t)
{
	return (double)t / 1000000000;
}

static inline void bench_begin(const char *name)
{
	printf("%s benchmark: ", name);
}

static inline void bench_end(void)
{
	printf("\n");
}

static inline void bench_entry(const char *name, uint64_t scale, uint64_t val, uint64_t base)
{
	printf("%s=%.9fs/op ", name, to_sec(val / scale));
	if (base)
		printf("x%.2f ", (float)base / (float)val);
}
