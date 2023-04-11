#pragma once

#include <time.h>

#define PROFILER_SIZE		16

static uint64_t profiler_last;
static uint64_t profiler_points[PROFILER_SIZE] = {};
static const char *profiler_names[PROFILER_SIZE] = {};

static inline uint64_t profiler_now(void)
{
	struct timespec ts = {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static inline double profiler_to_sec(uint64_t t)
{
	return (double)t / 1000000000;
}

static inline void profiler_start(void)
{
	profiler_last = profiler_now();
}

static inline void profiler_record(size_t pos, const char *name)
{
	uint64_t now;

	assert(pos < PROFILER_SIZE);

	now = profiler_now();
	profiler_points[pos] += now - profiler_last;
	profiler_names[pos] = name;
	profiler_last = now;
}

static inline void profiler_report(void)
{
	uint64_t total = 0;
	for (size_t i = 0; i < ARRAY_SIZE(profiler_points); i++) {
		uint64_t val;

		if (profiler_points[i] == 0)
			continue;

		val = profiler_points[i];
		printf("%.9fs %s\n", profiler_to_sec(val), profiler_names[i]);
		total += val;
	}
	printf("total=%fs\n", profiler_to_sec(total));
}
