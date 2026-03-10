#include "profiler.h"

uint64_t profiler_last;
uint64_t profiler_points[PROFILER_SIZE] = {};
const char *profiler_names[PROFILER_SIZE] = {};
