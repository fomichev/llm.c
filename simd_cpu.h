#pragma once

#include <math.h>

#define CPU_CHUNK   4
#define cpu_CHUNK   4

typedef struct {
	scalar_t v [CPU_CHUNK];
} cpu_vector_t;

static inline void cpu_vector_load(cpu_vector_t *dst, scalar_t *src)
{
	__builtin_memcpy(dst, src, CPU_CHUNK * sizeof(scalar_t));
}

static inline void cpu_vector_load1(cpu_vector_t *dst, scalar_t val)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = val;
	}
}

static inline void cpu_vector_store(scalar_t *dst, cpu_vector_t *src)
{
	__builtin_memcpy(dst, src, CPU_CHUNK * sizeof(scalar_t));
}

static inline void cpu_vector_add(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] + rhs->v[i];
	}
}

static inline void cpu_vector_sub(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] - rhs->v[i];
	}
}

static inline void cpu_vector_mul(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] * rhs->v[i];
	}
}

static inline void cpu_vector_div(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] / rhs->v[i];
	}
}

static inline void cpu_vector_exp(cpu_vector_t *dst, cpu_vector_t *lhs)
{
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = expf(lhs->v[i]);
	}
}

static inline void cpu_vector_tanh(cpu_vector_t *dst, cpu_vector_t *lhs)
{
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = tanhf(lhs->v[i]);
	}
}

static inline scalar_t cpu_vector_reduce_sum(cpu_vector_t *lhs)
{
	scalar_t sum = 0;
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		sum += lhs->v[i];
	}
	return sum;
}

static inline scalar_t cpu_vector_reduce_max(cpu_vector_t *lhs)
{
	scalar_t ret = lhs->v[0];
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		if (lhs->v[i] > ret) {
			ret = lhs->v[i];
		}
	}
	return ret;
}
