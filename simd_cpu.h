#pragma once

#include <math.h>

#define CPU_CHUNK   4
#define cpu_CHUNK   4

typedef struct {
	scalar_t v [CPU_CHUNK];
} cpu_fv_t;

static inline void cpu_fv_load(cpu_fv_t *dst, scalar_t *src)
{
	__builtin_memcpy(dst, src, CPU_CHUNK * sizeof(scalar_t));
}

static inline void cpu_fv_load1(cpu_fv_t *dst, scalar_t val)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = val;
	}
}

static inline void cpu_fv_store(scalar_t *dst, cpu_fv_t *src)
{
	__builtin_memcpy(dst, src, CPU_CHUNK * sizeof(scalar_t));
}

static inline void cpu_fv_add(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] + rhs->v[i];
	}
}

static inline void cpu_fv_sub(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] - rhs->v[i];
	}
}

static inline void cpu_fv_mul(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] * rhs->v[i];
	}
}

static inline void cpu_fv_div(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = lhs->v[i] / rhs->v[i];
	}
}

static inline void cpu_fv_exp(cpu_fv_t *dst, cpu_fv_t *lhs)
{
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = expf(lhs->v[i]);
	}
}

static inline void cpu_fv_tanh(cpu_fv_t *dst, cpu_fv_t *lhs)
{
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		dst->v[i] = tanhf(lhs->v[i]);
	}
}

static inline scalar_t cpu_fv_reduce_sum(cpu_fv_t *lhs)
{
	scalar_t sum = 0;
#pragma unroll
	for (size_t i = 0; i < CPU_CHUNK; i++) {
		sum += lhs->v[i];
	}
	return sum;
}

static inline scalar_t cpu_fv_reduce_max(cpu_fv_t *lhs)
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
