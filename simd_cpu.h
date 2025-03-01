#pragma once

#include <math.h>

#define CPU_N   4

typedef struct {
	FT_TYPE v[CPU_N];
} cpu_fv_t;

static inline void CPU_fv_load(cpu_fv_t *dst, FT_TYPE *src)
{
	__builtin_memcpy(dst, src, CPU_N * FT_SIZEOF);
}

static inline void cpu_fv_load(cpu_fv_t *dst, FT_TYPE val)
{
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = val;
	}
}

static inline void CPU_fv_store(FT_TYPE *dst, cpu_fv_t *src)
{
	__builtin_memcpy(dst, src, CPU_N * FT_SIZEOF);
}

static inline void CPU_fv_add(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = lhs->v[i] + rhs->v[i];
	}
}

static inline void CPU_fv_sub(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = lhs->v[i] - rhs->v[i];
	}
}

static inline void CPU_fv_mul(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = lhs->v[i] * rhs->v[i];
	}
}

static inline void CPU_fv_div(cpu_fv_t *dst, cpu_fv_t *lhs, cpu_fv_t *rhs)
{
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = lhs->v[i] / rhs->v[i];
	}
}

static inline void CPU_fv_exp(cpu_fv_t *dst, cpu_fv_t *lhs)
{
#pragma unroll
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = expf(lhs->v[i]);
	}
}

static inline void CPU_fv_tanh(cpu_fv_t *dst, cpu_fv_t *lhs)
{
#pragma unroll
	for (size_t i = 0; i < CPU_N; i++) {
		dst->v[i] = tanhf(lhs->v[i]);
	}
}

static inline FT_TYPE CPU_fv_reduce_sum(cpu_fv_t *lhs)
{
	FT_TYPE sum = 0;
#pragma unroll
	for (size_t i = 0; i < CPU_N; i++) {
		sum += lhs->v[i];
	}
	return sum;
}

static inline FT_TYPE CPU_fv_reduce_max(cpu_fv_t *lhs)
{
	FT_TYPE ret = lhs->v[0];
#pragma unroll
	for (size_t i = 0; i < CPU_N; i++) {
		if (lhs->v[i] > ret) {
			ret = lhs->v[i];
		}
	}
	return ret;
}
