#pragma once

#include <math.h>

#define CPU_BATCH   4

typedef struct {
	scalar_t v [CPU_BATCH];
} cpu_vector_t;

static inline void cpu_vector_load(cpu_vector_t *dst, scalar_t *src)
{
	__builtin_memcpy(dst, src, CPU_BATCH * sizeof(scalar_t));
}

static inline void cpu_vector_set(cpu_vector_t *dst, scalar_t val)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = val;
	}
}

static inline void cpu_vector_store(scalar_t *dst, cpu_vector_t *src)
{
	__builtin_memcpy(dst, src, CPU_BATCH * sizeof(scalar_t));
}

static inline void cpu_vector_add(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = lhs->v[i] + rhs->v[i];
	}
}

static inline void cpu_vector_sub(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = lhs->v[i] - rhs->v[i];
	}
}

static inline void cpu_vector_mul(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = lhs->v[i] * rhs->v[i];
	}
}

static inline void cpu_vector_div(cpu_vector_t *dst, cpu_vector_t *lhs, cpu_vector_t *rhs)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = lhs->v[i] / rhs->v[i];
	}
}

static inline void cpu_vector_exp(cpu_vector_t *dst, cpu_vector_t *lhs)
{
#pragma unroll(CPU_BATCH)
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = expf(lhs->v[i]);
	}
}

static inline void cpu_vector_tanh(cpu_vector_t *dst, cpu_vector_t *lhs)
{
#pragma unroll(CPU_BATCH)
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = tanhf(lhs->v[i]);
	}
}

static inline scalar_t cpu_vector_reduce_sum(cpu_vector_t *lhs)
{
	scalar_t sum = 0;
#pragma unroll(CPU_BATCH)
	for (size_t i = 0; i < CPU_BATCH; i++) {
		sum += lhs->v[i];
	}
	return sum;
}

static inline void cpu_vector_i8_to_f32(cpu_vector_t *dst, const int8_t *src)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = (float)src[i];
	}
}

static inline void cpu_vector_u4_lo_to_f32(cpu_vector_t *dst, const uint8_t *src)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = (float)((src[i] & 0x0F) - 8);
	}
}

static inline void cpu_vector_u4_hi_to_f32(cpu_vector_t *dst, const uint8_t *src)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = (float)((src[i] >> 4) - 8);
	}
}

static inline void cpu_vector_u4_lo_to_f32_unsigned(cpu_vector_t *dst, const uint8_t *src)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = (float)(src[i] & 0x0F);
	}
}

static inline void cpu_vector_u4_hi_to_f32_unsigned(cpu_vector_t *dst, const uint8_t *src)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = (float)(src[i] >> 4);
	}
}

static inline void cpu_vector_fma(cpu_vector_t *dst, cpu_vector_t *a, cpu_vector_t *b, cpu_vector_t *c)
{
	for (size_t i = 0; i < CPU_BATCH; i++) {
		dst->v[i] = a->v[i] * b->v[i] + c->v[i];
	}
}

static inline scalar_t cpu_vector_reduce_max(cpu_vector_t *lhs)
{
	scalar_t ret = lhs->v[0];
#pragma unroll(CPU_BATCH)
	for (size_t i = 0; i < CPU_BATCH; i++) {
		if (lhs->v[i] > ret) {
			ret = lhs->v[i];
		}
	}
	return ret;
}
