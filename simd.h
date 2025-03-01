#pragma once

#include <string.h>

typedef float scalar_t;

#include "simd_cpu.h"
#include "simd_avx2.h"
#include "simd_avx512.h"

#if defined(__AVX512F__)
typedef avx512_vector_t vector_t;
#define VECTOR_BATCH AVX512_BATCH
#elif defined(__AVX2__)
typedef avx2_vector_t vector_t;
#define VECTOR_BATCH AVX2_BATCH
#else
typedef cpu_vector_t vector_t;
#define VECTOR_BATCH CPU_BATCH
#endif

#define VECTOR_ALIGN (sizeof(scalar_t) * VECTOR_BATCH)

static inline size_t vector_batches(size_t size)
{
    return (size & ~(VECTOR_ALIGN-1));
}

static inline void vector_load(vector_t *dst, scalar_t *src)
{
#if defined(__AVX512F__)
    avx512_vector_load(dst, src);
#elif defined(__AVX2__)
    avx2_vector_load(dst, src);
#else
    cpu_vector_load(dst, src);
#endif
}

static inline void vector_set(vector_t *dst, scalar_t val)
{
#if defined(__AVX512F__)
    avx512_vector_load1(dst, val);
#elif defined(__AVX2__)
    avx2_vector_load1(dst, val);
#else
    cpu_vector_load1(dst, val);
#endif
}

static inline void vector_store(scalar_t *dst, vector_t *src)
{
#if defined(__AVX512F__)
    avx512_vector_store(dst, src);
#elif defined(__AVX2__)
    avx2_vector_store(dst, src);
#else
    cpu_vector_store(dst, src);
#endif
}

static inline void vector_add(vector_t *dst, vector_t *lhs, vector_t *rhs)
{
#if defined(__AVX512F__)
    avx512_vector_add(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_vector_add(dst, lhs, rhs);
#else
    cpu_vector_add(dst, lhs, rhs);
#endif
}

static inline void vector_sub(vector_t *dst, vector_t *lhs, vector_t *rhs)
{
#if defined(__AVX512F__)
    avx512_vector_sub(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_vector_sub(dst, lhs, rhs);
#else
    cpu_vector_sub(dst, lhs, rhs);
#endif
}

static inline void vector_mul(vector_t *dst, vector_t *lhs, vector_t *rhs)
{
#if defined(__AVX512F__)
    avx512_vector_mul(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_vector_mul(dst, lhs, rhs);
#else
    cpu_vector_mul(dst, lhs, rhs);
#endif
}

static inline void vector_div(vector_t *dst, vector_t *lhs, vector_t *rhs)
{
#if defined(__AVX512F__)
    avx512_vector_div(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_vector_div(dst, lhs, rhs);
#else
    cpu_vector_div(dst, lhs, rhs);
#endif
}

static inline void vector_exp(vector_t *dst, vector_t *lhs)
{
#if defined(__AVX512F__)
    avx512_vector_exp(dst, lhs);
#elif defined(__AVX2__)
    avx2_vector_exp(dst, lhs);
#else
    cpu_vector_exp(dst, lhs);
#endif
}

static inline void vector_tanh(vector_t *dst, vector_t *lhs)
{
#if defined(__AVX512F__)
    avx512_vector_tanh(dst, lhs);
#elif defined(__AVX2__)
    avx2_vector_tanh(dst, lhs);
#else
    cpu_vector_tanh(dst, lhs);
#endif
}

static inline scalar_t vector_reduce_sum(vector_t *lhs)
{
#if defined(__AVX512F__)
    return avx512_vector_reduce_sum(lhs);
#elif defined(__AVX2__)
    return avx2_vector_reduce_sum(lhs);
#else
    return cpu_vector_reduce_sum(lhs);
#endif
}

static inline scalar_t vector_reduce_max(vector_t *lhs)
{
#if defined(__AVX512F__)
    return avx512_vector_reduce_max(lhs);
#elif defined(__AVX2__)
    return avx2_vector_reduce_max(lhs);
#else
    return cpu_vector_reduce_max(lhs);
#endif
}
