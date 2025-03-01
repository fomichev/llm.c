#pragma once

#include <string.h>

/* TODO: fv_t -> vector_t */
/* TODO: ft_t -> tensor_t */
typedef float scalar_t;

#include "simd_cpu.h"
#include "simd_avx2.h"
#include "simd_avx512.h"

#if defined(__AVX512F__)
typedef avx512_fv_t fv_t;
#define FV_CHUNK            AVX512_CHUNK
#elif defined(__AVX2__)
typedef avx2_fv_t fv_t;
#define FV_CHUNK            AVX2_CHUNK
#else
typedef cpu_fv_t fv_t;
#define FV_CHUNK            CPU_CHUNK
#endif

#define FV_ALIGN	(sizeof(scalar_t) * FV_CHUNK)

static inline size_t fv_chunks(size_t size)
{
    return (size & ~(FV_ALIGN-1));
}

static inline void fv_load(fv_t *dst, scalar_t *src)
{
#if defined(__AVX512F__)
    avx512_fv_load(dst, src);
#elif defined(__AVX2__)
    avx2_fv_load(dst, src);
#else
    cpu_fv_load(dst, src);
#endif
}

static inline void fv_load1(fv_t *dst, scalar_t val)
{
#if defined(__AVX512F__)
    avx512_fv_load1(dst, val);
#elif defined(__AVX2__)
    avx2_fv_load1(dst, val);
#else
    cpu_fv_load1(dst, val);
#endif
}

static inline void fv_store(scalar_t *dst, fv_t *src)
{
#if defined(__AVX512F__)
    avx512_fv_store(dst, src);
#elif defined(__AVX2__)
    avx2_fv_store(dst, src);
#else
    cpu_fv_store(dst, src);
#endif
}

static inline void fv_add(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    avx512_fv_add(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_fv_add(dst, lhs, rhs);
#else
    cpu_fv_add(dst, lhs, rhs);
#endif
}

static inline void fv_sub(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    avx512_fv_sub(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_fv_sub(dst, lhs, rhs);
#else
    cpu_fv_sub(dst, lhs, rhs);
#endif
}

static inline void fv_mul(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    avx512_fv_mul(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_fv_mul(dst, lhs, rhs);
#else
    cpu_fv_mul(dst, lhs, rhs);
#endif
}

static inline void fv_div(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    avx512_fv_div(dst, lhs, rhs);
#elif defined(__AVX2__)
    avx2_fv_div(dst, lhs, rhs);
#else
    cpu_fv_div(dst, lhs, rhs);
#endif
}

static inline void fv_exp(fv_t *dst, fv_t *lhs)
{
#if defined(__AVX512F__)
    avx512_fv_exp(dst, lhs);
#elif defined(__AVX2__)
    avx2_fv_exp(dst, lhs);
#else
    cpu_fv_exp(dst, lhs);
#endif
}

static inline void fv_tanh(fv_t *dst, fv_t *lhs)
{
#if defined(__AVX512F__)
    avx512_fv_tanh(dst, lhs);
#elif defined(__AVX2__)
    avx2_fv_tanh(dst, lhs);
#else
    cpu_fv_tanh(dst, lhs);
#endif
}

static inline scalar_t fv_reduce_sum(fv_t *lhs)
{
#if defined(__AVX512F__)
    return avx512_fv_reduce_sum(lhs);
#elif defined(__AVX2__)
    return avx2_fv_reduce_sum(lhs);
#else
    return cpu_fv_reduce_sum(lhs);
#endif
}

static inline scalar_t fv_reduce_max(fv_t *lhs)
{
#if defined(__AVX512F__)
    return avx512_fv_reduce_max(lhs);
#elif defined(__AVX2__)
    return avx2_fv_reduce_max(lhs);
#else
    return cpu_fv_reduce_max(lhs);
#endif
}
