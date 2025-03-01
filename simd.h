#pragma once

#include <string.h>

/* TODO: FT_TYPE to scalar_t */
/* TODO: fv_t -> vector_t */
/* TODO: ft_t -> tensor_t */
#define FT_TYPE		float
#define FT_SIZEOF	sizeof(FT_TYPE)
#define FT_ALIGN	(FT_SIZEOF * FT_N)
#define FT_LEN(x)	(x & ~(FT_ALIGN-1))

#include "simd_cpu.h"
#include "simd_avx2.h"
#include "simd_avx512.h"


#if defined(__AVX512F__)
typedef avx512_fv_t fv_t;
#define FT_N            AVX512_N
#elif defined(__AVX2__)
typedef avx2_fv_t fv_t;
#define FT_N            AVX2_N
#else
typedef cpu_fv_t fv_t;
#define FT_N            CPU_N
#endif

static inline void fv_load(fv_t *dst, FT_TYPE *src)
{
#if defined(__AVX512F__)
    AVX512_fv_load(dst, src);
#elif defined(__AVX2__)
    AVX2_fv_load(dst, src);
#else
    CPU_fv_load(dst, src);
#endif
}

static inline void fv_load1(fv_t *dst, FT_TYPE val)
{
#if defined(__AVX512F__)
    avx512_fv_load1(dst, val);
#elif defined(__AVX2__)
    avx2_fv_load1(dst, val);
#else
    cpu_fv_load1(dst, val);
#endif
}

static inline void fv_store(FT_TYPE *dst, fv_t *src)
{
#if defined(__AVX512F__)
    AVX512_fv_store(dst, src);
#elif defined(__AVX2__)
    AVX2_fv_store(dst, src);
#else
    CPU_fv_store(dst, src);
#endif
}

static inline void fv_add(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    AVX512_fv_add(dst, lhs, rhs);
#elif defined(__AVX2__)
    AVX2_fv_add(dst, lhs, rhs);
#else
    CPU_fv_add(dst, lhs, rhs);
#endif
}

static inline void fv_sub(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    AVX512_fv_sub(dst, lhs, rhs);
#elif defined(__AVX2__)
    AVX2_fv_sub(dst, lhs, rhs);
#else
    CPU_fv_sub(dst, lhs, rhs);
#endif
}

static inline void fv_mul(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    AVX512_fv_mul(dst, lhs, rhs);
#elif defined(__AVX2__)
    AVX2_fv_mul(dst, lhs, rhs);
#else
    CPU_fv_mul(dst, lhs, rhs);
#endif
}

static inline void fv_div(fv_t *dst, fv_t *lhs, fv_t *rhs)
{
#if defined(__AVX512F__)
    AVX512_fv_div(dst, lhs, rhs);
#elif defined(__AVX2__)
    AVX2_fv_div(dst, lhs, rhs);
#else
    CPU_fv_div(dst, lhs, rhs);
#endif
}

static inline void fv_exp(fv_t *dst, fv_t *lhs)
{
#if defined(__AVX512F__)
    AVX512_fv_exp(dst, lhs);
#elif defined(__AVX2__)
    AVX2_fv_exp(dst, lhs);
#else
    CPU_fv_exp(dst, lhs);
#endif
}

static inline void fv_tanh(fv_t *dst, fv_t *lhs)
{
#if defined(__AVX512F__)
    AVX512_fv_tanh(dst, lhs);
#elif defined(__AVX2__)
    AVX2_fv_tanh(dst, lhs);
#else
    CPU_fv_tanh(dst, lhs);
#endif
}

static inline FT_TYPE fv_reduce_sum(fv_t *lhs)
{
#if defined(__AVX512F__)
    return AVX512_fv_reduce_sum(lhs);
#elif defined(__AVX2__)
    return AVX2_fv_reduce_sum(lhs);
#else
    return CPU_fv_reduce_sum(lhs);
#endif
}

static inline FT_TYPE fv_reduce_max(fv_t *lhs)
{
#if defined(__AVX512F__)
    return AVX512_fv_reduce_max(lhs);
#elif defined(__AVX2__)
    return AVX2_fv_reduce_max(lhs);
#else
    return CPU_fv_reduce_max(lhs);
#endif
}
