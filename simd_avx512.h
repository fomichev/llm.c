#pragma once

#include <math.h>
#include <immintrin.h>

/* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html */

typedef __m512 avx512_fv_t;

#define AVX512_N   16

static inline void AVX512_fv_load(avx512_fv_t *dst, FT_TYPE *src)
{
	*dst = _mm512_loadu_ps(src); \
}

static inline void avx512_fv_load1(avx512_fv_t *dst, FT_TYPE val)
{
	*dst = _mm512_set1_ps(val);
}

static inline void AVX512_fv_store(FT_TYPE *dst, avx512_fv_t *src)
{
	_mm512_storeu_ps(dst, *src);
}

static inline void AVX512_fv_add(avx512_fv_t *dst, avx512_fv_t *lhs, avx512_fv_t *rhs)
{
	*dst = _mm512_add_ps(*lhs, *rhs);
}

static inline void AVX512_fv_sub(avx512_fv_t *dst, avx512_fv_t *lhs, avx512_fv_t *rhs)
{
	*dst = _mm512_sub_ps(*lhs, *rhs);
}

static inline void AVX512_fv_mul(avx512_fv_t *dst, avx512_fv_t *lhs, avx512_fv_t *rhs)
{
	*dst = _mm512_mul_ps(*lhs, *rhs);
}

static inline void AVX512_fv_div(avx512_fv_t *dst, avx512_fv_t *lhs, avx512_fv_t *rhs)
{
	*dst = _mm512_div_ps(*lhs, *rhs);
}

static inline void AVX512_fv_exp(avx512_fv_t *dst, avx512_fv_t *lhs)
{
	FT_TYPE tmp[AVX512_N]; \
	AVX512_fv_store(tmp, lhs);
#pragma unroll
	for (size_t i = 0; i < AVX512_N; i++) {
		tmp[i] = expf(tmp[i]);
	}
	AVX512_fv_load(dst, tmp);
}

static inline void AVX512_fv_tanh(avx512_fv_t *dst, avx512_fv_t *lhs)
{
	FT_TYPE tmp[AVX512_N]; \
	AVX512_fv_store(tmp, lhs);
#pragma unroll
	for (size_t i = 0; i < AVX512_N; i++) {
		tmp[i] = tanh(tmp[i]);
	}
	AVX512_fv_load(dst, tmp);
}

static inline FT_TYPE AVX512_fv_reduce_sum(avx512_fv_t *lhs)
{
	__m256 v5 = _mm256_add_ps(_mm512_extractf32x8_ps(*lhs, 0), _mm512_extractf32x8_ps(*lhs, 1));
	__m128 v4 = _mm_add_ps(_mm256_extractf128_ps(v5, 0), _mm256_extractf128_ps(v5, 1));
	__m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4));
	__m128 v1 = _mm_add_ss(v2, _mm_movehdup_ps(v2));
	return _mm_cvtss_f32(v1);
}

static inline FT_TYPE AVX512_fv_reduce_max(avx512_fv_t *lhs)
{
        return _mm512_reduce_max_ps(*lhs);
}
