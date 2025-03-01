#pragma once

#include <math.h>
#include <immintrin.h>

/* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html */

typedef __m512 avx512_vector_t;

#define AVX512_CHUNK   16
#define avx512_CHUNK   16

static inline void avx512_vector_load(avx512_vector_t *dst, scalar_t *src)
{
	*dst = _mm512_loadu_ps(src); \
}

static inline void avx512_vector_load1(avx512_vector_t *dst, scalar_t val)
{
	*dst = _mm512_set1_ps(val);
}

static inline void avx512_vector_store(scalar_t *dst, avx512_vector_t *src)
{
	_mm512_storeu_ps(dst, *src);
}

static inline void avx512_vector_add(avx512_vector_t *dst, avx512_vector_t *lhs, avx512_vector_t *rhs)
{
	*dst = _mm512_add_ps(*lhs, *rhs);
}

static inline void avx512_vector_sub(avx512_vector_t *dst, avx512_vector_t *lhs, avx512_vector_t *rhs)
{
	*dst = _mm512_sub_ps(*lhs, *rhs);
}

static inline void avx512_vector_mul(avx512_vector_t *dst, avx512_vector_t *lhs, avx512_vector_t *rhs)
{
	*dst = _mm512_mul_ps(*lhs, *rhs);
}

static inline void avx512_vector_div(avx512_vector_t *dst, avx512_vector_t *lhs, avx512_vector_t *rhs)
{
	*dst = _mm512_div_ps(*lhs, *rhs);
}

static inline void avx512_vector_exp(avx512_vector_t *dst, avx512_vector_t *lhs)
{
	scalar_t tmp[AVX512_CHUNK]; \
	avx512_vector_store(tmp, lhs);
#pragma unroll
	for (size_t i = 0; i < AVX512_CHUNK; i++) {
		tmp[i] = expf(tmp[i]);
	}
	avx512_vector_load(dst, tmp);
}

static inline void avx512_vector_tanh(avx512_vector_t *dst, avx512_vector_t *lhs)
{
	scalar_t tmp[AVX512_CHUNK]; \
	avx512_vector_store(tmp, lhs);
#pragma unroll
	for (size_t i = 0; i < AVX512_CHUNK; i++) {
		tmp[i] = tanh(tmp[i]);
	}
	avx512_vector_load(dst, tmp);
}

static inline scalar_t avx512_vector_reduce_sum(avx512_vector_t *lhs)
{
	__m256 v5 = _mm256_add_ps(_mm512_extractf32x8_ps(*lhs, 0), _mm512_extractf32x8_ps(*lhs, 1));
	__m128 v4 = _mm_add_ps(_mm256_extractf128_ps(v5, 0), _mm256_extractf128_ps(v5, 1));
	__m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4));
	__m128 v1 = _mm_add_ss(v2, _mm_movehdup_ps(v2));
	return _mm_cvtss_f32(v1);
}

static inline scalar_t avx512_vector_reduce_max(avx512_vector_t *lhs)
{
    return _mm512_reduce_max_ps(*lhs);
}
