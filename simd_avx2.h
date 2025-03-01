#pragma once

#include <math.h>
#include <immintrin.h>

/* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html */

typedef __m256 avx2_fv_t;

#define AVX2_CHUNK   8
#define avx2_CHUNK   8

static inline void avx2_fv_load(avx2_fv_t *dst, scalar_t *src)
{
	*dst = _mm256_loadu_ps(src);
}

static inline void avx2_fv_load1(avx2_fv_t *dst, scalar_t val)
{
	*dst = _mm256_set1_ps(val);
}

static inline void avx2_fv_store(scalar_t *dst, avx2_fv_t *src)
{
	_mm256_storeu_ps(dst, *src);
}

static inline void avx2_fv_add(avx2_fv_t *dst, avx2_fv_t *lhs, avx2_fv_t *rhs)
{
	*dst = _mm256_add_ps(*lhs, *rhs);
}

static inline void avx2_fv_sub(avx2_fv_t *dst, avx2_fv_t *lhs, avx2_fv_t *rhs)
{
	*dst = _mm256_sub_ps(*lhs, *rhs);
}

static inline void avx2_fv_mul(avx2_fv_t *dst, avx2_fv_t *lhs, avx2_fv_t *rhs)
{
	*dst = _mm256_mul_ps(*lhs, *rhs);
}

static inline void avx2_fv_div(avx2_fv_t *dst, avx2_fv_t *lhs, avx2_fv_t *rhs)
{
	*dst = _mm256_div_ps(*lhs, *rhs);
}

static inline void avx2_fv_exp(avx2_fv_t *dst, avx2_fv_t *lhs)
{
	scalar_t tmp[AVX2_CHUNK]; \
	avx2_fv_store(tmp, lhs);
#pragma unroll
	for (size_t i = 0; i < AVX2_CHUNK; i++) {
		tmp[i] = expf(tmp[i]);
	}
	avx2_fv_load(dst, tmp);
}

static inline void avx2_fv_tanh(avx2_fv_t *dst, avx2_fv_t *lhs)
{
	scalar_t tmp[AVX2_CHUNK]; \
	avx2_fv_store(tmp, lhs);
#pragma unroll
	for (size_t i = 0; i < AVX2_CHUNK; i++) {
		tmp[i] = tanh(tmp[i]);
	}
	avx2_fv_load(dst, tmp);
}

static inline scalar_t avx2_fv_reduce_sum(avx2_fv_t *lhs)
{
	__m128 v4 = _mm_add_ps(_mm256_castps256_ps128(*lhs), _mm256_extractf128_ps(*lhs, 1));
	__m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4));
	__m128 v1 = _mm_add_ss(v2, _mm_movehdup_ps(v2));
	return _mm_cvtss_f32(v1);
}

static inline scalar_t avx2_fv_reduce_max(avx2_fv_t *lhs)
{
	scalar_t tmp[AVX2_CHUNK];
	avx2_fv_store(tmp, lhs);
	for (size_t i = 1; i < AVX2_CHUNK; i++) {
        if (tmp[i] > tmp[0]) {
            tmp[0] = tmp[i];
        }
    }
    return tmp[0];
}
