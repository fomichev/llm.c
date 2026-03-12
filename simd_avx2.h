#pragma once

#include <math.h>
#include <immintrin.h>
#ifdef USE_SLEEF
#include <sleef.h>
#endif

/* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html */

typedef __m256 avx2_vector_t;

#define AVX2_BATCH   8

static inline void avx2_vector_load(avx2_vector_t *dst, scalar_t *src)
{
	*dst = _mm256_loadu_ps(src);
}

static inline void avx2_vector_set(avx2_vector_t *dst, scalar_t val)
{
	*dst = _mm256_set1_ps(val);
}

static inline void avx2_vector_store(scalar_t *dst, avx2_vector_t *src)
{
	_mm256_storeu_ps(dst, *src);
}

static inline void avx2_vector_add(avx2_vector_t *dst, avx2_vector_t *lhs, avx2_vector_t *rhs)
{
	*dst = _mm256_add_ps(*lhs, *rhs);
}

static inline void avx2_vector_sub(avx2_vector_t *dst, avx2_vector_t *lhs, avx2_vector_t *rhs)
{
	*dst = _mm256_sub_ps(*lhs, *rhs);
}

static inline void avx2_vector_mul(avx2_vector_t *dst, avx2_vector_t *lhs, avx2_vector_t *rhs)
{
	*dst = _mm256_mul_ps(*lhs, *rhs);
}

static inline void avx2_vector_div(avx2_vector_t *dst, avx2_vector_t *lhs, avx2_vector_t *rhs)
{
	*dst = _mm256_div_ps(*lhs, *rhs);
}

static inline void avx2_vector_exp(avx2_vector_t *dst, avx2_vector_t *lhs)
{
#ifdef USE_SLEEF
    *dst = Sleef_expf8_u10(*lhs);
#else
	scalar_t tmp[AVX2_BATCH]; \
	avx2_vector_store(tmp, lhs);
#pragma unroll(AVX2_BATCH)
	for (size_t i = 0; i < AVX2_BATCH; i++) {
		tmp[i] = expf(tmp[i]);
	}
	avx2_vector_load(dst, tmp);
#endif
}

static inline void avx2_vector_tanh(avx2_vector_t *dst, avx2_vector_t *lhs)
{
#ifdef USE_SLEEF
    *dst = Sleef_tanhf8_u10(*lhs);
#else
	scalar_t tmp[AVX2_BATCH]; \
	avx2_vector_store(tmp, lhs);
#pragma unroll(AVX2_BATCH)
	for (size_t i = 0; i < AVX2_BATCH; i++) {
		tmp[i] = tanh(tmp[i]);
	}
	avx2_vector_load(dst, tmp);
#endif
}

static inline void avx2_vector_i8_to_f32(avx2_vector_t *dst, const int8_t *src)
{
	__m128i qi8 = _mm_loadl_epi64((const __m128i *)src);
	__m256i qi32 = _mm256_cvtepi8_epi32(qi8);
	*dst = _mm256_cvtepi32_ps(qi32);
}

static inline void avx2_vector_u4_lo_to_f32(avx2_vector_t *dst, const uint8_t *src)
{
	__m128i raw = _mm_loadl_epi64((const __m128i *)src);
	__m128i lo8 = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
	__m256i lo32 = _mm256_cvtepu8_epi32(lo8);
	__m256i biased = _mm256_sub_epi32(lo32, _mm256_set1_epi32(8));
	*dst = _mm256_cvtepi32_ps(biased);
}

static inline void avx2_vector_u4_hi_to_f32(avx2_vector_t *dst, const uint8_t *src)
{
	__m128i raw = _mm_loadl_epi64((const __m128i *)src);
	__m128i hi8 = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
	__m256i hi32 = _mm256_cvtepu8_epi32(hi8);
	__m256i biased = _mm256_sub_epi32(hi32, _mm256_set1_epi32(8));
	*dst = _mm256_cvtepi32_ps(biased);
}

static inline void avx2_vector_u4_lo_to_f32_unsigned(avx2_vector_t *dst, const uint8_t *src)
{
	__m128i raw = _mm_loadl_epi64((const __m128i *)src);
	__m128i lo8 = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
	__m256i lo32 = _mm256_cvtepu8_epi32(lo8);
	*dst = _mm256_cvtepi32_ps(lo32);
}

static inline void avx2_vector_u4_hi_to_f32_unsigned(avx2_vector_t *dst, const uint8_t *src)
{
	__m128i raw = _mm_loadl_epi64((const __m128i *)src);
	__m128i hi8 = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
	__m256i hi32 = _mm256_cvtepu8_epi32(hi8);
	*dst = _mm256_cvtepi32_ps(hi32);
}

static inline void avx2_vector_fma(avx2_vector_t *dst, avx2_vector_t *a, avx2_vector_t *b, avx2_vector_t *c)
{
	*dst = _mm256_fmadd_ps(*a, *b, *c);
}

static inline scalar_t avx2_vector_reduce_sum(avx2_vector_t *lhs)
{
	__m128 v4 = _mm_add_ps(_mm256_castps256_ps128(*lhs), _mm256_extractf128_ps(*lhs, 1));
	__m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4));
	__m128 v1 = _mm_add_ss(v2, _mm_movehdup_ps(v2));
	return _mm_cvtss_f32(v1);
}

static inline scalar_t avx2_vector_reduce_max(avx2_vector_t *lhs)
{
	scalar_t tmp[AVX2_BATCH];
	avx2_vector_store(tmp, lhs);
	for (size_t i = 1; i < AVX2_BATCH; i++) {
        if (tmp[i] > tmp[0]) {
            tmp[0] = tmp[i];
        }
    }
    return tmp[0];
}
