#pragma once

#include <immintrin.h>

/* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html */

typedef __m512 avx512_fv_t;

#define AVX512_N   16

#define AVX512_FV_LOAD(DST, SRC) \
	({ \
		(DST) = _mm512_loadu_ps((SRC) + 0); \
	})

#define AVX512_FV_LOAD1(DST, VAL) \
	({ \
		(DST) = _mm512_set1_ps((VAL)); \
	})

#define AVX512_FV_STORE(DST, SRC) \
	({ \
		_mm512_storeu_ps((DST + 0), (SRC)); \
	})

#define AVX512_FV_ADD(DST, LHS, RHS) \
	({ \
		(DST) = _mm512_add_ps((LHS), (RHS)); \
	})

#define AVX512_FV_SUB(DST, LHS, RHS) \
	({ \
		(DST) = _mm512_sub_ps((LHS), (RHS)); \
	})

#define AVX512_FV_MUL(DST, LHS, RHS) \
	({ \
		(DST) = _mm512_mul_ps((LHS), (RHS)); \
	})

#define AVX512_FV_DIV(DST, LHS, RHS) \
	({ \
		(DST) = _mm512_div_ps((LHS), (RHS)); \
	})

#define AVX512_FV_EXP(DST, LHS) \
	({ \
		FT_TYPE tmp[AVX512_N]; \
		AVX512_FV_STORE(tmp, LHS); \
		tmp[0] = expf(tmp[0]); \
		tmp[1] = expf(tmp[1]); \
		tmp[2] = expf(tmp[2]); \
		tmp[3] = expf(tmp[3]); \
		tmp[4] = expf(tmp[4]); \
		tmp[5] = expf(tmp[5]); \
		tmp[6] = expf(tmp[6]); \
		tmp[7] = expf(tmp[7]); \
		tmp[8] = expf(tmp[8]); \
		tmp[9] = expf(tmp[9]); \
		tmp[10] = expf(tmp[10]); \
		tmp[11] = expf(tmp[11]); \
		tmp[12] = expf(tmp[12]); \
		tmp[13] = expf(tmp[13]); \
		tmp[14] = expf(tmp[14]); \
		tmp[15] = expf(tmp[15]); \
		AVX512_FV_LOAD(DST, tmp); \
	})

#define AVX512_FV_TANH(DST, LHS) \
	({ \
		FT_TYPE tmp[AVX512_N]; \
		AVX512_FV_STORE(tmp, LHS); \
		tmp[0] = tanhf(tmp[0]); \
		tmp[1] = tanhf(tmp[1]); \
		tmp[2] = tanhf(tmp[2]); \
		tmp[3] = tanhf(tmp[3]); \
		tmp[4] = tanhf(tmp[4]); \
		tmp[5] = tanhf(tmp[5]); \
		tmp[6] = tanhf(tmp[6]); \
		tmp[7] = tanhf(tmp[7]); \
		tmp[8] = tanhf(tmp[8]); \
		tmp[9] = tanhf(tmp[9]); \
		tmp[10] = tanhf(tmp[10]); \
		tmp[11] = tanhf(tmp[11]); \
		tmp[12] = tanhf(tmp[12]); \
		tmp[13] = tanhf(tmp[13]); \
		tmp[14] = tanhf(tmp[14]); \
		tmp[15] = tanhf(tmp[15]); \
		AVX512_FV_LOAD(DST, tmp); \
	})

#define AVX512_FV_REDUCE_SUM(LHS) \
	({ \
        __m256 v5 = _mm256_add_ps(_mm512_extractf32x8_ps((LHS), 0), \
                                  _mm512_extractf32x8_ps((LHS), 1)); \
        __m128 v4 = _mm_add_ps(_mm256_extractf128_ps(v5, 0), \
                               _mm256_extractf128_ps(v5, 1)); \
		__m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4)); \
		__m128 v1 = _mm_add_ss(v2, _mm_movehdup_ps(v2)); \
		_mm_cvtss_f32(v1); \
	})

#define AVX512_FV_REDUCE_MAX(LHS) \
	({ \
        _mm512_reduce_max_ps((LHS)); \
	})
