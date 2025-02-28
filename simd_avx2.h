#pragma once

#include <immintrin.h>

/* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html */

typedef __m256 avx_fv_t;

#define AVX2_FV_LOAD(DST, SRC) \
	({ \
		(DST) = _mm256_loadu_ps((SRC) + 0); \
	})

#define AVX2_FV_LOAD1(DST, VAL) \
	({ \
		(DST) = _mm256_set1_ps((VAL)); \
	})

#define AVX2_FV_STORE(DST, SRC) \
	({ \
		_mm256_storeu_ps((DST + 0), (SRC)); \
	})

#define AVX2_FV_ADD(DST, LHS, RHS) \
	({ \
		(DST) = _mm256_add_ps((LHS), (RHS)); \
	})

#define AVX2_FV_SUB(DST, LHS, RHS) \
	({ \
		(DST) = _mm256_sub_ps((LHS), (RHS)); \
	})

#define AVX2_FV_MUL(DST, LHS, RHS) \
	({ \
		(DST) = _mm256_mul_ps((LHS), (RHS)); \
	})

#define AVX2_FV_DIV(DST, LHS, RHS) \
	({ \
		(DST) = _mm256_div_ps((LHS), (RHS)); \
	})

#define AVX2_FV_EXP(DST, LHS) \
	({ \
		FT_TYPE tmp[16]; \
		FV_STORE(tmp, LHS); \
		tmp[0] = expf(tmp[0]); \
		tmp[1] = expf(tmp[1]); \
		tmp[2] = expf(tmp[2]); \
		tmp[3] = expf(tmp[3]); \
		tmp[4] = expf(tmp[4]); \
		tmp[5] = expf(tmp[5]); \
		tmp[6] = expf(tmp[6]); \
		tmp[7] = expf(tmp[7]); \
		FV_LOAD(DST, tmp); \
	})

#define AVX2_FV_TANH(DST, LHS) \
	({ \
		FT_TYPE tmp[16]; \
		FV_STORE(tmp, LHS); \
		tmp[0] = tanhf(tmp[0]); \
		tmp[1] = tanhf(tmp[1]); \
		tmp[2] = tanhf(tmp[2]); \
		tmp[3] = tanhf(tmp[3]); \
		tmp[4] = tanhf(tmp[4]); \
		tmp[5] = tanhf(tmp[5]); \
		tmp[6] = tanhf(tmp[6]); \
		tmp[7] = tanhf(tmp[7]); \
		FV_LOAD(DST, tmp); \
	})

#define AVX2_FV_REDUCE_SUM(LHS) \
	({ \
		__m128 v4 = _mm_add_ps(_mm256_castps256_ps128((LHS)), \
				        _mm256_extractf128_ps((LHS), 1)); \
		__m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4)); \
		__m128 v1 = _mm_add_ss(v2, _mm_movehdup_ps(v2)); \
		_mm_cvtss_f32(v1); \
	})
