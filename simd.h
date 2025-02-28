#pragma once

#include <string.h>

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
#define FV_LOAD			AVX512_FV_LOAD
#define FV_LOAD1		AVX512_FV_LOAD1
#define FV_STORE		AVX512_FV_STORE
#define FV_ADD			AVX512_FV_ADD
#define FV_SUB			AVX512_FV_SUB
#define FV_MUL			AVX512_FV_MUL
#define FV_DIV			AVX512_FV_DIV
#define FV_EXP			AVX512_FV_EXP
#define FV_TANH			AVX512_FV_TANH
#define FV_REDUCE_SUM	AVX512_FV_REDUCE_SUM
#elif defined(__AVX2__)
typedef avx2_fv_t fv_t;
#define FT_N            AVX2_N
#define FV_LOAD			AVX2_FV_LOAD
#define FV_LOAD1		AVX2_FV_LOAD1
#define FV_STORE		AVX2_FV_STORE
#define FV_ADD			AVX2_FV_ADD
#define FV_SUB			AVX2_FV_SUB
#define FV_MUL			AVX2_FV_MUL
#define FV_DIV			AVX2_FV_DIV
#define FV_EXP			AVX2_FV_EXP
#define FV_TANH			AVX2_FV_TANH
#define FV_REDUCE_SUM	AVX2_FV_REDUCE_SUM
#else
typedef cpu_fv_t fv_t;
#define FT_N            CPU_N
#define FV_LOAD			CPU_FV_LOAD
#define FV_LOAD1		CPU_FV_LOAD1
#define FV_STORE		CPU_FV_STORE
#define FV_ADD			CPU_FV_ADD
#define FV_SUB			CPU_FV_SUB
#define FV_MUL			CPU_FV_MUL
#define FV_DIV			CPU_FV_DIV
#define FV_EXP			CPU_FV_EXP
#define FV_TANH			CPU_FV_TANH
#define FV_REDUCE_SUM		CPU_FV_REDUCE_SUM
#endif
