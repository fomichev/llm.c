#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__F16C__)
#include <immintrin.h>
#endif

#define GGML_QK 32 /* block size for Q4_0 and Q8_0 */

/* Q8_0 block: 2-byte f16 scale + 32 x int8 = 34 bytes */
typedef struct {
	uint16_t d;       /* f16 scale */
	int8_t   qs[GGML_QK]; /* quantized values */
} __attribute__((packed)) block_q8_0;

/* Q4_0 block: 2-byte f16 scale + 16 x packed nibble pairs = 18 bytes */
typedef struct {
	uint16_t d;               /* f16 scale */
	uint8_t  qs[GGML_QK / 2]; /* packed 4-bit quantized values */
} __attribute__((packed)) block_q4_0;

_Static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size");
_Static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size");

#define QK_K 256

/* Q4_K block: 256 elements, 8 sub-blocks of 32 with 6-bit scales/mins */
typedef struct {
	uint16_t d;                  /* f16 super-block scale */
	uint16_t dmin;               /* f16 super-block min */
	uint8_t  scales[12];         /* 8x 6-bit scales + 8x 6-bit mins, packed */
	uint8_t  qs[QK_K / 2];      /* 128 bytes: 4-bit unsigned values, 2 per byte */
} __attribute__((packed)) block_q4_K;

/* Q5_K block: 256 elements, like Q4_K but with 5th bit in qh */
typedef struct {
	uint16_t d;                  /* f16 super-block scale */
	uint16_t dmin;               /* f16 super-block min */
	uint8_t  scales[12];         /* same packing as Q4_K */
	uint8_t  qh[QK_K / 8];      /* 32 bytes: high bits (1 bit per value) */
	uint8_t  qs[QK_K / 2];      /* 128 bytes: low 4 bits */
} __attribute__((packed)) block_q5_K;

/* Q6_K block: 256 elements, symmetric 6-bit quantization */
typedef struct {
	uint8_t  ql[QK_K / 2];      /* 128 bytes: lower 4 bits of 6-bit quants */
	uint8_t  qh[QK_K / 4];      /* 64 bytes: upper 2 bits of 6-bit quants */
	int8_t   scales[QK_K / 16]; /* 16 bytes: 8-bit signed scales (one per 16 elements) */
	uint16_t d;                  /* f16 super-block scale */
} __attribute__((packed)) block_q6_K;

_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K size");
_Static_assert(sizeof(block_q5_K) == 176, "block_q5_K size");
_Static_assert(sizeof(block_q6_K) == 210, "block_q6_K size");

/* Unpack one of the 8 sub-block scales and mins from Q4_K/Q5_K packed format */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
	if (j < 4) {
		*d = q[j] & 63;
		*m = q[j + 4] & 63;
	} else {
		*d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
	}
}

#if defined(__F16C__)
static inline float f16_to_f32(uint16_t h)
{
	return _cvtsh_ss(h);
}
#else
float f16_to_f32(uint16_t h);
#endif

void dequant_row(const void *qdata, int type, size_t row, float *dst, size_t n);
float dot_f32_quant(const float *x, const void *qdata, int type, size_t row, size_t n);
