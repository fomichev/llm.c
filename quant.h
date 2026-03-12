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
