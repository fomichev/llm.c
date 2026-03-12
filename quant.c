#include "quant.h"
#include "tensor.h"
#include "simd.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

_Static_assert(GGML_QK % VECTOR_BATCH == 0, "VECTOR_BATCH must evenly divide GGML_QK");

#if !defined(__F16C__)
float f16_to_f32(uint16_t h)
{
	uint32_t sign = (uint32_t)(h & 0x8000) << 16;
	uint32_t exp  = (h >> 10) & 0x1f;
	uint32_t mant = h & 0x3ff;
	uint32_t f;

	if (exp == 0) {
		if (mant == 0) {
			f = sign; /* +/- zero */
		} else {
			/* denormalized */
			exp = 1;
			while (!(mant & 0x400)) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3ff;
			f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
		}
	} else if (exp == 0x1f) {
		f = sign | 0x7f800000 | (mant << 13); /* inf/nan */
	} else {
		f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
	}

	float result;
	memcpy(&result, &f, sizeof(result));
	return result;
}
#endif

static float dot_f32_q8_0(const float *x, const block_q8_0 *y, size_t n)
{
	assert(n % GGML_QK == 0);
	size_t nb = n / GGML_QK;
	vector_t acc;
	vector_set(&acc, 0);

	for (size_t i = 0; i < nb; i++) {
		vector_t vscale;
		vector_set(&vscale, f16_to_f32(y[i].d));

		for (int g = 0; g < GGML_QK / VECTOR_BATCH; g++) {
			vector_t q, xv, sq;
			vector_i8_to_f32(&q, &y[i].qs[g * VECTOR_BATCH]);
			vector_load(&xv, (scalar_t *)&x[i * GGML_QK + g * VECTOR_BATCH]);
			vector_mul(&sq, &vscale, &q);
			vector_fma(&acc, &sq, &xv, &acc);
		}
	}

	return vector_reduce_sum(&acc);
}

static float dot_f32_q4_0(const float *x, const block_q4_0 *y, size_t n)
{
	assert(n % GGML_QK == 0);
	size_t nb = n / GGML_QK;
	vector_t acc;
	vector_set(&acc, 0);

	for (size_t i = 0; i < nb; i++) {
		vector_t vscale;
		vector_set(&vscale, f16_to_f32(y[i].d));

		for (int g = 0; g < GGML_QK / 2 / VECTOR_BATCH; g++) {
			vector_t lo, hi, xv_lo, xv_hi, sq;

			vector_u4_lo_to_f32(&lo, &y[i].qs[g * VECTOR_BATCH]);
			vector_u4_hi_to_f32(&hi, &y[i].qs[g * VECTOR_BATCH]);

			vector_load(&xv_lo, (scalar_t *)&x[i * GGML_QK + g * VECTOR_BATCH]);
			vector_load(&xv_hi, (scalar_t *)&x[i * GGML_QK + g * VECTOR_BATCH + GGML_QK / 2]);

			vector_mul(&sq, &vscale, &lo);
			vector_fma(&acc, &sq, &xv_lo, &acc);

			vector_mul(&sq, &vscale, &hi);
			vector_fma(&acc, &sq, &xv_hi, &acc);
		}
	}

	return vector_reduce_sum(&acc);
}

static void dequant_row_q8_0(const block_q8_0 *src, float *dst, size_t n)
{
	assert(n % GGML_QK == 0);
	size_t nb = n / GGML_QK;

	for (size_t i = 0; i < nb; i++) {
		vector_t vscale;
		vector_set(&vscale, f16_to_f32(src[i].d));

		for (int g = 0; g < GGML_QK / VECTOR_BATCH; g++) {
			vector_t q, result;
			vector_i8_to_f32(&q, &src[i].qs[g * VECTOR_BATCH]);
			vector_mul(&result, &vscale, &q);
			vector_store(&dst[i * GGML_QK + g * VECTOR_BATCH], &result);
		}
	}
}

static void dequant_row_q4_0(const block_q4_0 *src, float *dst, size_t n)
{
	assert(n % GGML_QK == 0);
	size_t nb = n / GGML_QK;

	for (size_t i = 0; i < nb; i++) {
		vector_t vscale;
		vector_set(&vscale, f16_to_f32(src[i].d));

		for (int g = 0; g < GGML_QK / 2 / VECTOR_BATCH; g++) {
			vector_t lo, hi, result;
			vector_u4_lo_to_f32(&lo, &src[i].qs[g * VECTOR_BATCH]);
			vector_u4_hi_to_f32(&hi, &src[i].qs[g * VECTOR_BATCH]);

			vector_mul(&result, &vscale, &lo);
			vector_store(&dst[i * GGML_QK + g * VECTOR_BATCH], &result);

			vector_mul(&result, &vscale, &hi);
			vector_store(&dst[i * GGML_QK + g * VECTOR_BATCH + GGML_QK / 2], &result);
		}
	}
}

void dequant_row(const void *qdata, int type, size_t row, float *dst, size_t n)
{
	size_t blocks_per_row = n / GGML_QK;

	assert(n % GGML_QK == 0);

	switch (type) {
	case TENSOR_Q8_0: {
		const block_q8_0 *data = (const block_q8_0 *)qdata;
		dequant_row_q8_0(&data[row * blocks_per_row], dst, n);
		break;
	}
	case TENSOR_Q4_0: {
		const block_q4_0 *data = (const block_q4_0 *)qdata;
		dequant_row_q4_0(&data[row * blocks_per_row], dst, n);
		break;
	}
	default:
		fprintf(stderr, "dequant_row: unsupported type %d\n", type);
		abort();
	}
}

float dot_f32_quant(const float *x, const void *qdata, int type, size_t row, size_t n)
{
	size_t blocks_per_row = n / GGML_QK;

	assert(n % GGML_QK == 0);

	switch (type) {
	case TENSOR_Q8_0: {
		const block_q8_0 *data = (const block_q8_0 *)qdata;
		return dot_f32_q8_0(x, &data[row * blocks_per_row], n);
		break;
	}
	case TENSOR_Q4_0: {
		const block_q4_0 *data = (const block_q4_0 *)qdata;
		return dot_f32_q4_0(x, &data[row * blocks_per_row], n);
		break;
	}
	default:
		fprintf(stderr, "dot_f32_quant: unsupported type %d\n", type);
		abort();
	}
}
