#include "quant.h"
#include "tensor.h"
#include "simd.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

_Static_assert(GGML_QK % VECTOR_BATCH == 0, "VECTOR_BATCH must evenly divide GGML_QK");
_Static_assert(QK_K % VECTOR_BATCH == 0, "VECTOR_BATCH must evenly divide QK_K");

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

static float dot_f32_q4_K(const float *x, const block_q4_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	vector_t acc;
	vector_set(&acc, 0);

	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(y[i].d);
		float dmin = f16_to_f32(y[i].dmin);

		const uint8_t *q = y[i].qs;
		size_t x_off = i * QK_K;

		for (int j = 0; j < QK_K / 64; j++) {
			uint8_t sc1, m1, sc2, m2;
			get_scale_min_k4(2 * j, y[i].scales, &sc1, &m1);
			get_scale_min_k4(2 * j + 1, y[i].scales, &sc2, &m2);

			float d_sc1 = d * sc1;
			float dmin_m1 = dmin * m1;
			float d_sc2 = d * sc2;
			float dmin_m2 = dmin * m2;

			vector_t vscale1, vmin1, vscale2, vmin2;
			vector_set(&vscale1, d_sc1);
			vector_set(&vmin1, dmin_m1);
			vector_set(&vscale2, d_sc2);
			vector_set(&vmin2, dmin_m2);

			/* low nibble: 32 elements with scale d*sc1, min dmin*m1 */
			for (int g = 0; g < 32 / VECTOR_BATCH; g++) {
				vector_t lo, xv, sq;
				vector_u4_lo_to_f32_unsigned(&lo, &q[j * 32 + g * VECTOR_BATCH]);
				vector_load(&xv, (scalar_t *)&x[x_off + j * 64 + g * VECTOR_BATCH]);
				vector_mul(&sq, &vscale1, &lo);
				vector_sub(&sq, &sq, &vmin1);
				vector_fma(&acc, &sq, &xv, &acc);
			}

			/* high nibble: 32 elements with scale d*sc2, min dmin*m2 */
			for (int g = 0; g < 32 / VECTOR_BATCH; g++) {
				vector_t hi, xv, sq;
				vector_u4_hi_to_f32_unsigned(&hi, &q[j * 32 + g * VECTOR_BATCH]);
				vector_load(&xv, (scalar_t *)&x[x_off + j * 64 + 32 + g * VECTOR_BATCH]);
				vector_mul(&sq, &vscale2, &hi);
				vector_sub(&sq, &sq, &vmin2);
				vector_fma(&acc, &sq, &xv, &acc);
			}
		}
	}

	return vector_reduce_sum(&acc);
}

static float dot_f32_q5_K(const float *x, const block_q5_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	float sum = 0.0f;

	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(y[i].d);
		float dmin = f16_to_f32(y[i].dmin);

		const uint8_t *ql = y[i].qs;
		const uint8_t *qh = y[i].qh;
		size_t x_off = i * QK_K;

		for (int j = 0; j < QK_K / 64; j++) {
			uint8_t sc1, m1, sc2, m2;
			get_scale_min_k4(2 * j, y[i].scales, &sc1, &m1);
			get_scale_min_k4(2 * j + 1, y[i].scales, &sc2, &m2);

			float d_sc1 = d * sc1;
			float dmin_m1 = dmin * m1;
			float d_sc2 = d * sc2;
			float dmin_m2 = dmin * m2;

			/* low nibble: 32 elements */
			for (int k = 0; k < 32; k++) {
				uint8_t lo4 = ql[j * 32 + k] & 0x0F;
				uint8_t hi1 = (qh[k] >> (j * 2)) & 1;
				int val = lo4 | (hi1 << 4);
				sum += (d_sc1 * val - dmin_m1) * x[x_off + j * 64 + k];
			}

			/* high nibble: 32 elements */
			for (int k = 0; k < 32; k++) {
				uint8_t hi4 = ql[j * 32 + k] >> 4;
				uint8_t hi1 = (qh[k] >> (j * 2 + 1)) & 1;
				int val = hi4 | (hi1 << 4);
				sum += (d_sc2 * val - dmin_m2) * x[x_off + j * 64 + 32 + k];
			}
		}
	}

	return sum;
}

static float dot_f32_q6_K(const float *x, const block_q6_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	float sum = 0.0f;

	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(y[i].d);
		size_t x_off = i * QK_K;

		for (int chunk = 0; chunk < 2; chunk++) {
			const uint8_t *ql = y[i].ql + chunk * 64;
			const uint8_t *qh = y[i].qh + chunk * 32;
			int n_128 = chunk * 128;

			for (int l = 0; l < 32; l++) {
				int is = n_128 / 16 + l / 16;
				int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				sum += d * y[i].scales[is + 0] * q1 * x[x_off + n_128 + l + 0];
				sum += d * y[i].scales[is + 2] * q2 * x[x_off + n_128 + l + 32];
				sum += d * y[i].scales[is + 4] * q3 * x[x_off + n_128 + l + 64];
				sum += d * y[i].scales[is + 6] * q4 * x[x_off + n_128 + l + 96];
			}
		}
	}

	return sum;
}

static void dequant_row_q4_K(const block_q4_K *src, float *dst, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;

	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(src[i].d);
		float dmin = f16_to_f32(src[i].dmin);

		const uint8_t *q = src[i].qs;

		for (int j = 0; j < QK_K / 64; j++) {
			uint8_t sc1, m1, sc2, m2;
			get_scale_min_k4(2 * j, src[i].scales, &sc1, &m1);
			get_scale_min_k4(2 * j + 1, src[i].scales, &sc2, &m2);

			float d_sc1 = d * sc1;
			float dmin_m1 = dmin * m1;
			float d_sc2 = d * sc2;
			float dmin_m2 = dmin * m2;

			vector_t vscale1, vmin1, vscale2, vmin2;
			vector_set(&vscale1, d_sc1);
			vector_set(&vmin1, dmin_m1);
			vector_set(&vscale2, d_sc2);
			vector_set(&vmin2, dmin_m2);

			for (int g = 0; g < 32 / VECTOR_BATCH; g++) {
				vector_t lo, result;
				vector_u4_lo_to_f32_unsigned(&lo, &q[j * 32 + g * VECTOR_BATCH]);
				vector_mul(&result, &vscale1, &lo);
				vector_sub(&result, &result, &vmin1);
				vector_store(&dst[i * QK_K + j * 64 + g * VECTOR_BATCH], &result);
			}

			for (int g = 0; g < 32 / VECTOR_BATCH; g++) {
				vector_t hi, result;
				vector_u4_hi_to_f32_unsigned(&hi, &q[j * 32 + g * VECTOR_BATCH]);
				vector_mul(&result, &vscale2, &hi);
				vector_sub(&result, &result, &vmin2);
				vector_store(&dst[i * QK_K + j * 64 + 32 + g * VECTOR_BATCH], &result);
			}
		}
	}
}

static void dequant_row_q5_K(const block_q5_K *src, float *dst, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;

	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(src[i].d);
		float dmin = f16_to_f32(src[i].dmin);

		const uint8_t *ql = src[i].qs;
		const uint8_t *qh = src[i].qh;

		for (int j = 0; j < QK_K / 64; j++) {
			uint8_t sc1, m1, sc2, m2;
			get_scale_min_k4(2 * j, src[i].scales, &sc1, &m1);
			get_scale_min_k4(2 * j + 1, src[i].scales, &sc2, &m2);

			float d_sc1 = d * sc1;
			float dmin_m1 = dmin * m1;
			float d_sc2 = d * sc2;
			float dmin_m2 = dmin * m2;

			for (int k = 0; k < 32; k++) {
				uint8_t lo4 = ql[j * 32 + k] & 0x0F;
				uint8_t hi1 = (qh[k] >> (j * 2)) & 1;
				int val = lo4 | (hi1 << 4);
				dst[i * QK_K + j * 64 + k] = d_sc1 * val - dmin_m1;
			}

			for (int k = 0; k < 32; k++) {
				uint8_t hi4 = ql[j * 32 + k] >> 4;
				uint8_t hi1 = (qh[k] >> (j * 2 + 1)) & 1;
				int val = hi4 | (hi1 << 4);
				dst[i * QK_K + j * 64 + 32 + k] = d_sc2 * val - dmin_m2;
			}
		}
	}
}

static void dequant_row_q6_K(const block_q6_K *src, float *dst, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;

	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(src[i].d);

		for (int chunk = 0; chunk < 2; chunk++) {
			const uint8_t *ql = src[i].ql + chunk * 64;
			const uint8_t *qh = src[i].qh + chunk * 32;
			int n_128 = chunk * 128;

			for (int l = 0; l < 32; l++) {
				int is = n_128 / 16 + l / 16;
				int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				dst[i * QK_K + n_128 + l + 0]  = d * src[i].scales[is + 0] * q1;
				dst[i * QK_K + n_128 + l + 32] = d * src[i].scales[is + 2] * q2;
				dst[i * QK_K + n_128 + l + 64] = d * src[i].scales[is + 4] * q3;
				dst[i * QK_K + n_128 + l + 96] = d * src[i].scales[is + 6] * q4;
			}
		}
	}
}

void dequant_row(const void *qdata, int type, size_t row, float *dst, size_t n)
{
	switch (type) {
	case TENSOR_Q8_0: {
		size_t blocks_per_row = n / GGML_QK;
		assert(n % GGML_QK == 0);
		const block_q8_0 *data = (const block_q8_0 *)qdata;
		dequant_row_q8_0(&data[row * blocks_per_row], dst, n);
		break;
	}
	case TENSOR_Q4_0: {
		size_t blocks_per_row = n / GGML_QK;
		assert(n % GGML_QK == 0);
		const block_q4_0 *data = (const block_q4_0 *)qdata;
		dequant_row_q4_0(&data[row * blocks_per_row], dst, n);
		break;
	}
	case TENSOR_Q4_K: {
		size_t blocks_per_row = n / QK_K;
		assert(n % QK_K == 0);
		const block_q4_K *data = (const block_q4_K *)qdata;
		dequant_row_q4_K(&data[row * blocks_per_row], dst, n);
		break;
	}
	case TENSOR_Q5_K: {
		size_t blocks_per_row = n / QK_K;
		assert(n % QK_K == 0);
		const block_q5_K *data = (const block_q5_K *)qdata;
		dequant_row_q5_K(&data[row * blocks_per_row], dst, n);
		break;
	}
	case TENSOR_Q6_K: {
		size_t blocks_per_row = n / QK_K;
		assert(n % QK_K == 0);
		const block_q6_K *data = (const block_q6_K *)qdata;
		dequant_row_q6_K(&data[row * blocks_per_row], dst, n);
		break;
	}
	default:
		fprintf(stderr, "dequant_row: unsupported type %d\n", type);
		abort();
	}
}

float dot_f32_quant(const float *x, const void *qdata, int type, size_t row, size_t n)
{
	switch (type) {
	case TENSOR_Q8_0: {
		size_t blocks_per_row = n / GGML_QK;
		assert(n % GGML_QK == 0);
		const block_q8_0 *data = (const block_q8_0 *)qdata;
		return dot_f32_q8_0(x, &data[row * blocks_per_row], n);
	}
	case TENSOR_Q4_0: {
		size_t blocks_per_row = n / GGML_QK;
		assert(n % GGML_QK == 0);
		const block_q4_0 *data = (const block_q4_0 *)qdata;
		return dot_f32_q4_0(x, &data[row * blocks_per_row], n);
	}
	case TENSOR_Q4_K: {
		size_t blocks_per_row = n / QK_K;
		assert(n % QK_K == 0);
		const block_q4_K *data = (const block_q4_K *)qdata;
		return dot_f32_q4_K(x, &data[row * blocks_per_row], n);
	}
	case TENSOR_Q5_K: {
		size_t blocks_per_row = n / QK_K;
		assert(n % QK_K == 0);
		const block_q5_K *data = (const block_q5_K *)qdata;
		return dot_f32_q5_K(x, &data[row * blocks_per_row], n);
	}
	case TENSOR_Q6_K: {
		size_t blocks_per_row = n / QK_K;
		assert(n % QK_K == 0);
		const block_q6_K *data = (const block_q6_K *)qdata;
		return dot_f32_q6_K(x, &data[row * blocks_per_row], n);
	}
	default:
		fprintf(stderr, "dot_f32_quant: unsupported type %d\n", type);
		abort();
	}
}
