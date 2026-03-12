#include "tensor.h"
#include "quant.h"

#include <stdbool.h>
#include <math.h>
#include "test/test.h"

static void quantize_q8_0_block(const float *src, block_q8_0 *dst)
{
	float amax = 0.0f;
	for (int j = 0; j < GGML_QK; j++) {
		float av = fabsf(src[j]);
		if (av > amax) amax = av;
	}

	float scale = amax / 127.0f;

	/* f32 to f16 */
	uint32_t fs;
	memcpy(&fs, &scale, sizeof(fs));
	uint32_t sign = (fs >> 16) & 0x8000;
	uint32_t exp  = ((fs >> 23) & 0xff);
	uint32_t mant = fs & 0x7fffff;
	uint16_t h;
	if (exp == 0) h = sign;
	else if (exp == 0xff) h = sign | 0x7c00 | (mant ? 0x200 : 0);
	else {
		int newexp = (int)exp - 127 + 15;
		if (newexp >= 0x1f) h = sign | 0x7c00;
		else if (newexp <= 0) h = sign;
		else h = sign | (newexp << 10) | (mant >> 13);
	}
	dst->d = h;

	float inv_scale = scale > 0 ? 127.0f / amax : 0.0f;
	for (int j = 0; j < GGML_QK; j++) {
		int v = (int)roundf(src[j] * inv_scale);
		if (v > 127) v = 127;
		if (v < -128) v = -128;
		dst->qs[j] = (int8_t)v;
	}
}

static void quantize_q4_0_block(const float *src, block_q4_0 *dst)
{
	float amax = 0.0f;
	for (int j = 0; j < GGML_QK; j++) {
		float av = fabsf(src[j]);
		if (av > amax) amax = av;
	}

	float scale = amax / 7.0f;

	uint32_t fs;
	memcpy(&fs, &scale, sizeof(fs));
	uint32_t sign = (fs >> 16) & 0x8000;
	uint32_t exp  = ((fs >> 23) & 0xff);
	uint32_t mant = fs & 0x7fffff;
	uint16_t h;
	if (exp == 0) h = sign;
	else if (exp == 0xff) h = sign | 0x7c00 | (mant ? 0x200 : 0);
	else {
		int newexp = (int)exp - 127 + 15;
		if (newexp >= 0x1f) h = sign | 0x7c00;
		else if (newexp <= 0) h = sign;
		else h = sign | (newexp << 10) | (mant >> 13);
	}
	dst->d = h;

	float inv_scale = scale > 0 ? 7.0f / amax : 0.0f;
	for (int j = 0; j < GGML_QK / 2; j++) {
		int lo = (int)roundf(src[j] * inv_scale) + 8;
		int hi = (int)roundf(src[j + GGML_QK / 2] * inv_scale) + 8;
		if (lo < 0) lo = 0; if (lo > 15) lo = 15;
		if (hi < 0) hi = 0; if (hi > 15) hi = 15;
		dst->qs[j] = (uint8_t)(lo | (hi << 4));
	}
}

static tensor_t *make_q8_0_tensor(const tensor_t *src)
{
	size_t rows = src->dim[0];
	size_t cols = src->dim[1];
	size_t blocks_per_row = cols / GGML_QK;
	size_t total_blocks = rows * blocks_per_row;

	block_q8_0 *blocks = calloc(total_blocks, sizeof(block_q8_0));
	assert(blocks);

	for (size_t r = 0; r < rows; r++)
		for (size_t b = 0; b < blocks_per_row; b++)
			quantize_q8_0_block(&src->data[r * cols + b * GGML_QK],
					    &blocks[r * blocks_per_row + b]);

	tensor_t *t = tensor_new_mapped(blocks, rows * cols, TENSOR_Q8_0);
	tensor_reshape_2d(t, rows, cols);
	return t;
}

static tensor_t *make_q4_0_tensor(const tensor_t *src)
{
	size_t rows = src->dim[0];
	size_t cols = src->dim[1];
	size_t blocks_per_row = cols / GGML_QK;
	size_t total_blocks = rows * blocks_per_row;

	block_q4_0 *blocks = calloc(total_blocks, sizeof(block_q4_0));
	assert(blocks);

	for (size_t r = 0; r < rows; r++)
		for (size_t b = 0; b < blocks_per_row; b++)
			quantize_q4_0_block(&src->data[r * cols + b * GGML_QK],
					    &blocks[r * blocks_per_row + b]);

	tensor_t *t = tensor_new_mapped(blocks, rows * cols, TENSOR_Q4_0);
	tensor_reshape_2d(t, rows, cols);
	return t;
}

static float ref_dot_q8_0(const float *x, const block_q8_0 *y, size_t n)
{
	size_t nb = n / GGML_QK;
	float sum = 0.0f;
	for (size_t i = 0; i < nb; i++) {
		float scale = f16_to_f32(y[i].d);
		for (int j = 0; j < GGML_QK; j++)
			sum += scale * (float)y[i].qs[j] * x[i * GGML_QK + j];
	}
	return sum;
}

static float ref_dot_q4_0(const float *x, const block_q4_0 *y, size_t n)
{
	size_t nb = n / GGML_QK;
	float sum = 0.0f;
	for (size_t i = 0; i < nb; i++) {
		float scale = f16_to_f32(y[i].d);
		for (int j = 0; j < GGML_QK / 2; j++) {
			int lo = (y[i].qs[j] & 0x0F) - 8;
			int hi = (y[i].qs[j] >> 4) - 8;
			sum += scale * (float)lo * x[i * GGML_QK + j];
			sum += scale * (float)hi * x[i * GGML_QK + j + GGML_QK / 2];
		}
	}
	return sum;
}

static void ref_dequant_q8_0(const block_q8_0 *src, float *dst, size_t n)
{
	size_t nb = n / GGML_QK;
	for (size_t i = 0; i < nb; i++) {
		float scale = f16_to_f32(src[i].d);
		for (int j = 0; j < GGML_QK; j++)
			dst[i * GGML_QK + j] = scale * (float)src[i].qs[j];
	}
}

static void ref_dequant_q4_0(const block_q4_0 *src, float *dst, size_t n)
{
	size_t nb = n / GGML_QK;
	for (size_t i = 0; i < nb; i++) {
		float scale = f16_to_f32(src[i].d);
		for (int j = 0; j < GGML_QK / 2; j++) {
			int lo = (src[i].qs[j] & 0x0F) - 8;
			int hi = (src[i].qs[j] >> 4) - 8;
			dst[i * GGML_QK + j]              = scale * (float)lo;
			dst[i * GGML_QK + j + GGML_QK / 2] = scale * (float)hi;
		}
	}
}

static void assert_close(float a, float b, float tol, const char *msg)
{
	float diff = fabsf(a - b);
	if (diff > tol) {
		fprintf(stderr, "%s: %f != %f (diff=%f, tol=%f)\n", msg, a, b, diff, tol);
		abort();
	}
}

static void test_dequant_q8_0(void)
{
	size_t sizes[] = { GGML_QK, 2 * GGML_QK, 4 * GGML_QK };

	for (int s = 0; s < 3; s++) {
		size_t n = sizes[s];
		size_t nb = n / GGML_QK;

		float src[4 * GGML_QK];
		for (size_t i = 0; i < n; i++)
			src[i] = (float)i * 0.1f - (float)n * 0.05f;

		block_q8_0 blocks[4];
		for (size_t i = 0; i < nb; i++)
			quantize_q8_0_block(&src[i * GGML_QK], &blocks[i]);

		float expected[4 * GGML_QK];
		ref_dequant_q8_0(blocks, expected, n);

		float got[4 * GGML_QK];
		dequant_row(blocks, TENSOR_Q8_0, 0, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_q8_0");
	}
}

static void test_dequant_q4_0(void)
{
	size_t sizes[] = { GGML_QK, 2 * GGML_QK, 4 * GGML_QK };

	for (int s = 0; s < 3; s++) {
		size_t n = sizes[s];
		size_t nb = n / GGML_QK;

		float src[4 * GGML_QK];
		for (size_t i = 0; i < n; i++)
			src[i] = (float)i * 0.1f - (float)n * 0.05f;

		block_q4_0 blocks[4];
		for (size_t i = 0; i < nb; i++)
			quantize_q4_0_block(&src[i * GGML_QK], &blocks[i]);

		float expected[4 * GGML_QK];
		ref_dequant_q4_0(blocks, expected, n);

		float got[4 * GGML_QK];
		dequant_row(blocks, TENSOR_Q4_0, 0, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_q4_0");
	}
}

static void test_dot_q8_0(void)
{
	size_t sizes[] = { GGML_QK, 2 * GGML_QK, 4 * GGML_QK };

	for (int s = 0; s < 3; s++) {
		size_t n = sizes[s];
		size_t nb = n / GGML_QK;

		float x[4 * GGML_QK];
		float w[4 * GGML_QK];
		for (size_t i = 0; i < n; i++) {
			x[i] = (float)i * 0.01f - 0.5f;
			w[i] = (float)(n - i) * 0.02f - 0.3f;
		}

		block_q8_0 blocks[4];
		for (size_t i = 0; i < nb; i++)
			quantize_q8_0_block(&w[i * GGML_QK], &blocks[i]);

		float expected = ref_dot_q8_0(x, blocks, n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q8_0, 0, n);

		assert_close(got, expected, 1e-4f, "dot_q8_0");
	}
}

static void test_dot_q4_0(void)
{
	size_t sizes[] = { GGML_QK, 2 * GGML_QK, 4 * GGML_QK };

	for (int s = 0; s < 3; s++) {
		size_t n = sizes[s];
		size_t nb = n / GGML_QK;

		float x[4 * GGML_QK];
		float w[4 * GGML_QK];
		for (size_t i = 0; i < n; i++) {
			x[i] = (float)i * 0.01f - 0.5f;
			w[i] = (float)(n - i) * 0.02f - 0.3f;
		}

		block_q4_0 blocks[4];
		for (size_t i = 0; i < nb; i++)
			quantize_q4_0_block(&w[i * GGML_QK], &blocks[i]);

		float expected = ref_dot_q4_0(x, blocks, n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q4_0, 0, n);

		assert_close(got, expected, 1e-4f, "dot_q4_0");
	}
}

static void test_dequant_row_indexing_q8_0(void)
{
	size_t n = 2 * GGML_QK;
	size_t nb = n / GGML_QK;
	size_t nrows = 3;

	float src[3][2 * GGML_QK];
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r][i] = (float)(r * 100 + i) * 0.01f;

	block_q8_0 blocks[3 * 2];
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q8_0_block(&src[r][b * GGML_QK], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected[2 * GGML_QK];
		ref_dequant_q8_0(&blocks[r * nb], expected, n);

		float got[2 * GGML_QK];
		dequant_row(blocks, TENSOR_Q8_0, r, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_row_indexing q8_0");
	}
}

static void test_dequant_row_indexing_q4_0(void)
{
	size_t n = 2 * GGML_QK;
	size_t nb = n / GGML_QK;
	size_t nrows = 3;

	float src[3][2 * GGML_QK];
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r][i] = (float)(r * 100 + i) * 0.01f;

	block_q4_0 blocks[3 * 2];
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q4_0_block(&src[r][b * GGML_QK], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected[2 * GGML_QK];
		ref_dequant_q4_0(&blocks[r * nb], expected, n);

		float got[2 * GGML_QK];
		dequant_row(blocks, TENSOR_Q4_0, r, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_row_indexing q4_0");
	}
}

static void test_dot_row_indexing_q8_0(void)
{
	size_t n = 2 * GGML_QK;
	size_t nb = n / GGML_QK;
	size_t nrows = 3;

	float x[2 * GGML_QK];
	for (size_t i = 0; i < n; i++)
		x[i] = (float)i * 0.01f;

	float src[3][2 * GGML_QK];
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r][i] = (float)(r * 100 + i) * 0.02f - 1.0f;

	block_q8_0 blocks[3 * 2];
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q8_0_block(&src[r][b * GGML_QK], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected = ref_dot_q8_0(x, &blocks[r * nb], n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q8_0, r, n);
		assert_close(got, expected, 1e-4f, "dot_row_indexing q8_0");
	}
}

static void test_dot_row_indexing_q4_0(void)
{
	size_t n = 2 * GGML_QK;
	size_t nb = n / GGML_QK;
	size_t nrows = 3;

	float x[2 * GGML_QK];
	for (size_t i = 0; i < n; i++)
		x[i] = (float)i * 0.01f;

	float src[3][2 * GGML_QK];
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r][i] = (float)(r * 100 + i) * 0.02f - 1.0f;

	block_q4_0 blocks[3 * 2];
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q4_0_block(&src[r][b * GGML_QK], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected = ref_dot_q4_0(x, &blocks[r * nb], n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q4_0, r, n);
		assert_close(got, expected, 1e-4f, "dot_row_indexing q4_0");
	}
}

static uint16_t f32_to_f16(float val)
{
	uint32_t fs;
	memcpy(&fs, &val, sizeof(fs));
	uint32_t sign = (fs >> 16) & 0x8000;
	uint32_t exp  = ((fs >> 23) & 0xff);
	uint32_t mant = fs & 0x7fffff;
	uint16_t h;
	if (exp == 0) h = sign;
	else if (exp == 0xff) h = sign | 0x7c00 | (mant ? 0x200 : 0);
	else {
		int newexp = (int)exp - 127 + 15;
		if (newexp >= 0x1f) h = sign | 0x7c00;
		else if (newexp <= 0) h = sign;
		else h = sign | (newexp << 10) | (mant >> 13);
	}
	return h;
}

static void quantize_q4_K_block(const float *src, block_q4_K *dst)
{
	memset(dst, 0, sizeof(*dst));

	/* find per-sub-block scale and min */
	float sc[8], mn[8];
	for (int sb = 0; sb < 8; sb++) {
		float smin = src[sb * 32], smax = src[sb * 32];
		for (int k = 1; k < 32; k++) {
			float v = src[sb * 32 + k];
			if (v < smin) smin = v;
			if (v > smax) smax = v;
		}
		mn[sb] = smin < 0 ? -smin : 0;
		sc[sb] = (smax + mn[sb]) / 15.0f;
		if (sc[sb] < 1e-10f) sc[sb] = 1e-10f;
	}

	/* find super-block scale and dmin */
	float max_sc = 0, max_mn = 0;
	for (int sb = 0; sb < 8; sb++) {
		if (sc[sb] > max_sc) max_sc = sc[sb];
		if (mn[sb] > max_mn) max_mn = mn[sb];
	}
	float d = max_sc / 63.0f;
	float dmin = max_mn / 63.0f;
	if (d < 1e-10f) d = 1e-10f;
	if (dmin < 1e-10f) dmin = 1e-10f;

	dst->d = f32_to_f16(d);
	dst->dmin = f32_to_f16(dmin);

	/* quantize sub-block scales and mins to 6-bit */
	uint8_t qsc[8], qmn[8];
	for (int sb = 0; sb < 8; sb++) {
		qsc[sb] = (uint8_t)(sc[sb] / d + 0.5f);
		if (qsc[sb] > 63) qsc[sb] = 63;
		qmn[sb] = (uint8_t)(mn[sb] / dmin + 0.5f);
		if (qmn[sb] > 63) qmn[sb] = 63;
	}

	/* pack scales into 12 bytes */
	for (int j = 0; j < 4; j++) {
		dst->scales[j] = (qsc[j] & 63) | ((qsc[j + 4] >> 4) << 6);
		dst->scales[j + 4] = (qmn[j] & 63) | ((qmn[j + 4] >> 4) << 6);
	}
	for (int j = 0; j < 4; j++) {
		dst->scales[j + 8] = (qsc[j + 4] & 0xF) | ((qmn[j + 4] & 0xF) << 4);
	}

	/* quantize values to 4-bit unsigned */
	float d_f = f16_to_f32(dst->d);
	float dmin_f = f16_to_f32(dst->dmin);
	for (int j = 0; j < QK_K / 64; j++) {
		uint8_t s1, m1, s2, m2;
		get_scale_min_k4(2 * j, dst->scales, &s1, &m1);
		get_scale_min_k4(2 * j + 1, dst->scales, &s2, &m2);

		float inv1 = (d_f * s1) > 0 ? 1.0f / (d_f * s1) : 0;
		float inv2 = (d_f * s2) > 0 ? 1.0f / (d_f * s2) : 0;

		for (int k = 0; k < 32; k++) {
			int lo = (int)roundf((src[j * 64 + k] + dmin_f * m1) * inv1);
			if (lo < 0) lo = 0; if (lo > 15) lo = 15;
			int hi = (int)roundf((src[j * 64 + 32 + k] + dmin_f * m2) * inv2);
			if (hi < 0) hi = 0; if (hi > 15) hi = 15;
			dst->qs[j * 32 + k] = (uint8_t)(lo | (hi << 4));
		}
	}
}

static void quantize_q5_K_block(const float *src, block_q5_K *dst)
{
	memset(dst, 0, sizeof(*dst));

	float sc[8], mn[8];
	for (int sb = 0; sb < 8; sb++) {
		float smin = src[sb * 32], smax = src[sb * 32];
		for (int k = 1; k < 32; k++) {
			float v = src[sb * 32 + k];
			if (v < smin) smin = v;
			if (v > smax) smax = v;
		}
		mn[sb] = smin < 0 ? -smin : 0;
		sc[sb] = (smax + mn[sb]) / 31.0f;
		if (sc[sb] < 1e-10f) sc[sb] = 1e-10f;
	}

	float max_sc = 0, max_mn = 0;
	for (int sb = 0; sb < 8; sb++) {
		if (sc[sb] > max_sc) max_sc = sc[sb];
		if (mn[sb] > max_mn) max_mn = mn[sb];
	}
	float d = max_sc / 63.0f;
	float dmin = max_mn / 63.0f;
	if (d < 1e-10f) d = 1e-10f;
	if (dmin < 1e-10f) dmin = 1e-10f;

	dst->d = f32_to_f16(d);
	dst->dmin = f32_to_f16(dmin);

	uint8_t qsc[8], qmn[8];
	for (int sb = 0; sb < 8; sb++) {
		qsc[sb] = (uint8_t)(sc[sb] / d + 0.5f);
		if (qsc[sb] > 63) qsc[sb] = 63;
		qmn[sb] = (uint8_t)(mn[sb] / dmin + 0.5f);
		if (qmn[sb] > 63) qmn[sb] = 63;
	}

	/* same scale packing as Q4_K */
	for (int j = 0; j < 4; j++) {
		dst->scales[j] = (qsc[j] & 63) | ((qsc[j + 4] >> 4) << 6);
		dst->scales[j + 4] = (qmn[j] & 63) | ((qmn[j + 4] >> 4) << 6);
	}
	for (int j = 0; j < 4; j++) {
		dst->scales[j + 8] = (qsc[j + 4] & 0xF) | ((qmn[j + 4] & 0xF) << 4);
	}

	/* quantize values to 5-bit unsigned */
	float d_f = f16_to_f32(dst->d);
	float dmin_f = f16_to_f32(dst->dmin);
	for (int j = 0; j < QK_K / 64; j++) {
		uint8_t s1, m1, s2, m2;
		get_scale_min_k4(2 * j, dst->scales, &s1, &m1);
		get_scale_min_k4(2 * j + 1, dst->scales, &s2, &m2);

		float inv1 = (d_f * s1) > 0 ? 1.0f / (d_f * s1) : 0;
		float inv2 = (d_f * s2) > 0 ? 1.0f / (d_f * s2) : 0;

		for (int k = 0; k < 32; k++) {
			int v1 = (int)roundf((src[j * 64 + k] + dmin_f * m1) * inv1);
			if (v1 < 0) v1 = 0; if (v1 > 31) v1 = 31;
			int v2 = (int)roundf((src[j * 64 + 32 + k] + dmin_f * m2) * inv2);
			if (v2 < 0) v2 = 0; if (v2 > 31) v2 = 31;

			dst->qs[j * 32 + k] = (uint8_t)((v1 & 0xF) | ((v2 & 0xF) << 4));
			/* high bit into qh: v1 bit4 → qh[k] bit (j*2), v2 bit4 → qh[k] bit (j*2+1) */
			if (v1 & 16) dst->qh[k] |= (1 << (j * 2));
			if (v2 & 16) dst->qh[k] |= (1 << (j * 2 + 1));
		}
	}
}

static void quantize_q6_K_block(const float *src, block_q6_K *dst)
{
	memset(dst, 0, sizeof(*dst));

	/* 16 groups of 16 elements, each with an 8-bit scale */
	float sc[16];
	for (int g = 0; g < 16; g++) {
		float amax = 0;
		for (int k = 0; k < 16; k++) {
			float av = fabsf(src[g * 16 + k]);
			if (av > amax) amax = av;
		}
		sc[g] = amax / 31.0f; /* range -32..31, symmetric around 0 */
		if (sc[g] < 1e-10f) sc[g] = 1e-10f;
	}

	/* super-block scale */
	float max_sc = 0;
	for (int g = 0; g < 16; g++)
		if (sc[g] > max_sc) max_sc = sc[g];
	float d = max_sc / 127.0f;
	if (d < 1e-10f) d = 1e-10f;
	dst->d = f32_to_f16(d);

	/* quantize per-group scales to int8 */
	for (int g = 0; g < 16; g++) {
		int qs = (int)roundf(sc[g] / d);
		if (qs > 127) qs = 127;
		if (qs < -128) qs = -128;
		dst->scales[g] = (int8_t)qs;
	}

	/* quantize values to 6-bit (0-63, centered at 32) */
	/* pack into ql/qh using ggml interleaved layout */
	float d_f = f16_to_f32(dst->d);

	for (int chunk = 0; chunk < 2; chunk++) {
		uint8_t *ql = dst->ql + chunk * 64;
		uint8_t *qh = dst->qh + chunk * 32;
		int n_128 = chunk * 128;

		for (int l = 0; l < 32; l++) {
			int is = n_128 / 16 + l / 16;

			/* compute q6 values for 4 elements */
			float sc1 = d_f * dst->scales[is + 0];
			float sc2 = d_f * dst->scales[is + 2];
			float sc3 = d_f * dst->scales[is + 4];
			float sc4 = d_f * dst->scales[is + 6];

			int q1 = sc1 > 0 ? (int)roundf(src[n_128 + l + 0] / sc1) + 32 : 32;
			int q2 = sc2 > 0 ? (int)roundf(src[n_128 + l + 32] / sc2) + 32 : 32;
			int q3 = sc3 > 0 ? (int)roundf(src[n_128 + l + 64] / sc3) + 32 : 32;
			int q4 = sc4 > 0 ? (int)roundf(src[n_128 + l + 96] / sc4) + 32 : 32;

			if (q1 < 0) q1 = 0; if (q1 > 63) q1 = 63;
			if (q2 < 0) q2 = 0; if (q2 > 63) q2 = 63;
			if (q3 < 0) q3 = 0; if (q3 > 63) q3 = 63;
			if (q4 < 0) q4 = 0; if (q4 > 63) q4 = 63;

			/* pack: ql[l] lo nibble = q1 lo4, ql[l] hi nibble = q3 lo4 */
			/*        ql[l+32] lo nibble = q2 lo4, ql[l+32] hi nibble = q4 lo4 */
			ql[l]      = (uint8_t)((q1 & 0xF) | ((q3 & 0xF) << 4));
			ql[l + 32] = (uint8_t)((q2 & 0xF) | ((q4 & 0xF) << 4));

			/* qh[l] bits 0-1 = q1 hi2, bits 2-3 = q2 hi2, bits 4-5 = q3 hi2, bits 6-7 = q4 hi2 */
			qh[l] = (uint8_t)(((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) |
					  (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6));
		}
	}
}

static tensor_t *make_q4_K_tensor(const tensor_t *src)
{
	size_t rows = src->dim[0];
	size_t cols = src->dim[1];
	size_t blocks_per_row = cols / QK_K;
	size_t total_blocks = rows * blocks_per_row;

	block_q4_K *blocks = calloc(total_blocks, sizeof(block_q4_K));
	assert(blocks);

	for (size_t r = 0; r < rows; r++)
		for (size_t b = 0; b < blocks_per_row; b++)
			quantize_q4_K_block(&src->data[r * cols + b * QK_K],
					    &blocks[r * blocks_per_row + b]);

	tensor_t *t = tensor_new_mapped(blocks, rows * cols, TENSOR_Q4_K);
	tensor_reshape_2d(t, rows, cols);
	return t;
}

static tensor_t *make_q5_K_tensor(const tensor_t *src)
{
	size_t rows = src->dim[0];
	size_t cols = src->dim[1];
	size_t blocks_per_row = cols / QK_K;
	size_t total_blocks = rows * blocks_per_row;

	block_q5_K *blocks = calloc(total_blocks, sizeof(block_q5_K));
	assert(blocks);

	for (size_t r = 0; r < rows; r++)
		for (size_t b = 0; b < blocks_per_row; b++)
			quantize_q5_K_block(&src->data[r * cols + b * QK_K],
					    &blocks[r * blocks_per_row + b]);

	tensor_t *t = tensor_new_mapped(blocks, rows * cols, TENSOR_Q5_K);
	tensor_reshape_2d(t, rows, cols);
	return t;
}

static tensor_t *make_q6_K_tensor(const tensor_t *src)
{
	size_t rows = src->dim[0];
	size_t cols = src->dim[1];
	size_t blocks_per_row = cols / QK_K;
	size_t total_blocks = rows * blocks_per_row;

	block_q6_K *blocks = calloc(total_blocks, sizeof(block_q6_K));
	assert(blocks);

	for (size_t r = 0; r < rows; r++)
		for (size_t b = 0; b < blocks_per_row; b++)
			quantize_q6_K_block(&src->data[r * cols + b * QK_K],
					    &blocks[r * blocks_per_row + b]);

	tensor_t *t = tensor_new_mapped(blocks, rows * cols, TENSOR_Q6_K);
	tensor_reshape_2d(t, rows, cols);
	return t;
}

static void ref_dequant_q4_K(const block_q4_K *src, float *dst, size_t n)
{
	size_t nb = n / QK_K;
	for (size_t i = 0; i < nb; i++) {
		float d = f16_to_f32(src[i].d);
		float dmin = f16_to_f32(src[i].dmin);
		const uint8_t *q = src[i].qs;

		for (int j = 0; j < QK_K / 64; j++) {
			uint8_t sc1, m1, sc2, m2;
			get_scale_min_k4(2 * j, src[i].scales, &sc1, &m1);
			get_scale_min_k4(2 * j + 1, src[i].scales, &sc2, &m2);

			for (int k = 0; k < 32; k++)
				dst[i * QK_K + j * 64 + k] = d * sc1 * (q[j * 32 + k] & 0xF) - dmin * m1;
			for (int k = 0; k < 32; k++)
				dst[i * QK_K + j * 64 + 32 + k] = d * sc2 * (q[j * 32 + k] >> 4) - dmin * m2;
		}
	}
}

static float ref_dot_q4_K(const float *x, const block_q4_K *y, size_t n)
{
	float ref[QK_K * 4];
	ref_dequant_q4_K(y, ref, n);
	float sum = 0;
	for (size_t i = 0; i < n; i++)
		sum += x[i] * ref[i];
	return sum;
}

static void ref_dequant_q5_K(const block_q5_K *src, float *dst, size_t n)
{
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

			for (int k = 0; k < 32; k++) {
				uint8_t lo4 = ql[j * 32 + k] & 0x0F;
				uint8_t hi1 = (qh[k] >> (j * 2)) & 1;
				int val = lo4 | (hi1 << 4);
				dst[i * QK_K + j * 64 + k] = d * sc1 * val - dmin * m1;
			}
			for (int k = 0; k < 32; k++) {
				uint8_t hi4 = ql[j * 32 + k] >> 4;
				uint8_t hi1 = (qh[k] >> (j * 2 + 1)) & 1;
				int val = hi4 | (hi1 << 4);
				dst[i * QK_K + j * 64 + 32 + k] = d * sc2 * val - dmin * m2;
			}
		}
	}
}

static float ref_dot_q5_K(const float *x, const block_q5_K *y, size_t n)
{
	float ref[QK_K * 4];
	ref_dequant_q5_K(y, ref, n);
	float sum = 0;
	for (size_t i = 0; i < n; i++)
		sum += x[i] * ref[i];
	return sum;
}

static void ref_dequant_q6_K(const block_q6_K *src, float *dst, size_t n)
{
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

static float ref_dot_q6_K(const float *x, const block_q6_K *y, size_t n)
{
	float ref[QK_K * 4];
	ref_dequant_q6_K(y, ref, n);
	float sum = 0;
	for (size_t i = 0; i < n; i++)
		sum += x[i] * ref[i];
	return sum;
}

static void test_dequant_q4_K(void)
{
	size_t sizes[] = { QK_K, 2 * QK_K };

	for (int s = 0; s < 2; s++) {
		size_t n = sizes[s];
		size_t nb = n / QK_K;

		float *src = calloc(n, sizeof(float));
		for (size_t i = 0; i < n; i++)
			src[i] = (float)i * 0.01f - (float)n * 0.005f;

		block_q4_K *blocks = calloc(nb, sizeof(block_q4_K));
		for (size_t i = 0; i < nb; i++)
			quantize_q4_K_block(&src[i * QK_K], &blocks[i]);

		float *expected = calloc(n, sizeof(float));
		ref_dequant_q4_K(blocks, expected, n);

		float *got = calloc(n, sizeof(float));
		dequant_row(blocks, TENSOR_Q4_K, 0, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_q4_K");

		free(src); free(blocks); free(expected); free(got);
	}
}

static void test_dot_q4_K(void)
{
	size_t sizes[] = { QK_K, 2 * QK_K };

	for (int s = 0; s < 2; s++) {
		size_t n = sizes[s];
		size_t nb = n / QK_K;

		float *x = calloc(n, sizeof(float));
		float *w = calloc(n, sizeof(float));
		for (size_t i = 0; i < n; i++) {
			x[i] = (float)i * 0.01f - 0.5f;
			w[i] = (float)(n - i) * 0.02f - 0.3f;
		}

		block_q4_K *blocks = calloc(nb, sizeof(block_q4_K));
		for (size_t i = 0; i < nb; i++)
			quantize_q4_K_block(&w[i * QK_K], &blocks[i]);

		float expected = ref_dot_q4_K(x, blocks, n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q4_K, 0, n);

		assert_close(got, expected, 1e-2f, "dot_q4_K");

		free(x); free(w); free(blocks);
	}
}

static void test_dequant_q5_K(void)
{
	size_t sizes[] = { QK_K, 2 * QK_K };

	for (int s = 0; s < 2; s++) {
		size_t n = sizes[s];
		size_t nb = n / QK_K;

		float *src = calloc(n, sizeof(float));
		for (size_t i = 0; i < n; i++)
			src[i] = (float)i * 0.01f - (float)n * 0.005f;

		block_q5_K *blocks = calloc(nb, sizeof(block_q5_K));
		for (size_t i = 0; i < nb; i++)
			quantize_q5_K_block(&src[i * QK_K], &blocks[i]);

		float *expected = calloc(n, sizeof(float));
		ref_dequant_q5_K(blocks, expected, n);

		float *got = calloc(n, sizeof(float));
		dequant_row(blocks, TENSOR_Q5_K, 0, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_q5_K");

		free(src); free(blocks); free(expected); free(got);
	}
}

static void test_dot_q5_K(void)
{
	size_t sizes[] = { QK_K, 2 * QK_K };

	for (int s = 0; s < 2; s++) {
		size_t n = sizes[s];
		size_t nb = n / QK_K;

		float *x = calloc(n, sizeof(float));
		float *w = calloc(n, sizeof(float));
		for (size_t i = 0; i < n; i++) {
			x[i] = (float)i * 0.01f - 0.5f;
			w[i] = (float)(n - i) * 0.02f - 0.3f;
		}

		block_q5_K *blocks = calloc(nb, sizeof(block_q5_K));
		for (size_t i = 0; i < nb; i++)
			quantize_q5_K_block(&w[i * QK_K], &blocks[i]);

		float expected = ref_dot_q5_K(x, blocks, n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q5_K, 0, n);

		assert_close(got, expected, 1e-2f, "dot_q5_K");

		free(x); free(w); free(blocks);
	}
}

static void test_dequant_q6_K(void)
{
	size_t sizes[] = { QK_K, 2 * QK_K };

	for (int s = 0; s < 2; s++) {
		size_t n = sizes[s];
		size_t nb = n / QK_K;

		float *src = calloc(n, sizeof(float));
		for (size_t i = 0; i < n; i++)
			src[i] = (float)i * 0.01f - (float)n * 0.005f;

		block_q6_K *blocks = calloc(nb, sizeof(block_q6_K));
		for (size_t i = 0; i < nb; i++)
			quantize_q6_K_block(&src[i * QK_K], &blocks[i]);

		float *expected = calloc(n, sizeof(float));
		ref_dequant_q6_K(blocks, expected, n);

		float *got = calloc(n, sizeof(float));
		dequant_row(blocks, TENSOR_Q6_K, 0, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_q6_K");

		free(src); free(blocks); free(expected); free(got);
	}
}

static void test_dot_q6_K(void)
{
	size_t sizes[] = { QK_K, 2 * QK_K };

	for (int s = 0; s < 2; s++) {
		size_t n = sizes[s];
		size_t nb = n / QK_K;

		float *x = calloc(n, sizeof(float));
		float *w = calloc(n, sizeof(float));
		for (size_t i = 0; i < n; i++) {
			x[i] = (float)i * 0.01f - 0.5f;
			w[i] = (float)(n - i) * 0.02f - 0.3f;
		}

		block_q6_K *blocks = calloc(nb, sizeof(block_q6_K));
		for (size_t i = 0; i < nb; i++)
			quantize_q6_K_block(&w[i * QK_K], &blocks[i]);

		float expected = ref_dot_q6_K(x, blocks, n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q6_K, 0, n);

		assert_close(got, expected, 1e-2f, "dot_q6_K");

		free(x); free(w); free(blocks);
	}
}

static void test_dequant_row_indexing_q4_K(void)
{
	size_t n = QK_K;
	size_t nb = 1;
	size_t nrows = 3;

	float *src = calloc(nrows * n, sizeof(float));
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r * n + i] = (float)(r * 100 + i) * 0.01f;

	block_q4_K *blocks = calloc(nrows * nb, sizeof(block_q4_K));
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q4_K_block(&src[r * n + b * QK_K], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float *expected = calloc(n, sizeof(float));
		ref_dequant_q4_K(&blocks[r * nb], expected, n);

		float *got = calloc(n, sizeof(float));
		dequant_row(blocks, TENSOR_Q4_K, r, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_row_indexing q4_K");

		free(expected); free(got);
	}
	free(src); free(blocks);
}

static void test_dot_row_indexing_q4_K(void)
{
	size_t n = QK_K;
	size_t nb = 1;
	size_t nrows = 3;

	float *x = calloc(n, sizeof(float));
	for (size_t i = 0; i < n; i++)
		x[i] = (float)i * 0.01f;

	float *src = calloc(nrows * n, sizeof(float));
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r * n + i] = (float)(r * 100 + i) * 0.02f - 1.0f;

	block_q4_K *blocks = calloc(nrows * nb, sizeof(block_q4_K));
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q4_K_block(&src[r * n + b * QK_K], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected = ref_dot_q4_K(x, &blocks[r * nb], n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q4_K, r, n);
		assert_close(got, expected, 1e-2f, "dot_row_indexing q4_K");
	}
	free(x); free(src); free(blocks);
}

static void test_dequant_row_indexing_q5_K(void)
{
	size_t n = QK_K;
	size_t nb = 1;
	size_t nrows = 3;

	float *src = calloc(nrows * n, sizeof(float));
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r * n + i] = (float)(r * 100 + i) * 0.01f;

	block_q5_K *blocks = calloc(nrows * nb, sizeof(block_q5_K));
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q5_K_block(&src[r * n + b * QK_K], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float *expected = calloc(n, sizeof(float));
		ref_dequant_q5_K(&blocks[r * nb], expected, n);

		float *got = calloc(n, sizeof(float));
		dequant_row(blocks, TENSOR_Q5_K, r, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_row_indexing q5_K");

		free(expected); free(got);
	}
	free(src); free(blocks);
}

static void test_dot_row_indexing_q5_K(void)
{
	size_t n = QK_K;
	size_t nb = 1;
	size_t nrows = 3;

	float *x = calloc(n, sizeof(float));
	for (size_t i = 0; i < n; i++)
		x[i] = (float)i * 0.01f;

	float *src = calloc(nrows * n, sizeof(float));
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r * n + i] = (float)(r * 100 + i) * 0.02f - 1.0f;

	block_q5_K *blocks = calloc(nrows * nb, sizeof(block_q5_K));
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q5_K_block(&src[r * n + b * QK_K], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected = ref_dot_q5_K(x, &blocks[r * nb], n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q5_K, r, n);
		assert_close(got, expected, 1e-2f, "dot_row_indexing q5_K");
	}
	free(x); free(src); free(blocks);
}

static void test_dequant_row_indexing_q6_K(void)
{
	size_t n = QK_K;
	size_t nb = 1;
	size_t nrows = 3;

	float *src = calloc(nrows * n, sizeof(float));
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r * n + i] = (float)(r * 100 + i) * 0.01f - 1.0f;

	block_q6_K *blocks = calloc(nrows * nb, sizeof(block_q6_K));
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q6_K_block(&src[r * n + b * QK_K], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float *expected = calloc(n, sizeof(float));
		ref_dequant_q6_K(&blocks[r * nb], expected, n);

		float *got = calloc(n, sizeof(float));
		dequant_row(blocks, TENSOR_Q6_K, r, got, n);

		for (size_t i = 0; i < n; i++)
			assert_close(got[i], expected[i], 1e-6f, "dequant_row_indexing q6_K");

		free(expected); free(got);
	}
	free(src); free(blocks);
}

static void test_dot_row_indexing_q6_K(void)
{
	size_t n = QK_K;
	size_t nb = 1;
	size_t nrows = 3;

	float *x = calloc(n, sizeof(float));
	for (size_t i = 0; i < n; i++)
		x[i] = (float)i * 0.01f;

	float *src = calloc(nrows * n, sizeof(float));
	for (size_t r = 0; r < nrows; r++)
		for (size_t i = 0; i < n; i++)
			src[r * n + i] = (float)(r * 100 + i) * 0.02f - 1.0f;

	block_q6_K *blocks = calloc(nrows * nb, sizeof(block_q6_K));
	for (size_t r = 0; r < nrows; r++)
		for (size_t b = 0; b < nb; b++)
			quantize_q6_K_block(&src[r * n + b * QK_K], &blocks[r * nb + b]);

	for (size_t r = 0; r < nrows; r++) {
		float expected = ref_dot_q6_K(x, &blocks[r * nb], n);
		float got = dot_f32_quant(x, blocks, TENSOR_Q6_K, r, n);
		assert_close(got, expected, 1e-2f, "dot_row_indexing q6_K");
	}
	free(x); free(src); free(blocks);
}

static void bench_mma(int rounds, size_t M, size_t K, size_t N, const char *title)
{
	tensor_t *lhs, *rhs_f32, *rhs_q8, *rhs_q4, *rhs_q4k, *rhs_q5k, *rhs_q6k, *add, *ret;

	lhs = tensor_new_zero(2, (size_t)M, (size_t)K);
	rhs_f32 = tensor_new_zero(2, (size_t)N, (size_t)K);
	add = tensor_new_zero(1, (size_t)N);
	ret = tensor_new_zero(2, (size_t)M, (size_t)N);

	for (size_t i = 0; i < lhs->totlen; i++)
		lhs->data[i] = (float)(i % 17) * 0.01f - 0.08f;
	for (size_t i = 0; i < rhs_f32->totlen; i++)
		rhs_f32->data[i] = (float)(i % 13) * 0.01f - 0.06f;
	for (size_t i = 0; i < add->totlen; i++)
		add->data[i] = 0.001f * (float)i;

	rhs_q8 = make_q8_0_tensor(rhs_f32);
	rhs_q4 = make_q4_0_tensor(rhs_f32);
	rhs_q4k = K % QK_K == 0 ? make_q4_K_tensor(rhs_f32) : NULL;
	rhs_q5k = K % QK_K == 0 ? make_q5_K_tensor(rhs_f32) : NULL;
	rhs_q6k = K % QK_K == 0 ? make_q6_K_tensor(rhs_f32) : NULL;

	printf("%zux%zu @ %zux%zu (transposed), %d rounds, %s:\n", M, K, N, K, rounds, title);

	bench_begin("f32");
	uint64_t start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_transposed_2x2(ret, lhs, rhs_f32, add);
	uint64_t dur_f32 = now() - start;
	bench_entry("f32", rounds, dur_f32, 0);
	bench_end();

	bench_begin("q8_0");
	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_transposed_2x2(ret, lhs, rhs_q8, add);
	uint64_t dur_q8 = now() - start;
	bench_entry("q8_0", rounds, dur_q8, dur_f32);
	bench_end();

	bench_begin("q4_0");
	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_transposed_2x2(ret, lhs, rhs_q4, add);
	uint64_t dur_q4 = now() - start;
	bench_entry("q4_0", rounds, dur_q4, dur_f32);
	bench_end();

	if (rhs_q4k) {
		bench_begin("q4_K");
		start = now();
		for (int i = 0; i < rounds; i++)
			tensor_mma_transposed_2x2(ret, lhs, rhs_q4k, add);
		uint64_t dur = now() - start;
		bench_entry("q4_K", rounds, dur, dur_f32);
		bench_end();
	}

	if (rhs_q5k) {
		bench_begin("q5_K");
		start = now();
		for (int i = 0; i < rounds; i++)
			tensor_mma_transposed_2x2(ret, lhs, rhs_q5k, add);
		uint64_t dur = now() - start;
		bench_entry("q5_K", rounds, dur, dur_f32);
		bench_end();
	}

	if (rhs_q6k) {
		bench_begin("q6_K");
		start = now();
		for (int i = 0; i < rounds; i++)
			tensor_mma_transposed_2x2(ret, lhs, rhs_q6k, add);
		uint64_t dur = now() - start;
		bench_entry("q6_K", rounds, dur, dur_f32);
		bench_end();
	}
}

int main(void)
{
	/* correctness tests */
	test_dequant_q8_0();
	test_dequant_q4_0();
	test_dot_q8_0();
	test_dot_q4_0();
	test_dequant_row_indexing_q8_0();
	test_dequant_row_indexing_q4_0();
	test_dot_row_indexing_q8_0();
	test_dot_row_indexing_q4_0();

	test_dequant_q4_K();
	test_dot_q4_K();
	test_dequant_q5_K();
	test_dot_q5_K();
	test_dequant_q6_K();
	test_dot_q6_K();
	test_dequant_row_indexing_q4_K();
	test_dot_row_indexing_q4_K();
	test_dequant_row_indexing_q5_K();
	test_dot_row_indexing_q5_K();
	test_dequant_row_indexing_q6_K();
	test_dot_row_indexing_q6_K();
	printf("quant: ok\n");

	/* benchmarks */
	bench_mma(1000, 1, 768, 3072, "GPT-2 124M sized matmul (decode, single token)");
	bench_mma(1000, 1, 768, 2304, "GPT-2 124M (QKV projection)");
	bench_mma(200, 8, 768, 3072, "small prefill batch");

	return 0;
}
