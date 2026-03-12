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

static void bench_mma(int rounds, size_t M, size_t K, size_t N, const char *title)
{
	tensor_t *lhs, *rhs_f32, *rhs_q8, *rhs_q4, *add, *ret;

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
	printf("quant: ok\n");

	/* benchmarks */
	bench_mma(1000, 1, 768, 3072, "GPT-2 124M sized matmul (decode, single token)");
	bench_mma(1000, 1, 768, 2304, "GPT-2 124M (QKV projection)");
	bench_mma(200, 8, 768, 3072, "small prefill batch");

	return 0;
}
