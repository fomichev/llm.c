#include <math.h>

#include "llm.h"
#include "test/test.h"

static FT_TYPE *bench_alloc(size_t num, double value)
{
	FT_TYPE *vec;

	vec = aligned_alloc(FT_ALIGN, sizeof(FT_TYPE) * num);

	for (int i = 0; i < num; i++)
		vec[i] = value;

	return vec;
}

static void bench_verify(FT_TYPE *vec, size_t num, double value)
{
	for (int i = 0; i < num; i++) {
		double eps = 0.001;

		if (vec[i] >= value + eps || vec[i] <= value - eps) {
			printf("unexpected value %f vs %f\n", vec[i], value);
			exit(1);
		}
	}
}

#define BENCHMARK_OP2(PREFIX, TYPE, OP, ROUNDS) \
	({ \
		uint64_t start = now(); \
		memset(ret, 0, num * FT_SIZEOF); \
		for (int i = 0; i < ROUNDS; i++) { \
			TYPE vret, vlhs, vrhs; \
			for (int j = 0; j < num; j += PREFIX ## _N) { \
				PREFIX ## _FV_LOAD(vlhs, &lhs[j]); \
				PREFIX ## _FV_LOAD(vrhs, &rhs[j]); \
				PREFIX ## _ ## OP(vret, vlhs, vrhs); \
				PREFIX ## _FV_STORE(&ret[j], vret); \
			} \
		} \
		now() - start; \
	})

static void bench_add(int rounds, int num)
{
	FT_TYPE *lhs = bench_alloc(num, 2);
	FT_TYPE *rhs = bench_alloc(num, 3);
	FT_TYPE *ret = bench_alloc(num, 0);

	bench_begin("add");

	uint64_t duration_base = BENCHMARK_OP2(CPU, cpu_fv_t, FV_ADD, rounds);
	bench_verify(ret, num, 5);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(AVX2, avx2_fv_t, FV_ADD, rounds);
	bench_verify(ret, num, 5);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(AVX512, avx512_fv_t, FV_ADD, rounds);
	bench_verify(ret, num, 5);
	bench_entry("avx512", rounds, duration_avx512, duration_avx2);
#endif

	bench_end();
}

static void bench_sub(int rounds, int num)
{
	FT_TYPE *lhs = bench_alloc(num, 3);
	FT_TYPE *rhs = bench_alloc(num, 2);
	FT_TYPE *ret = bench_alloc(num, 0);

	bench_begin("sub");

	uint64_t duration_base = BENCHMARK_OP2(CPU, cpu_fv_t, FV_SUB, rounds);
	bench_verify(ret, num, 1);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(AVX2, avx2_fv_t, FV_SUB, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(AVX512, avx512_fv_t, FV_SUB, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx512", rounds, duration_avx512, duration_avx2);
#endif

	bench_end();
}

static void bench_mul(int rounds, int num)
{
	FT_TYPE *lhs = bench_alloc(num, 2);
	FT_TYPE *rhs = bench_alloc(num, 3);
	FT_TYPE *ret = bench_alloc(num, 0);

	bench_begin("mul");

	uint64_t duration_base = BENCHMARK_OP2(CPU, cpu_fv_t, FV_MUL, rounds);
	bench_verify(ret, num, 6);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(AVX2, avx2_fv_t, FV_MUL, rounds);
	bench_verify(ret, num, 6);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(AVX512, avx512_fv_t, FV_MUL, rounds);
	bench_verify(ret, num, 6);
	bench_entry("avx512", rounds, duration_avx512, duration_avx2);
#endif

	bench_end();
}

static void bench_div(int rounds, int num)
{
	FT_TYPE *lhs = bench_alloc(num, 9);
	FT_TYPE *rhs = bench_alloc(num, 3);
	FT_TYPE *ret = bench_alloc(num, 0);

	bench_begin("div");

	uint64_t duration_base = BENCHMARK_OP2(CPU, cpu_fv_t, FV_DIV, rounds);
	bench_verify(ret, num, 3);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(AVX2, avx2_fv_t, FV_DIV, rounds);
	bench_verify(ret, num, 3);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(AVX512, avx512_fv_t, FV_DIV, rounds);
	bench_verify(ret, num, 3);
	bench_entry("avx512", rounds, duration_avx512, duration_avx2);
#endif

	bench_end();
}

#define BENCHMARK_OP1(PREFIX, TYPE, OP, ROUNDS) \
	({ \
		uint64_t start = now(); \
		memset(ret, 0, num * FT_SIZEOF); \
		for (int i = 0; i < ROUNDS; i++) { \
			TYPE vret, vlhs; \
			for (int j = 0; j < num; j += PREFIX ## _N) { \
				PREFIX ## _FV_LOAD(vlhs, &lhs[j]); \
				PREFIX ## _ ## OP(vret, vlhs); \
				PREFIX ## _FV_STORE(&ret[j], vret); \
			} \
		} \
		now() - start; \
	})

static void bench_exp(int rounds, int num)
{
	FT_TYPE *lhs = bench_alloc(num, 9);
	FT_TYPE *rhs = bench_alloc(num, 3);
	FT_TYPE *ret = bench_alloc(num, 0);

	bench_begin("exp");

	uint64_t duration_base = BENCHMARK_OP1(CPU, cpu_fv_t, FV_EXP, rounds);
	bench_verify(ret, num, 8103.083984);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP1(AVX2, avx2_fv_t, FV_EXP, rounds);
	bench_verify(ret, num, 8103.083984);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP1(AVX512, avx512_fv_t, FV_EXP, rounds);
	bench_verify(ret, num, 8103.083984);
	bench_entry("avx512", rounds, duration_avx512, duration_avx2);
#endif

	bench_end();
}

static void bench_tanh(int rounds, int num)
{
	FT_TYPE *lhs = bench_alloc(num, 9);
	FT_TYPE *rhs = bench_alloc(num, 3);
	FT_TYPE *ret = bench_alloc(num, 0);

	bench_begin("tanh");

	uint64_t duration_base = BENCHMARK_OP1(CPU, cpu_fv_t, FV_TANH, rounds);
	bench_verify(ret, num, 1);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP1(AVX2, avx2_fv_t, FV_TANH, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP1(AVX512, avx512_fv_t, FV_TANH, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx512", rounds, duration_avx512, duration_avx2);
#endif

	bench_end();
}

int main(int argc, char *argv[])
{
	int num = 16 * 10000;
	int rounds = 1000;

	bench_add(rounds, num);
	bench_sub(rounds, num);
	bench_mul(rounds, num);
	bench_div(rounds, num);
	bench_exp(rounds, num);
	bench_tanh(rounds, num);
}
