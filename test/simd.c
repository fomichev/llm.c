#include <math.h>

#include "llm.h"
#include "test/test.h"

static scalar_t *bench_alloc(size_t num, double value)
{
	scalar_t *vec;

	vec = aligned_alloc(VECTOR_ALIGN, sizeof(scalar_t) * num);

	for (int i = 0; i < num; i++)
		vec[i] = value;

	return vec;
}

static void bench_verify_single(scalar_t got, scalar_t expected)
{
	double eps = 0.001;

	if (got >= expected + eps || got <= expected - eps) {
		printf("unexpected value %f vs %f\n", got, expected);
		exit(1);
	}
}

static void bench_verify(scalar_t *vec, size_t num, double expected)
{
	for (int i = 0; i < num; i++)
		bench_verify_single(vec[i], expected);
}

#define BENCHMARK_OP2(PREFIX, BATCH, OP, ROUNDS) \
	({ \
		memset(ret, 0, num * sizeof(scalar_t)); \
		uint64_t start = now(); \
		for (int i = 0; i < ROUNDS; i++) { \
			PREFIX ## _vector_t vret, vlhs, vrhs; \
			for (int j = 0; j < num; j += BATCH) { \
				PREFIX ## _vector_load(&vlhs, &lhs[j]); \
				PREFIX ## _vector_load(&vrhs, &rhs[j]); \
				PREFIX ## _vector_ ## OP(&vret, &vlhs, &vrhs); \
				PREFIX ## _vector_store(&ret[j], &vret); \
			} \
		} \
		now() - start; \
	})

static void bench_add(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 2);
	scalar_t *rhs = bench_alloc(num, 3);
	scalar_t *ret = bench_alloc(num, 0);

	bench_begin("add");

	uint64_t duration_base = BENCHMARK_OP2(cpu, CPU_BATCH, add, rounds);
	bench_verify(ret, num, 5);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(avx2, AVX2_BATCH, add, rounds);
	bench_verify(ret, num, 5);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(avx512, AVX512_BATCH, add, rounds);
	bench_verify(ret, num, 5);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

static void bench_sub(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 3);
	scalar_t *rhs = bench_alloc(num, 2);
	scalar_t *ret = bench_alloc(num, 0);

	bench_begin("sub");

	uint64_t duration_base = BENCHMARK_OP2(cpu, CPU_BATCH, sub, rounds);
	bench_verify(ret, num, 1);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(avx2, AVX2_BATCH, sub, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(avx512, AVX512_BATCH, sub, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

static void bench_mul(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 2);
	scalar_t *rhs = bench_alloc(num, 3);
	scalar_t *ret = bench_alloc(num, 0);

	bench_begin("mul");

	uint64_t duration_base = BENCHMARK_OP2(cpu, CPU_BATCH, mul, rounds);
	bench_verify(ret, num, 6);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(avx2, AVX2_BATCH, mul, rounds);
	bench_verify(ret, num, 6);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(avx512, AVX512_BATCH, mul, rounds);
	bench_verify(ret, num, 6);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

static void bench_div(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 9);
	scalar_t *rhs = bench_alloc(num, 3);
	scalar_t *ret = bench_alloc(num, 0);

	bench_begin("div");

	uint64_t duration_base = BENCHMARK_OP2(cpu, CPU_BATCH, div, rounds);
	bench_verify(ret, num, 3);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP2(avx2, AVX2_BATCH, div, rounds);
	bench_verify(ret, num, 3);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP2(avx512, AVX512_BATCH, div, rounds);
	bench_verify(ret, num, 3);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

#define BENCHMARK_OP1(PREFIX, BATCH, OP, ROUNDS) \
	({ \
		memset(ret, 0, num * sizeof(scalar_t)); \
		uint64_t start = now(); \
		for (int i = 0; i < ROUNDS; i++) { \
			PREFIX ## _vector_t vret, vlhs; \
			for (int j = 0; j < num; j += BATCH) { \
				PREFIX ## _vector_load(&vlhs, &lhs[j]); \
				PREFIX ## _vector_ ## OP(&vret, &vlhs); \
				PREFIX ## _vector_store(&ret[j], &vret); \
			} \
		} \
		now() - start; \
	})

static void bench_exp(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 9);
	scalar_t *ret = bench_alloc(num, 0);

	bench_begin("exp");

	uint64_t duration_base = BENCHMARK_OP1(cpu, CPU_BATCH, exp, rounds);
	bench_verify(ret, num, 8103.083984);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP1(avx2, AVX2_BATCH, exp, rounds);
	bench_verify(ret, num, 8103.083984);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP1(avx512, AVX512_BATCH, exp, rounds);
	bench_verify(ret, num, 8103.083984);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

static void bench_tanh(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 9);
	scalar_t *ret = bench_alloc(num, 0);

	bench_begin("tanh");

	uint64_t duration_base = BENCHMARK_OP1(cpu, CPU_BATCH, tanh, rounds);
	bench_verify(ret, num, 1);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_OP1(avx2, AVX2_BATCH, tanh, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_OP1(avx512, AVX512_BATCH, tanh, rounds);
	bench_verify(ret, num, 1);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

#define BENCHMARK_REDUCE(PREFIX, BATCH, OP, ROUNDS, ACC) \
	({ \
		uint64_t start = now(); \
		for (int i = 0; i < ROUNDS; i++) { \
			PREFIX ## _vector_t vlhs; \
			(ACC) = 0; \
			for (int j = 0; j < num; j += BATCH) { \
				PREFIX ## _vector_load(&vlhs, &lhs[j]); \
				(ACC) += PREFIX ## _vector_reduce_ ## OP(&vlhs); \
			} \
		} \
		now() - start; \
	})

static void bench_sum(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 1);
	scalar_t acc;

	bench_begin("sum");

	uint64_t duration_base = BENCHMARK_REDUCE(cpu, CPU_BATCH, sum, rounds, acc);
	bench_verify_single(acc, num);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_REDUCE(avx2, AVX2_BATCH, sum, rounds, acc);
	bench_verify_single(acc, num);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_REDUCE(avx512, AVX512_BATCH, sum, rounds, acc);
	bench_verify_single(acc, num);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
#endif

	bench_end();
}

static void bench_max(int rounds, int num)
{
	scalar_t *lhs = bench_alloc(num, 1);
	scalar_t acc;

	bench_begin("max");

	uint64_t duration_base = BENCHMARK_REDUCE(cpu, CPU_BATCH, max, rounds, acc);
	bench_verify_single(acc, num / CPU_BATCH);
	bench_entry("base", rounds, duration_base, 0);

	uint64_t duration_avx2 = BENCHMARK_REDUCE(avx2, AVX2_BATCH, max, rounds, acc);
	bench_verify_single(acc, num / AVX2_BATCH);
	bench_entry("avx2", rounds, duration_avx2, duration_base);

#ifdef __AVX512F__
	uint64_t duration_avx512 = BENCHMARK_REDUCE(avx512, AVX512_BATCH, max, rounds, acc);
	bench_verify_single(acc, num / AVX512_BATCH);
	bench_entry("avx512", rounds, duration_avx512, duration_base);
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
	bench_sum(rounds, num);
	bench_max(rounds, num);
}
