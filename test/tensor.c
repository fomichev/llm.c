#include "llm.h"
#include "test/test.h"

static void ft_eq_str(ft_t *t, const char *expected)
{
	char *s = ft_to_string(t);
	if (strcmp(s, expected)) {
		printf("'%s' != '%s'\n", s, expected);
		assert(false);
	}
}

static void test_ft_to_string(void)
{
	ft_t *t;

	t = ft_new_1d(1,
		      1.0);
	ft_eq_str(t, "[+1.00e+00]");

	t = ft_new_1d(2,
		      1.0, 2.0);
	ft_eq_str(t, "[+1.00e+00 +2.00e+00]");

	t = ft_new_2d(2, 1,
		      1.0, 2.0);
	ft_eq_str(t, "[[+1.00e+00][+2.00e+00]]");

	t = ft_new_2d(1, 2,
		      1.0, 2.0);
	ft_eq_str(t, "[[+1.00e+00 +2.00e+00]]");

	t = ft_new_3d(2, 1, 1,
		      1.0, 2.0);
	ft_eq_str(t, "[[[+1.00e+00]][[+2.00e+00]]]");

	t = ft_new_3d(1, 1, 2,
		      1.0, 2.0);
	ft_eq_str(t, "[[[+1.00e+00 +2.00e+00]]]");

	t = ft_new_3d(2, 1, 2,
		      1.0, 2.0, 3.0, 4.0);
	ft_eq_str(t, "[[[+1.00e+00 +2.00e+00]][[+3.00e+00 +4.00e+00]]]");

	t = ft_new_3d(1, 2, 2,
		      1.0, 2.0, 3.0, 4.0);
	ft_eq_str(t, "[[[+1.00e+00 +2.00e+00][+3.00e+00 +4.00e+00]]]");
}

static void test_ft_add(void)
{
	ft_t *ret, *lhs, *rhs;

	lhs = ft_new_2d(3, 2,
			1.0, 1.0, 2.0, 2.0, 3.0, 3.0);
	rhs = ft_new_2d(3, 2,
			2.0, 2.0, 3.0, 3.0, 4.0, 4.0);
	ret = ft_new_zero(2, 3, 2);

	ft_add(ret, lhs, rhs);

	ft_eq_str(ret, "["
		  "[+3.00e+00 +3.00e+00]"
		  "[+5.00e+00 +5.00e+00]"
		  "[+7.00e+00 +7.00e+00]"
		  "]");
}

static void test_ft_sub(void)
{
	ft_t *ret, *lhs, *rhs;

	lhs = ft_new_2d(3, 2,
			1.0, 1.0, 2.0, 2.0, 3.0, 3.0);
	rhs = ft_new_2d(3, 2,
			3.0, 3.0, 2.0, 2.0, 1.0, 1.0);
	ret = ft_new_zero(2, 3, 2);

	ft_sub(ret, lhs, rhs);

	ft_eq_str(ret, "["
		  "[-2.00e+00 -2.00e+00]"
		  "[+0.00e+00 +0.00e+00]"
		  "[+2.00e+00 +2.00e+00]"
		  "]");
}

static void test_ft_mul(void)
{
	ft_t *ret, *lhs, *rhs;

	lhs = ft_new_2d(3, 2,
			1.0, 1.0, 2.0, 2.0, 3.0, 3.0);
	rhs = ft_new_2d(3, 2,
			2.0, 2.0, 3.0, 3.0, 4.0, 4.0);
	ret = ft_new_zero(2, 3, 2);

	ft_mul(ret, lhs, rhs);

	ft_eq_str(ret, "["
		  "[+2.00e+00 +2.00e+00]"
		  "[+6.00e+00 +6.00e+00]"
		  "[+1.20e+01 +1.20e+01]"
		  "]");
}

static void test_ft_div(void)
{
	ft_t *ret, *lhs, *rhs;

	lhs = ft_new_2d(3, 2,
			2.0, 2.0, 4.0, 4.0, 9.0, 9.0);
	rhs = ft_new_2d( 3, 2,
			 2.0, 2.0, 2.0, 2.0, 3.0, 3.0);
	ret = ft_new_zero(2, 3, 2);

	ft_div(ret, lhs, rhs);

	ft_eq_str(ret, "["
		  "[+1.00e+00 +1.00e+00]"
		  "[+2.00e+00 +2.00e+00]"
		  "[+3.00e+00 +3.00e+00]"
		  "]");
}

static void test_ft_mma(void)
{
	ft_t *ret, *lhs, *rhs, *add;

	lhs = ft_new_2d(2, 3,
			0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
	rhs = ft_new_2d(3, 2,
			6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
	add = ft_new_2d(2, 2,
			1.0, 1.0, 1.0, 1.0);
	ret = ft_new_zero(2, 2, 2);
	ft_mma_2x2(ret, lhs, rhs, add);
	ft_eq_str(ret, "["
		  "[+2.90e+01 +3.20e+01]"
		  "[+1.01e+02 +1.13e+02]"
		  "]");

	lhs = ft_new_2d(2, 3,
			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0);
	rhs = ft_new_2d(3, 4,
			1.0, 2.0, 3.0, 5.0,
			6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0);
	ret = ft_new_zero(2, 2, 4);
	ft_mma_2x2(ret, lhs, rhs, NULL);
	ft_eq_str(ret, "[[+4.30e+01 +4.90e+01 +5.50e+01 +6.20e+01][+9.40e+01 +1.09e+02 +1.24e+02 +1.43e+02]]");

	rhs = ft_new_2d(4, 3,
			1.0, 6.0, 10.0,
			2.0, 7.0, 11.0,
			3.0, 8.0, 12.0,
			5.0, 9.0, 13.0);
	ft_mma_transposed_2x2(ret, lhs, rhs, NULL);
	ft_eq_str(ret, "[[+4.30e+01 +4.90e+01 +5.50e+01 +6.20e+01][+9.40e+01 +1.09e+02 +1.24e+02 +1.43e+02]]");

	lhs = ft_new_2d(2, 17,
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
			11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0);
	rhs = ft_new_2d(3, 17,
			1.0, 2.0, 3.0,
			21.0, 22.0, 23.0,
			31.0, 32.0, 33.0,
			41.0, 42.0, 43.0,
			51.0, 52.0, 53.0,
			61.0, 62.0, 63.0,
			71.0, 32.0, 73.0,
			81.0, 82.0, 83.0,
			91.0, 92.0, 93.0,
			101.0, 102.0, 103.0,
			111.0, 112.0, 113.0,
			121.0, 122.0, 123.0,
			131.0, 132.0, 133.0,
			141.0, 142.0, 143.0,
			151.0, 152.0, 153.0,
			161.0, 162.0, 163.0,
			171.0, 172.0, 173.0);
	ret = ft_new_zero(2, 2, 3);
	ft_mma_transposed_2x2(ret, lhs, rhs, NULL);
	ft_eq_str(ret, "[[+6.72e+03 +1.53e+04 +2.41e+04][+4.89e+04 +1.08e+05 +1.67e+05]]");
}

static void test_ft_max(void)
{
	size_t pos = 0;
	size_t n;
	ft_t *t;

	n = FV_CHUNK * 2;
	t = ft_new_zero(1, n);
	t->data[n - 1] = 1;

	assert(ft_max(t, &pos) == t->data[n - 1]);
	assert(pos == n - 1);

	n = FV_CHUNK * 2 - 1;
	t = ft_new_zero(1, n);
	t->data[n - 1] = 1;

	assert(ft_max(t, &pos) == t->data[n - 1]);
	assert(pos == n - 1);
}

static void bench_ft_mma_transposed(int rounds)
{
	ft_t *ret, *lhs, *rhs, *add;

	lhs = ft_new_zero(2, 4*8, 4*8);
	rhs = ft_new_zero(2, 4*8, 4*8);
	add = ft_new_zero(2, 4*8, 4*8);
	ret = ft_new_zero(2, 4*8, 4*8);

	bench_begin("mma_transposed");

	uint64_t start = now();
	for (int i = 0; i < rounds; i++)
		ft_mma_transposed_2x2(ret, lhs, rhs, add);
	uint64_t duration_base = now() - start;
	bench_entry("base", rounds, duration_base, 0);

	bench_end();
}

int main(int argc, char *argv[])
{
#ifdef __SSE__
	printf("__SSE__\n");
#endif
#ifdef __SSE2__
	printf("__SSE2__\n");
#endif
#ifdef __SSE3__
	printf("__SSE3__\n");
#endif
#ifdef __SSE4_1__
	printf("__SSE4_1__\n");
#endif
#ifdef __SSE4_2__
	printf("__SSE4_2__\n");
#endif
#ifdef __AVX__
	printf("__AVX__\n");
#endif
#ifdef __AVX2__
	printf("__AVX2__\n");
#endif
#ifdef __AVX512F__
	printf("__AVX512F__\n");
#endif

	test_ft_to_string();
	test_ft_add();
	test_ft_sub();
	test_ft_mul();
	test_ft_div();
	test_ft_mma();
	test_ft_max();
	printf("ft: ok\n");
	bench_ft_mma_transposed(1000);
}
