#include "test.h"
#include "../tensor.h"
#include "../nn.h"

static void top_k_test(void)
{
	tensor_t *x = tensor_new_1d(10,
			    0.0,  /* 0 */
			    1.0,  /* 1 */
			    7.0,  /* 2 */
			    2.0,  /* 3 */
			    9.0,  /* 4 */
			    3.0,  /* 5 */
			    6.0,  /* 6 */
			    4.0,  /* 7 */
			    5.0,  /* 8 */
			    8.0   /* 9 */);

	size_t top_n[5];
	scalar_t top_v[5];

	top_k(x, &top_n[0], &top_v[0], 5);
	assert(top_n[0] == 8);
	assert(top_n[1] == 6);
	assert(top_n[2] == 2);
	assert(top_n[3] == 9);
	assert(top_n[4] == 4);
	assert(top_v[0] == 5.0);
	assert(top_v[1] == 6.0);
	assert(top_v[2] == 7.0);
	assert(top_v[3] == 8.0);
	assert(top_v[4] == 9.0);
}

int main(void)
{
	top_k_test();
	printf("nn: ok\n");
}
