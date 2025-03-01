#pragma once

#include <assert.h>
#include <stdio.h>

typedef struct {
	scalar_t *data;
	uint64_t totlen;
	uint64_t maxcap;
	uint64_t dim[4];
	uint32_t ndim;
} tensor_t;

tensor_t *tensor_new_zero(size_t ndim, ...);
tensor_t *tensor_new(size_t ndim, ...);
tensor_t *tensor_new_1d(size_t d1, ...);
tensor_t *tensor_new_2d(size_t d1, size_t d2, ...);
tensor_t *tensor_new_3d(size_t d1, size_t d2, size_t d3, ...);
void tensor_free(tensor_t *t);

tensor_t *tensor_new_mapped(void *data, size_t totlen);
void tensor_free_mapped(const tensor_t *t);

char *tensor_to_string(const tensor_t *t);
char *tensor_to_short_string(const tensor_t *t);
void tensor_pick_rows(tensor_t *dst, const tensor_t *src, const int *rows, size_t num);

static inline size_t tensor_len(const tensor_t *t)
{
	if (t->ndim == 0)
		return 0;
	return t->dim[0];
}

/* convert tensor to 1 dimension with specific size */
static inline void tensor_reshape_1d(tensor_t *t, size_t d1)
{
	assert(d1 <= t->maxcap);

	t->ndim = 1;
	t->dim[0] = d1;
	t->totlen = d1;
}

/* convert tensor to 2 dimensions with specific sizes */
static inline void tensor_reshape_2d(tensor_t *t, size_t d1, size_t d2)
{
	assert(d1 * d2 <= t->maxcap);

	t->ndim = 2;
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->totlen = d1 * d2;
}

/* convert tensor to 3 dimensions with specific sizes */
static inline void tensor_reshape_3d(tensor_t *t, size_t d1, size_t d2, size_t d3)
{
	assert(d1 * d2 * d3 <= t->maxcap);

	t->ndim = 3;
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->dim[2] = d3;
	t->totlen = d1 * d2 * d3;
}

/* convert tensor to 4 dimensions with specific sizes */
static inline void tensor_reshape_4d(tensor_t *t, size_t d1, size_t d2, size_t d3, size_t d4)
{
	assert(d1 * d2 * d3 * d4 <= t->maxcap);

	t->ndim = 2;
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->dim[2] = d3;
	t->dim[3] = d4;
	t->totlen = d1 * d2 * d3 * d4;
}

/* resize other dimension */
static inline void tensor_resize(tensor_t *t, size_t d)
{
	if (t->ndim == 1) {
		assert(d <= t->maxcap);

		t->dim[0] = d;
		t->totlen = t->dim[0];
	} else if (t->ndim == 2) {
		assert(t->dim[1] * d <= t->maxcap);

		t->dim[0] = d;
		t->totlen = t->dim[0] * t->dim[1];
	}
}

static inline void tensor_resize_2d(tensor_t *t, size_t d1, size_t d2)
{
	assert(t->ndim == 2);
	assert(d1 * d2 <= t->maxcap);

	t->dim[0] = d1;
	t->dim[1] = d2;
	t->totlen = d1 * d2;
}

/* returns a ft view over original ft */
#define tensor_at(t, idx, view) \
do { \
	assert((t)->ndim > 1); \
	assert((idx) < (t)->dim[0]); \
	/* f & view can alias */ \
	(view)->totlen = (t)->totlen / (t)->dim[0]; \
	(view)->maxcap = (t)->totlen; \
	(view)->data = &(t)->data[(view)->totlen * (idx)]; \
	(view)->ndim = (t)->ndim - 1; \
	(view)->dim[0] = (t)->dim[1]; \
	(view)->dim[1] = (t)->dim[2]; \
	(view)->dim[3] = (t)->dim[3]; \
} while (0)

static inline void tensor_assert_1d(const tensor_t *t, size_t d1)
{
	assert(t->ndim == 1);
	if (d1 > 0)
		assert(t->dim[0] == d1);
}

static inline void tensor_assert_2d(const tensor_t *t, size_t d1, size_t d2)
{
	assert(t->ndim == 2);
	if (d1 > 0)
		assert(t->dim[0] == d1);
	if (d2 > 0)
		assert(t->dim[1] == d2);
}

static inline void tensor_assert_1d_match(const tensor_t *lhs, const tensor_t *rhs)
{
	assert(lhs->dim[0] == rhs->dim[0]);
}

static inline void tensor_assert_2d_match(const tensor_t *lhs, const tensor_t *rhs)
{
	assert(lhs->dim[1] == rhs->dim[1]);
}

void tensor_set(tensor_t *ret, scalar_t val);
#define tensor_set_inner(dst, dst_idx, src) \
do { \
	assert((dst)->ndim == 2); \
	assert((dst)->dim[1] == (src)->totlen); \
	assert((dst_idx) < (dst)->dim[0]); \
	__tensor_set_inner((dst), (dst_idx), (src)); \
} while (0)

void __tensor_set_inner(tensor_t *dst, size_t dst_idx, const tensor_t *src);
void tensor_copy(tensor_t *dst, const tensor_t *src);
void tensor_add(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs);
void tensor_add_2x1(
	tensor_t *dst,
	const tensor_t *src,
	const tensor_t *vec);
void tensor_sub(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs);
void tensor_mul(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs);
void tensor_div(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs);
void tensor_div_scalar(
	tensor_t *ret,
	const tensor_t *lhs,
	scalar_t scalar);
scalar_t tensor_mean(const tensor_t *lhs);
scalar_t tensor_max(const tensor_t *lhs, size_t *pos);

/* ret = lhs @ rhs + add */
void tensor_mma_2x2(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add);
/* ret = lhs @ rhs.T + add */
void tensor_mma_transposed_2x2(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add);
