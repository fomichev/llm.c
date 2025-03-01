#pragma once

#include <assert.h>
#include <stdio.h>

typedef struct {
	scalar_t *data;
	uint64_t totlen;
	uint64_t maxcap;
	uint64_t dim[4];
	uint32_t ndim;
} ft_t;

ft_t *ft_new_zero(size_t ndim, ...);
ft_t *ft_new(size_t ndim, ...);
ft_t *ft_new_1d(size_t d1, ...);
ft_t *ft_new_2d(size_t d1, size_t d2, ...);
ft_t *ft_new_3d(size_t d1, size_t d2, size_t d3, ...);
void ft_free(ft_t *t);

ft_t *ft_new_mapped(void *data, size_t totlen);
void ft_free_mapped(const ft_t *t);

char *ft_to_string(const ft_t *t);
char *ft_to_short_string(const ft_t *t);
void ft_pick_rows(ft_t *dst, const ft_t *src, const int *rows, size_t num);

static inline size_t ft_len(const ft_t *t)
{
	if (t->ndim == 0)
		return 0;
	return t->dim[0];
}

/* convert tensor to 1 dimension with specific size */
static inline void ft_reshape_1d(ft_t *t, size_t d1)
{
	assert(d1 <= t->maxcap);

	t->ndim = 1;
	t->dim[0] = d1;
	t->totlen = d1;
}

/* convert tensor to 2 dimensions with specific sizes */
static inline void ft_reshape_2d(ft_t *t, size_t d1, size_t d2)
{
	assert(d1 * d2 <= t->maxcap);

	t->ndim = 2;
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->totlen = d1 * d2;
}

/* convert tensor to 3 dimensions with specific sizes */
static inline void ft_reshape_3d(ft_t *t, size_t d1, size_t d2, size_t d3)
{
	assert(d1 * d2 * d3 <= t->maxcap);

	t->ndim = 3;
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->dim[2] = d3;
	t->totlen = d1 * d2 * d3;
}

/* convert tensor to 4 dimensions with specific sizes */
static inline void ft_reshape_4d(ft_t *t, size_t d1, size_t d2, size_t d3, size_t d4)
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
static inline void ft_resize(ft_t *t, size_t d)
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

static inline void ft_resize_2d(ft_t *t, size_t d1, size_t d2)
{
	assert(t->ndim == 2);
	assert(d1 * d2 <= t->maxcap);

	t->dim[0] = d1;
	t->dim[1] = d2;
	t->totlen = d1 * d2;
}

/* returns a ft view over original ft */
#define ft_at(t, idx, view) \
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

static inline void ft_assert_1d(const ft_t *t, size_t d1)
{
	assert(t->ndim == 1);
	if (d1 > 0)
		assert(t->dim[0] == d1);
}

static inline void ft_assert_2d(const ft_t *t, size_t d1, size_t d2)
{
	assert(t->ndim == 2);
	if (d1 > 0)
		assert(t->dim[0] == d1);
	if (d2 > 0)
		assert(t->dim[1] == d2);
}

static inline void ft_assert_1d_match(const ft_t *lhs, const ft_t *rhs)
{
	assert(lhs->dim[0] == rhs->dim[0]);
}

static inline void ft_assert_2d_match(const ft_t *lhs, const ft_t *rhs)
{
	assert(lhs->dim[1] == rhs->dim[1]);
}

void ft_set(ft_t *ret, scalar_t val);
#define ft_set_inner(dst, dst_idx, src) \
do { \
	assert((dst)->ndim == 2); \
	assert((dst)->dim[1] == (src)->totlen); \
	assert((dst_idx) < (dst)->dim[0]); \
	__ft_set_inner((dst), (dst_idx), (src)); \
} while (0)

void __ft_set_inner(ft_t *dst, size_t dst_idx, const ft_t *src);
void ft_copy(ft_t *dst, const ft_t *src);
void ft_add(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs);
void ft_add_2x1(
	ft_t *dst,
	const ft_t *src,
	const ft_t *vec);
void ft_sub(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs);
void ft_mul(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs);
void ft_div(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs);
void ft_div_scalar(
	ft_t *ret,
	const ft_t *lhs,
	scalar_t scalar);
scalar_t ft_mean(const ft_t *lhs);
scalar_t ft_max(const ft_t *lhs, size_t *pos);

/* ret = lhs @ rhs + add */
void ft_mma_2x2(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs,
	const ft_t *add);
/* ret = lhs @ rhs.T + add */
void ft_mma_transposed_2x2(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs,
	const ft_t *add);
