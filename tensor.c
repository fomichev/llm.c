#include "llm.h"
#include "simd.h"

#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <cblas.h>

#include "tensor.h"

static ft_t *__ft_new(void *data, size_t ndim)
{
	uint32_t ndim_sz = ndim;
	ft_t *t;

	t = calloc(1, sizeof(*t) + sizeof(size_t));
	if (!t)
		return NULL;

	t->data = data;
	t->ndim = ndim;

	return t;
}

static void *__ft_alloc_data(size_t len)
{
	size_t sz;
	void *p;

	sz = len * sizeof(scalar_t);
	sz = (sz + (FV_ALIGN - 1)) / FV_ALIGN * FV_ALIGN;

	p = aligned_alloc(FV_ALIGN, sz);
	memset(p, 0, sz);
	return p;
}

ft_t *ft_new_zero(size_t ndim, ...)
{
	ft_t *t;
	size_t len = 1;
	va_list ap;

	t = __ft_new(NULL, ndim);
	va_start(ap, ndim);
	for (size_t i = 0; i < ndim; i++) {
		t->dim[i] = va_arg(ap, size_t);
		len *= t->dim[i];
	}
	va_end(ap);

	t->data = __ft_alloc_data(len);
	assert(t->data);
	t->totlen = len;
	t->maxcap = len;

	return t;
}

ft_t *ft_new(size_t ndim, ...)
{
	ft_t *t;
	size_t len = 1;
	va_list ap;

	t = __ft_new(NULL, ndim);
	va_start(ap, ndim);
	for (size_t i = 0; i < ndim; i++) {
		t->dim[i] = va_arg(ap, size_t);
		len *= t->dim[i];
	}

	t->data = __ft_alloc_data(len);
	assert(t->data);
	t->totlen = len;
	t->maxcap = len;

	for (size_t i = 0; i < len; i++) {
		t->data[i] = va_arg(ap, double); /* promotion */
	}
	va_end(ap);

	return t;
}

static ft_t *__ft_new_xd(ft_t *t, va_list ap)
{
	t->data = __ft_alloc_data(t->totlen);
	assert(t->data);

	for (size_t i = 0; i < t->totlen; i++) {
		t->data[i] = va_arg(ap, double); /* promotion */
	}

	return t;
}

ft_t *ft_new_1d(size_t d1, ...)
{
	ft_t *t;
	va_list ap;

	t = __ft_new(NULL, 1);
	t->dim[0] = d1;
	t->totlen = d1;

	va_start(ap, d1);
	__ft_new_xd(t, ap);
	va_end(ap);

	return t;
}

ft_t *ft_new_2d(size_t d1, size_t d2, ...)
{
	ft_t *t;
	va_list ap;

	t = __ft_new(NULL, 2);
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->totlen = d1 * d2;

	va_start(ap, d2);
	__ft_new_xd(t, ap);
	va_end(ap);

	return t;
}

ft_t *ft_new_3d(size_t d1, size_t d2, size_t d3, ...)
{
	ft_t *t;
	va_list ap;

	t = __ft_new(NULL, 3);
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->dim[2] = d3;
	t->totlen = d1 * d2 * d3;

	va_start(ap, d3);
	__ft_new_xd(t, ap);
	va_end(ap);

	return t;
}

void ft_free(ft_t *t)
{
	free(t->data);
	free(t);
}

ft_t *ft_new_mapped(void *data, size_t totlen)
{
	ft_t *t;

	t = __ft_new(data, 0);
	t->totlen = totlen;
	t->maxcap = totlen;

	return t;
}

void ft_free_mapped(const ft_t *t)
{
	free((void *)t);
}

static void __ft_to_string(FILE *f, const ft_t *t, int show_only)
{
	if (t->ndim == 1) {
		fprintf(f, "[");
		for (size_t i = 0; i < ft_len(t); i++) {
			if (show_only && i == show_only) {
				size_t old_i = i;

				i = ft_len(t) - show_only - 1;
				fprintf(f, " ..%zu..", i - old_i);
				continue;
			}

			if (i != 0) {
				fprintf(f, " ");
			}
			fprintf(f, "%+.2e", t->data[i]);

		}
		fprintf(f, "]");
		return;
	}

	fprintf(f, "[");
	for (size_t i = 0; i < ft_len(t); i++) {
		ft_t view = {};

		if (show_only && i == show_only) {
			size_t old_i = i;

			i = ft_len(t) - show_only - 1;
			fprintf(f, "\n [..%zu..]\n ", i - old_i + 1);
			continue;
		}

		ft_at(t, i, &view);
		__ft_to_string(f, &view, show_only);

		if (show_only && i + 1 != ft_len(t) && i + 1 != show_only)
			fprintf(f, "\n ");
	}
	fprintf(f, "]");
}

char *ft_to_string(const ft_t *t)
{
	FILE *f;
	char *ptr;
	size_t size;

	f = open_memstream(&ptr, &size);
	__ft_to_string(f, t, 0);
	fclose(f);

	return ptr;
}

char *ft_to_short_string(const ft_t *t)
{
	FILE *f;
	char *ptr;
	size_t size;

	f = open_memstream(&ptr, &size);
	__ft_to_string(f, t, 2);
	fclose(f);

	return ptr;
}

void ft_pick_rows(ft_t *dst, const ft_t *src, const int *rows, size_t num)
{
	ft_assert_2d(dst, 0, 0);
	ft_assert_2d(src, 0, 0);
	ft_assert_2d_match(dst, src);

	ft_resize(dst, num);

	size_t m = src->dim[1];

	for (size_t i = 0; i < num; i++) {
		ft_t row;

		assert(rows[i] < ft_len(src));

		ft_at(src, rows[i], &row);
		ft_assert_1d(&row, m);

		memcpy(&dst->data[i * m], row.data, sizeof(scalar_t) * m);
	}

	dst->dim[0] = num;
	dst->totlen = dst->dim[0] * dst->dim[1];
}

static void __ft_same_size(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs)
{
	assert(lhs->totlen == ret->totlen);
	if (rhs)
		assert(lhs->totlen == rhs->totlen);
}

void ft_set(ft_t *ret, scalar_t val)
{
	fv_t v;

	if (val == 0) {
		memset(ret->data, 0, sizeof(ret->totlen * sizeof(scalar_t)));
		return;
	}

	fv_load1(&v, val);

	for (size_t i = 0; i < fv_chunks(ret->totlen); i += FV_CHUNK) {
		fv_store(&ret->data[i], &v);
	}

	for (size_t i = fv_chunks(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = val;
	}
}

void __ft_set_inner(ft_t *dst, size_t dst_idx, const ft_t *src)
{
	memcpy(&dst->data[dst_idx * dst->dim[1]], src->data, src->totlen * sizeof(scalar_t));
}

void ft_copy(ft_t *dst, const ft_t *src)
{
	assert(dst->totlen == src->totlen);
	assert(dst->ndim == src->ndim);
	assert(dst->dim[0] == src->dim[0]);
	memcpy(dst->data, src->data, src->totlen * sizeof(scalar_t));
}

void ft_add(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs)
{
	fv_t r, l;

	__ft_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < fv_chunks(ret->totlen); i += FV_CHUNK) {
		fv_load(&l, &lhs->data[i]);
		fv_load(&r, &rhs->data[i]);
		fv_add(&r, &r, &l);
		fv_store(&ret->data[i], &r);
	}

	for (size_t i = fv_chunks(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] + rhs->data[i];
	}
}

void ft_add_2x1(
	ft_t *dst,
	const ft_t *src,
	const ft_t *vec)
{
	assert(src->ndim == 2);
	assert(vec->ndim == 1);
	assert(src->totlen == vec->totlen * src->dim[0]);

	for (size_t i = 0; i < ft_len(src); i++) {
		ft_t row;

		ft_at(src, i, &row);
		ft_add(&row, &row, vec);
	}
}

void ft_sub(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs)
{
	fv_t r, l;

	__ft_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < fv_chunks(ret->totlen); i += FV_CHUNK) {
		fv_load(&l, &lhs->data[i]);
		fv_load(&r, &rhs->data[i]);
		fv_sub(&r, &r, &l);
		fv_store(&ret->data[i], &r);
	}

	for (size_t i = fv_chunks(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] - rhs->data[i];
	}
}

void ft_mul(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs)
{
	fv_t r, l;

	__ft_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < fv_chunks(ret->totlen); i += FV_CHUNK) {
		fv_load(&l, &lhs->data[i]);
		fv_load(&r, &rhs->data[i]);
		fv_mul(&r, &r, &l);
		fv_store(&ret->data[i], &r);
	}

	for (size_t i = fv_chunks(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] * rhs->data[i];
	}
}

void ft_div(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs)
{
	fv_t r, l;

	__ft_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < fv_chunks(ret->totlen); i += FV_CHUNK) {
		fv_load(&l, &lhs->data[i]);
		fv_load(&r, &rhs->data[i]);
		fv_div(&r, &r, &l);
		fv_store(&ret->data[i], &r);
	}

	for (size_t i = fv_chunks(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] / rhs->data[i];
	}
}

void ft_div_scalar(
	ft_t *ret,
	const ft_t *lhs,
	scalar_t scalar)
{
	fv_t vscalar;
	fv_t vtmp;

	__ft_same_size(ret, lhs, NULL);

	fv_load1(&vscalar, scalar);

	for (size_t i = 0; i < fv_chunks(ret->totlen); i += FV_CHUNK) {
		fv_load(&vtmp, &lhs->data[i]);
		fv_div(&vtmp, &vtmp, &vscalar);
		fv_store(&ret->data[i], &vtmp);
	}

	for (size_t i = fv_chunks(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] / scalar;
	}
}

scalar_t ft_mean(const ft_t *lhs)
{
	size_t nr = 0;
	fv_t t, s;

	fv_load1(&s, 0);

	for (size_t i = 0; i < fv_chunks(lhs->totlen); i += FV_CHUNK) {
		fv_load(&t, &lhs->data[i]);
		fv_add(&s, &s, &t);
		nr += FV_CHUNK;
	}

	scalar_t sum = fv_reduce_sum(&s);
	for (size_t i = fv_chunks(lhs->totlen); i < lhs->totlen; i++) {
		sum += lhs->data[i];
		nr++;
	}

	return sum / nr;
}

scalar_t ft_max(const ft_t *lhs, size_t *pos)
{
	size_t max = lhs->data[0];
	size_t max_pos = 0;
	size_t res;
	fv_t t;

	for (size_t i = 0; i < fv_chunks(lhs->totlen); i += FV_CHUNK) {
		fv_load(&t, &lhs->data[i]);
		res = fv_reduce_max(&t);
		if (res > max) {
			max = res;
			max_pos = i;
		}
	}

	for (size_t i = fv_chunks(lhs->totlen); i < lhs->totlen; i++) {
		if (lhs->data[i] > max) {
			max = lhs->data[i];
			max_pos = i;
		}
	}

	if (pos) {
		size_t tail = lhs->totlen - max_pos;
		if (tail > FV_CHUNK)
			tail = FV_CHUNK;

		for (size_t i = 0; i < tail; i++) {
			if (lhs->data[max_pos + i] == max) {
				*pos = max_pos + i;
				break;
			}
		}
	}

	return max;
}

void ft_mma_2x2(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs,
	const ft_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[0]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[1];
	float beta = 0.0;

	ret->ndim = 2;
	ret->dim[0] = m;
	ret->dim[1] = n;
	ret->totlen = m * n;

	if (add) {
		if (ret->totlen == add->totlen) {
			memcpy(ret->data, add->data, add->totlen * sizeof(scalar_t));
			beta = 1.0;
		} else {
			assert(n == add->totlen);

			for (size_t i = 0; i < m; i++)
				memcpy(&ret->data[i * n], add->data, add->totlen * sizeof(scalar_t));
			beta = 1.0;
		}
	}

	cblas_sgemm(/*layout=*/CblasRowMajor,
		    /*TransA=*/CblasNoTrans,
		    /*TransB=*/CblasNoTrans,
		    /*M=*/m, /* op(A).rows and C.rows  */
		    /*N=*/n, /* op(B).cols and C.cols */
		    /*K=*/k, /* op(A).cols and op(B).rows */
		    /*alpha=*/1.0,
		    /*A=*/lhs->data,
		    /*lda=*/k,
		    /*B=*/rhs->data,
		    /*ldb=*/n,
		    /*beta=*/beta,
		    /*C=*/ret->data,
		    /*ldc=*/n);
}

void ft_mma_transposed_2x2(
	ft_t *ret,
	const ft_t *lhs,
	const ft_t *rhs,
	const ft_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[1]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[0];
	float beta = 0.0;

	ret->ndim = 2;
	ret->dim[0] = m;
	ret->dim[1] = n;
	ret->totlen = m * n;

	if (add) {
		if (ret->totlen == add->totlen) {
			memcpy(ret->data, add->data, add->totlen * sizeof(scalar_t));
			beta = 1.0;
		} else {
			assert(n == add->totlen);

			for (size_t i = 0; i < m; i++)
				memcpy(&ret->data[i * n], add->data, add->totlen * sizeof(scalar_t));
			beta = 1.0;
		}
	}

	cblas_sgemm(/*layout=*/CblasRowMajor,
		    /*TransA=*/CblasNoTrans,
		    /*TransB=*/CblasTrans,
		    /*M=*/m, /* op(A).rows and C.rows  */
		    /*N=*/n, /* op(B).cols and C.cols */
		    /*K=*/k, /* op(A).cols and op(B).rows */
		    /*alpha=*/1.0,
		    /*A=*/lhs->data,
		    /*lda=*/k,
		    /*B=*/rhs->data,
		    /*ldb=*/rhs->dim[1],
		    /*beta=*/beta,
		    /*C=*/ret->data,
		    /*ldc=*/n);
}
