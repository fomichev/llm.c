#include "llm.h"
#include "simd.h"

#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <cblas.h>

#include "tensor.h"

static tensor_t *__tensor_new(void *data, size_t ndim)
{
	uint32_t ndim_sz = ndim;
	tensor_t *t;

	t = calloc(1, sizeof(*t) + sizeof(size_t));
	if (!t)
		return NULL;

	t->data = data;
	t->ndim = ndim;

	return t;
}

static void *__tensor_alloc_data(size_t len)
{
	size_t sz;
	void *p;

	sz = len * sizeof(scalar_t);
	sz = (sz + (VECTOR_ALIGN - 1)) / VECTOR_ALIGN * VECTOR_ALIGN;

	p = aligned_alloc(VECTOR_ALIGN, sz);
	memset(p, 0, sz);
	return p;
}

tensor_t *tensor_new_zero(size_t ndim, ...)
{
	tensor_t *t;
	size_t len = 1;
	va_list ap;

	t = __tensor_new(NULL, ndim);
	va_start(ap, ndim);
	for (size_t i = 0; i < ndim; i++) {
		t->dim[i] = va_arg(ap, size_t);
		len *= t->dim[i];
	}
	va_end(ap);

	t->data = __tensor_alloc_data(len);
	assert(t->data);
	t->totlen = len;
	t->maxcap = len;

	return t;
}

tensor_t *tensor_new(size_t ndim, ...)
{
	tensor_t *t;
	size_t len = 1;
	va_list ap;

	t = __tensor_new(NULL, ndim);
	va_start(ap, ndim);
	for (size_t i = 0; i < ndim; i++) {
		t->dim[i] = va_arg(ap, size_t);
		len *= t->dim[i];
	}

	t->data = __tensor_alloc_data(len);
	assert(t->data);
	t->totlen = len;
	t->maxcap = len;

	for (size_t i = 0; i < len; i++) {
		t->data[i] = va_arg(ap, double); /* promotion */
	}
	va_end(ap);

	return t;
}

static tensor_t *__tensor_new_xd(tensor_t *t, va_list ap)
{
	t->data = __tensor_alloc_data(t->totlen);
	assert(t->data);

	for (size_t i = 0; i < t->totlen; i++) {
		t->data[i] = va_arg(ap, double); /* promotion */
	}

	return t;
}

tensor_t *tensor_new_1d(size_t d1, ...)
{
	tensor_t *t;
	va_list ap;

	t = __tensor_new(NULL, 1);
	t->dim[0] = d1;
	t->totlen = d1;

	va_start(ap, d1);
	__tensor_new_xd(t, ap);
	va_end(ap);

	return t;
}

tensor_t *tensor_new_2d(size_t d1, size_t d2, ...)
{
	tensor_t *t;
	va_list ap;

	t = __tensor_new(NULL, 2);
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->totlen = d1 * d2;

	va_start(ap, d2);
	__tensor_new_xd(t, ap);
	va_end(ap);

	return t;
}

tensor_t *tensor_new_3d(size_t d1, size_t d2, size_t d3, ...)
{
	tensor_t *t;
	va_list ap;

	t = __tensor_new(NULL, 3);
	t->dim[0] = d1;
	t->dim[1] = d2;
	t->dim[2] = d3;
	t->totlen = d1 * d2 * d3;

	va_start(ap, d3);
	__tensor_new_xd(t, ap);
	va_end(ap);

	return t;
}

void tensor_free(tensor_t *t)
{
	free(t->data);
	free(t);
}

tensor_t *tensor_new_mapped(void *data, size_t totlen)
{
	tensor_t *t;

	t = __tensor_new(data, 0);
	t->totlen = totlen;
	t->maxcap = totlen;

	return t;
}

void tensor_free_mapped(const tensor_t *t)
{
	free((void *)t);
}

static void __tensor_to_string(FILE *f, const tensor_t *t, int show_only)
{
	if (t->ndim == 1) {
		fprintf(f, "[");
		for (size_t i = 0; i < tensor_len(t); i++) {
			if (show_only && i == show_only) {
				size_t old_i = i;

				i = tensor_len(t) - show_only - 1;
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
	for (size_t i = 0; i < tensor_len(t); i++) {
		tensor_t view = {};

		if (show_only && i == show_only) {
			size_t old_i = i;

			i = tensor_len(t) - show_only - 1;
			fprintf(f, "\n [..%zu..]\n ", i - old_i + 1);
			continue;
		}

		tensor_at(t, i, &view);
		__tensor_to_string(f, &view, show_only);

		if (show_only && i + 1 != tensor_len(t) && i + 1 != show_only)
			fprintf(f, "\n ");
	}
	fprintf(f, "]");
}

char *tensor_to_string(const tensor_t *t)
{
	FILE *f;
	char *ptr;
	size_t size;

	f = open_memstream(&ptr, &size);
	__tensor_to_string(f, t, 0);
	fclose(f);

	return ptr;
}

char *tensor_to_short_string(const tensor_t *t)
{
	FILE *f;
	char *ptr;
	size_t size;

	f = open_memstream(&ptr, &size);
	__tensor_to_string(f, t, 2);
	fclose(f);

	return ptr;
}

void tensor_pick_rows(tensor_t *dst, const tensor_t *src, const int *rows, size_t num)
{
	tensor_assert_2d(dst, 0, 0);
	tensor_assert_2d(src, 0, 0);
	tensor_assert_2d_match(dst, src);

	tensor_resize(dst, num);

	size_t m = src->dim[1];

	for (size_t i = 0; i < num; i++) {
		tensor_t row;

		assert(rows[i] < tensor_len(src));

		tensor_at(src, rows[i], &row);
		tensor_assert_1d(&row, m);

		memcpy(&dst->data[i * m], row.data, sizeof(scalar_t) * m);
	}

	dst->dim[0] = num;
	dst->totlen = dst->dim[0] * dst->dim[1];
}

static void __tensor_same_size(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs)
{
	assert(lhs->totlen == ret->totlen);
	if (rhs)
		assert(lhs->totlen == rhs->totlen);
}

void tensor_set(tensor_t *ret, scalar_t val)
{
	vector_t v;

	if (val == 0) {
		memset(ret->data, 0, sizeof(ret->totlen * sizeof(scalar_t)));
		return;
	}

	vector_set(&v, val);

	for (size_t i = 0; i < vector_batches(ret->totlen); i += VECTOR_BATCH) {
		vector_store(&ret->data[i], &v);
	}

	for (size_t i = vector_batches(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = val;
	}
}

void __tensor_set_inner(tensor_t *dst, size_t dst_idx, const tensor_t *src)
{
	memcpy(&dst->data[dst_idx * dst->dim[1]], src->data, src->totlen * sizeof(scalar_t));
}

void tensor_copy(tensor_t *dst, const tensor_t *src)
{
	assert(dst->totlen == src->totlen);
	assert(dst->ndim == src->ndim);
	assert(dst->dim[0] == src->dim[0]);
	memcpy(dst->data, src->data, src->totlen * sizeof(scalar_t));
}

void tensor_add(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs)
{
	vector_t r, l;

	__tensor_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < vector_batches(ret->totlen); i += VECTOR_BATCH) {
		vector_load(&l, &lhs->data[i]);
		vector_load(&r, &rhs->data[i]);
		vector_add(&r, &r, &l);
		vector_store(&ret->data[i], &r);
	}

	for (size_t i = vector_batches(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] + rhs->data[i];
	}
}

void tensor_add_2x1(
	tensor_t *dst,
	const tensor_t *src,
	const tensor_t *vec)
{
	assert(src->ndim == 2);
	assert(vec->ndim == 1);
	assert(src->totlen == vec->totlen * src->dim[0]);

	for (size_t i = 0; i < tensor_len(src); i++) {
		tensor_t row;

		tensor_at(src, i, &row);
		tensor_add(&row, &row, vec);
	}
}

void tensor_sub(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs)
{
	vector_t r, l;

	__tensor_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < vector_batches(ret->totlen); i += VECTOR_BATCH) {
		vector_load(&l, &lhs->data[i]);
		vector_load(&r, &rhs->data[i]);
		vector_sub(&r, &r, &l);
		vector_store(&ret->data[i], &r);
	}

	for (size_t i = vector_batches(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] - rhs->data[i];
	}
}

void tensor_mul(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs)
{
	vector_t r, l;

	__tensor_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < vector_batches(ret->totlen); i += VECTOR_BATCH) {
		vector_load(&l, &lhs->data[i]);
		vector_load(&r, &rhs->data[i]);
		vector_mul(&r, &r, &l);
		vector_store(&ret->data[i], &r);
	}

	for (size_t i = vector_batches(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] * rhs->data[i];
	}
}

void tensor_div(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs)
{
	vector_t r, l;

	__tensor_same_size(ret, lhs, rhs);

	for (size_t i = 0; i < vector_batches(ret->totlen); i += VECTOR_BATCH) {
		vector_load(&l, &lhs->data[i]);
		vector_load(&r, &rhs->data[i]);
		vector_div(&r, &r, &l);
		vector_store(&ret->data[i], &r);
	}

	for (size_t i = vector_batches(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] / rhs->data[i];
	}
}

void tensor_div_scalar(
	tensor_t *ret,
	const tensor_t *lhs,
	scalar_t scalar)
{
	vector_t vscalar;
	vector_t vtmp;

	__tensor_same_size(ret, lhs, NULL);

	vector_set(&vscalar, scalar);

	for (size_t i = 0; i < vector_batches(ret->totlen); i += VECTOR_BATCH) {
		vector_load(&vtmp, &lhs->data[i]);
		vector_div(&vtmp, &vtmp, &vscalar);
		vector_store(&ret->data[i], &vtmp);
	}

	for (size_t i = vector_batches(ret->totlen); i < ret->totlen; i++) {
		ret->data[i] = lhs->data[i] / scalar;
	}
}

scalar_t tensor_mean(const tensor_t *lhs)
{
	size_t nr = 0;
	vector_t t, s;

	vector_set(&s, 0);

	for (size_t i = 0; i < vector_batches(lhs->totlen); i += VECTOR_BATCH) {
		vector_load(&t, &lhs->data[i]);
		vector_add(&s, &s, &t);
		nr += VECTOR_BATCH;
	}

	scalar_t sum = vector_reduce_sum(&s);
	for (size_t i = vector_batches(lhs->totlen); i < lhs->totlen; i++) {
		sum += lhs->data[i];
		nr++;
	}

	return sum / nr;
}

scalar_t tensor_max(const tensor_t *lhs, size_t *pos)
{
	size_t max = lhs->data[0];
	size_t max_pos = 0;
	size_t res;
	vector_t t;

	for (size_t i = 0; i < vector_batches(lhs->totlen); i += VECTOR_BATCH) {
		vector_load(&t, &lhs->data[i]);
		res = vector_reduce_max(&t);
		if (res > max) {
			max = res;
			max_pos = i;
		}
	}

	for (size_t i = vector_batches(lhs->totlen); i < lhs->totlen; i++) {
		if (lhs->data[i] > max) {
			max = lhs->data[i];
			max_pos = i;
		}
	}

	if (pos) {
		size_t tail = lhs->totlen - max_pos;
		if (tail > VECTOR_BATCH)
			tail = VECTOR_BATCH;

		for (size_t i = 0; i < tail; i++) {
			if (lhs->data[max_pos + i] == max) {
				*pos = max_pos + i;
				break;
			}
		}
	}

	return max;
}

void tensor_mma_2x2(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
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

void tensor_mma_transposed_2x2(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
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
