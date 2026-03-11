#include "gguf.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#define GGUF_MAGIC 0x46554747 /* "GGUF" in little-endian */
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

/* metadata value types */
enum gguf_type {
	GGUF_TYPE_UINT8   = 0,
	GGUF_TYPE_INT8    = 1,
	GGUF_TYPE_UINT16  = 2,
	GGUF_TYPE_INT16   = 3,
	GGUF_TYPE_UINT32  = 4,
	GGUF_TYPE_INT32   = 5,
	GGUF_TYPE_FLOAT32 = 6,
	GGUF_TYPE_BOOL    = 7,
	GGUF_TYPE_STRING  = 8,
	GGUF_TYPE_ARRAY   = 9,
	GGUF_TYPE_UINT64  = 10,
	GGUF_TYPE_INT64   = 11,
	GGUF_TYPE_FLOAT64 = 12,
};

/* tensor data types (only f32 supported for now) */
enum ggml_type {
	GGML_TYPE_F32  = 0,
	GGML_TYPE_F16  = 1,
};

struct gguf_str {
	uint64_t len;
	char *str; /* NOT null-terminated in the file; we null-terminate on parse */
};

struct gguf_kv {
	struct gguf_str key;
	uint32_t type;

	union {
		uint8_t   u8;
		int8_t    i8;
		uint16_t  u16;
		int16_t   i16;
		uint32_t  u32;
		int32_t   i32;
		float     f32;
		uint64_t  u64;
		int64_t   i64;
		double    f64;
		uint8_t   b;
		struct gguf_str str;
		struct {
			uint32_t elem_type;
			uint64_t count;
			void *data;
		} arr;
	};
};

struct gguf_tensor_info {
	struct gguf_str name;
	uint32_t ndim;
	uint64_t dim[4];
	uint32_t type;
	uint64_t offset; /* relative to start of tensor data section */
};

struct gguf {
	void *data;
	size_t data_len;

	uint32_t version;
	uint64_t tensor_count;
	uint64_t metadata_kv_count;

	struct gguf_kv *kv;
	struct gguf_tensor_info *tensors;

	/* offset where tensor data begins in the file */
	size_t tensor_data_offset;

	uint32_t alignment;
};

/* cursor-based reader over the mmap'd file */
struct reader {
	const uint8_t *base;
	size_t pos;
	size_t len;
};

static const void *reader_read(struct reader *r, size_t n)
{
	assert(r->pos + n <= r->len);
	const void *p = r->base + r->pos;
	r->pos += n;
	return p;
}

static uint8_t read_u8(struct reader *r)
{
	return *(const uint8_t *)reader_read(r, 1);
}

static uint16_t read_u16(struct reader *r)
{
	uint16_t v;
	memcpy(&v, reader_read(r, 2), 2);
	return v;
}

static uint32_t read_u32(struct reader *r)
{
	uint32_t v;
	memcpy(&v, reader_read(r, 4), 4);
	return v;
}

static uint64_t read_u64(struct reader *r)
{
	uint64_t v;
	memcpy(&v, reader_read(r, 8), 8);
	return v;
}

static float read_f32(struct reader *r)
{
	float v;
	memcpy(&v, reader_read(r, 4), 4);
	return v;
}

static double read_f64(struct reader *r)
{
	double v;
	memcpy(&v, reader_read(r, 8), 8);
	return v;
}

static struct gguf_str read_str(struct reader *r)
{
	struct gguf_str s;
	s.len = read_u64(r);
	const char *p = reader_read(r, s.len);
	/* make a null-terminated copy */
	s.str = malloc(s.len + 1);
	assert(s.str);
	memcpy(s.str, p, s.len);
	s.str[s.len] = '\0';
	return s;
}

/* skip over a metadata value without storing it (used for array elements) */
static void skip_value(struct reader *r, uint32_t type)
{
	switch (type) {
	case GGUF_TYPE_UINT8:
	case GGUF_TYPE_INT8:
	case GGUF_TYPE_BOOL:
		reader_read(r, 1);
		break;
	case GGUF_TYPE_UINT16:
	case GGUF_TYPE_INT16:
		reader_read(r, 2);
		break;
	case GGUF_TYPE_UINT32:
	case GGUF_TYPE_INT32:
	case GGUF_TYPE_FLOAT32:
		reader_read(r, 4);
		break;
	case GGUF_TYPE_UINT64:
	case GGUF_TYPE_INT64:
	case GGUF_TYPE_FLOAT64:
		reader_read(r, 8);
		break;
	case GGUF_TYPE_STRING: {
		struct gguf_str s = read_str(r);
		free(s.str);
		break;
	}
	case GGUF_TYPE_ARRAY: {
		uint32_t et = read_u32(r);
		uint64_t n = read_u64(r);
		for (uint64_t i = 0; i < n; i++)
			skip_value(r, et);
		break;
	}
	default:
		fprintf(stderr, "gguf: unknown value type %u\n", type);
		assert(0);
	}
}

static void read_kv(struct reader *r, struct gguf_kv *kv)
{
	kv->key = read_str(r);
	kv->type = read_u32(r);

	switch (kv->type) {
	case GGUF_TYPE_UINT8:   kv->u8  = read_u8(r);  break;
	case GGUF_TYPE_INT8:    kv->i8  = (int8_t)read_u8(r); break;
	case GGUF_TYPE_UINT16:  kv->u16 = read_u16(r); break;
	case GGUF_TYPE_INT16:   kv->i16 = (int16_t)read_u16(r); break;
	case GGUF_TYPE_UINT32:  kv->u32 = read_u32(r); break;
	case GGUF_TYPE_INT32:   kv->i32 = (int32_t)read_u32(r); break;
	case GGUF_TYPE_FLOAT32: kv->f32 = read_f32(r); break;
	case GGUF_TYPE_BOOL:    kv->b   = read_u8(r);  break;
	case GGUF_TYPE_STRING:  kv->str = read_str(r);  break;
	case GGUF_TYPE_UINT64:  kv->u64 = read_u64(r); break;
	case GGUF_TYPE_INT64:   kv->i64 = (int64_t)read_u64(r); break;
	case GGUF_TYPE_FLOAT64: kv->f64 = read_f64(r); break;
	case GGUF_TYPE_ARRAY:
		kv->arr.elem_type = read_u32(r);
		kv->arr.count = read_u64(r);
		/* we don't store array values, just skip them */
		for (uint64_t i = 0; i < kv->arr.count; i++)
			skip_value(r, kv->arr.elem_type);
		break;
	default:
		fprintf(stderr, "gguf: unknown kv type %u for key '%s'\n",
			kv->type, kv->key.str);
		assert(0);
	}
}

static void read_tensor_info(struct reader *r, struct gguf_tensor_info *t)
{
	t->name = read_str(r);
	t->ndim = read_u32(r);
	assert(t->ndim <= 4);

	for (uint32_t i = 0; i < t->ndim; i++)
		t->dim[i] = read_u64(r);
	for (uint32_t i = t->ndim; i < 4; i++)
		t->dim[i] = 0;

	t->type = read_u32(r);
	t->offset = read_u64(r);
}

static size_t align_up(size_t v, size_t alignment)
{
	return (v + alignment - 1) & ~(alignment - 1);
}

struct gguf *gguf_load(const char *path)
{
	struct stat st;
	int fd;
	void *data;

	if (stat(path, &st) < 0) {
		fprintf(stderr, "gguf: stat('%s'): %s\n", path, strerror(errno));
		return NULL;
	}

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "gguf: open('%s'): %s\n", path, strerror(errno));
		return NULL;
	}

	data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (data == MAP_FAILED) {
		fprintf(stderr, "gguf: mmap('%s'): %s\n", path, strerror(errno));
		return NULL;
	}

	struct reader r = { .base = data, .pos = 0, .len = st.st_size };

	/* header */
	uint32_t magic = read_u32(&r);
	if (magic != GGUF_MAGIC) {
		fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n",
			magic, GGUF_MAGIC);
		munmap(data, st.st_size);
		return NULL;
	}

	struct gguf *g = calloc(1, sizeof(*g));
	assert(g);
	g->data = data;
	g->data_len = st.st_size;

	g->version = read_u32(&r);
	if (g->version < 2 || g->version > 3) {
		fprintf(stderr, "gguf: unsupported version %u\n", g->version);
		munmap(data, st.st_size);
		free(g);
		return NULL;
	}

	g->tensor_count = read_u64(&r);
	g->metadata_kv_count = read_u64(&r);
	g->alignment = GGUF_DEFAULT_ALIGNMENT;

	/* metadata */
	g->kv = calloc(g->metadata_kv_count, sizeof(*g->kv));
	assert(g->kv);
	for (uint64_t i = 0; i < g->metadata_kv_count; i++) {
		read_kv(&r, &g->kv[i]);

		/* check for alignment override */
		if (g->kv[i].type == GGUF_TYPE_UINT32 &&
		    strcmp(g->kv[i].key.str, "general.alignment") == 0)
			g->alignment = g->kv[i].u32;
	}

	/* tensor info */
	g->tensors = calloc(g->tensor_count, sizeof(*g->tensors));
	assert(g->tensors);
	for (uint64_t i = 0; i < g->tensor_count; i++)
		read_tensor_info(&r, &g->tensors[i]);

	/* tensor data starts after the header, aligned */
	g->tensor_data_offset = align_up(r.pos, g->alignment);

	fprintf(stderr, "gguf: loaded '%s' v%u, %lu tensors, %lu metadata keys\n",
		path, g->version, g->tensor_count, g->metadata_kv_count);

	return g;
}

void gguf_close(struct gguf *g)
{
	for (uint64_t i = 0; i < g->metadata_kv_count; i++) {
		free(g->kv[i].key.str);
		if (g->kv[i].type == GGUF_TYPE_STRING)
			free(g->kv[i].str.str);
	}
	free(g->kv);

	for (uint64_t i = 0; i < g->tensor_count; i++)
		free(g->tensors[i].name.str);
	free(g->tensors);

	munmap(g->data, g->data_len);
	free(g);
}

size_t gguf_tensor_count(const struct gguf *g)
{
	return g->tensor_count;
}

size_t gguf_metadata_count(const struct gguf *g)
{
	return g->metadata_kv_count;
}

static const struct gguf_kv *gguf_find_kv(const struct gguf *g, const char *key)
{
	for (uint64_t i = 0; i < g->metadata_kv_count; i++) {
		if (strcmp(g->kv[i].key.str, key) == 0)
			return &g->kv[i];
	}
	return NULL;
}

const char *gguf_get_str(const struct gguf *g, const char *key)
{
	const struct gguf_kv *kv = gguf_find_kv(g, key);
	if (!kv || kv->type != GGUF_TYPE_STRING)
		return NULL;
	return kv->str.str;
}

uint64_t gguf_get_uint64(const struct gguf *g, const char *key)
{
	const struct gguf_kv *kv = gguf_find_kv(g, key);
	assert(kv);
	switch (kv->type) {
	case GGUF_TYPE_UINT64: return kv->u64;
	case GGUF_TYPE_UINT32: return kv->u32;
	case GGUF_TYPE_UINT16: return kv->u16;
	case GGUF_TYPE_UINT8:  return kv->u8;
	case GGUF_TYPE_INT64:  return (uint64_t)kv->i64;
	case GGUF_TYPE_INT32:  return (uint64_t)kv->i32;
	default:
		fprintf(stderr, "gguf: key '%s' is not an integer (type=%u)\n",
			key, kv->type);
		assert(0);
		return 0;
	}
}

uint32_t gguf_get_uint32(const struct gguf *g, const char *key)
{
	return (uint32_t)gguf_get_uint64(g, key);
}

float gguf_get_float32(const struct gguf *g, const char *key)
{
	const struct gguf_kv *kv = gguf_find_kv(g, key);
	assert(kv);
	if (kv->type == GGUF_TYPE_FLOAT32)
		return kv->f32;
	if (kv->type == GGUF_TYPE_FLOAT64)
		return (float)kv->f64;
	fprintf(stderr, "gguf: key '%s' is not a float (type=%u)\n",
		key, kv->type);
	assert(0);
	return 0;
}

static const struct gguf_tensor_info *gguf_find_tensor(const struct gguf *g,
						       const char *name)
{
	for (uint64_t i = 0; i < g->tensor_count; i++) {
		if (strcmp(g->tensors[i].name.str, name) == 0)
			return &g->tensors[i];
	}
	return NULL;
}

tensor_t *gguf_tensor(const struct gguf *g, const char *name, size_t ndim, ...)
{
	const struct gguf_tensor_info *ti = gguf_find_tensor(g, name);
	if (!ti) {
		fprintf(stderr, "gguf: tensor '%s' not found\n", name);
		return NULL;
	}

	if (ti->type != GGML_TYPE_F32) {
		fprintf(stderr, "gguf: tensor '%s' is type %u, only f32 supported\n",
			name, ti->type);
		return NULL;
	}

	/* compute total elements from the GGUF-stored dimensions */
	size_t total_elements = 1;
	for (uint32_t i = 0; i < ti->ndim; i++)
		total_elements *= ti->dim[i];
	size_t byte_size = total_elements * sizeof(scalar_t);

	void *tensor_data = (uint8_t *)g->data + g->tensor_data_offset + ti->offset;

	tensor_t *t = tensor_new_mapped(tensor_data, byte_size);

	va_list ap;
	va_start(ap, ndim);
	size_t d1 = va_arg(ap, size_t);
	size_t d2 = va_arg(ap, size_t);
	size_t d3 = va_arg(ap, size_t);
	size_t d4 = va_arg(ap, size_t);

	switch (ndim) {
	case 1: tensor_reshape_1d(t, d1); break;
	case 2: tensor_reshape_2d(t, d1, d2); break;
	case 3: tensor_reshape_3d(t, d1, d2, d3); break;
	case 4: tensor_reshape_4d(t, d1, d2, d3, d4); break;
	default: assert(0);
	}
	va_end(ap);

	assert(t->totlen == total_elements);

	return t;
}
