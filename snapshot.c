#include "llm.h"

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>
#include <errno.h>

struct file_item {
	uint64_t off;
	uint64_t len;
};

struct file {
	size_t meta_len;
	size_t data_len;

	struct file_item *meta;
	void *data;

	/* current position for file tensor */
	size_t pos;
};

static void file_maybe_mlock(const char *path, void *p, size_t len)
{
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t num_pages = sysconf(_SC_PHYS_PAGES);
	size_t total_mem = page_size * num_pages;

	if (len > total_mem) {
		fprintf(stderr,
			"WARNING: '%s' doesn't fit into physical memory (%zu vz %zu), skipping mlock!\n",
			path, total_mem, len);
		return;
	}

	if (mlock(p, len) < 0) {
		fprintf(stderr,
			"WARNING: mlock('%s') failed with errno=%d (%zu vz %zu)!\n",
			path, errno, total_mem, len);
	}
}

/* 2 files:
 * - meta - contains the metadata about the chunks in the data file
 * - data - raw data
 *
 * meta is an array of 2 u64's:
 * - u64 offset in the data file
 * - u64 size in the data file
 */
static void *mmap_file(const char *path, size_t *len)
{
	struct stat st;
	void *p;
	int fd;

	if (stat(path, &st) < 0)
		return NULL;

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return NULL;

	p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	*len = st.st_size;

	file_maybe_mlock(path, p, *len);

	close(fd);

	return p;
}

struct file *file_load(const char *meta, const char *vocab)
{
	struct file *f = calloc(1, sizeof(*f));
	if (!f)
		return NULL;

	f->meta = mmap_file(meta, &f->meta_len);
	if (!f->meta)
		goto err;

	f->data = mmap_file(vocab, &f->data_len);
	if (!f->data)
		goto err;

	return f;

err:
	free(f);
	return NULL;
}

void file_close(struct file *f)
{
	if (f->meta)
		munmap(f->meta, f->meta_len);
	if (f->data)
		munmap(f->data, f->data_len);
	free(f);
}

size_t file_len(const struct file *f)
{
	return f->meta_len / sizeof(*f->meta);
}

void *file_at(const struct file *f, size_t pos)
{
	return f->data + f->meta[pos].off;
}

size_t file_at_len(const struct file *f, size_t pos)
{
	return f->meta[pos].len;
}

bool file_is_eof(struct file *f)
{
	return f->pos >= file_len(f);
}

tensor_t *file_tensor(struct file *f, size_t ndim, ...)
{
	tensor_t *t;
	size_t len = 1;
	va_list ap;
	size_t dim;

	assert((uintptr_t)file_at(f, f->pos) % VECTOR_ALIGN == 0);
	assert(ndim >= 1 && ndim <= 4);

	len = file_at_len(f, f->pos);
	t = tensor_new_mapped(file_at(f, f->pos), len);

	f->pos++;

	va_start(ap, ndim);
	size_t d1 = va_arg(ap, size_t);
	size_t d2 = va_arg(ap, size_t);
	size_t d3 = va_arg(ap, size_t);
	size_t d4 = va_arg(ap, size_t);

	switch (ndim) {
	case 1:
		tensor_reshape_1d(t, d1);
		break;
	case 2:
		tensor_reshape_2d(t, d1, d2);
		break;
	case 3:
		tensor_reshape_3d(t, d1, d2, d3);
		break;
	case 4:
		tensor_reshape_4d(t, d1, d2, d3, d4);
		break;
	}
	va_end(ap);

	assert(t->totlen * sizeof(scalar_t) == len);

	return t;
}

struct snapshot {
	FILE *config;
	struct file *file_param;
	struct file *file_vocab;
};

struct snapshot *snapshot_load(const char *path)
{
	struct snapshot *ss = calloc(1, sizeof(*ss));
	if (!ss)
		return NULL;

	ss->config = fopen(path, "r");
	if (!ss->config)
		goto err;

	return ss;

err:
	free(ss);
	return NULL;
}

void snapshot_close(struct snapshot *ss)
{
	fclose(ss->config);
	free(ss);
}

char *snapshot_config_str(struct snapshot *ss, const char *k)
{
	char *line = NULL;
	size_t len = 0;

	fseek(ss->config, 0, SEEK_SET);

	while (getline(&line, &len, ss->config) >= 0) {
		if (strstr(line, k) != line)
			continue;

		char *v = line + strlen(k);
		assert(*v == ' ');
		v += 1;

		v = strdup(v);

		char *eov = strchr(v, '#');
		if (eov)
			*eov = '\0';
		eov = strchr(v, '\n');
		if (eov)
			*eov = '\0';

		free(line);
		return v;
	}

	return NULL;
}

int snapshot_config_int(struct snapshot *ss, const char *k)
{
	char *v = snapshot_config_str(ss, k);
	int ret;

	assert(v);
	ret = atoi(v);
	free(v);

	return ret;
}

struct file *snapshot_param(struct snapshot *ss)
{
	if (!ss->file_param)
		ss->file_param = file_load(snapshot_config_str(ss, "param_meta"),
					   snapshot_config_str(ss, "param_data"));
	return ss->file_param;
}

struct file *snapshot_vocab(struct snapshot *ss)
{
	if (!ss->file_vocab)
		ss->file_vocab = file_load(snapshot_config_str(ss, "vocab_meta"),
					   snapshot_config_str(ss, "vocab_data"));
	return ss->file_vocab;
}
