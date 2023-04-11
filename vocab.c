#include "llm.h"

int vocab_decode(struct file *f, const char *s, int *sz)
{
	int max_len = 0;
	int max_i = -1;

	if (*s == 0)
		return -1;

	/* In theory, something more efficient than O(n) could be used here.
	 * In practice, this is the operation we do at the beginning,
	 * so it's probably ok.
	 */

	for (size_t i = 0; i < file_len(f); i++) {
		const char *tok = file_at(f, i);

		if (strstr(s, tok) == s) {
			if (max_len > file_at_len(f, i))
				continue;

			max_len = file_at_len(f, i);
			max_i = i;
		}
	}

	*sz = max_len - 1;
	return max_i;
}

const char *vocab_encode(struct file *f, size_t token)
{
	assert(token < file_len(f));
	return file_at(f, token);
}
