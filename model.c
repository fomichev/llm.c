#include "model.h"

#include <string.h>
#include <stdio.h>

#define MAX_MODELS 16

static const struct model *models[MAX_MODELS];
static int num_models;

void register_model(const struct model *m)
{
	if (num_models >= MAX_MODELS) {
		fprintf(stderr, "too many models registered\n");
		return;
	}
	models[num_models++] = m;
}

const struct model *find_model(const char *name)
{
	for (int i = 0; i < num_models; i++) {
		if (strcmp(models[i]->name, name) == 0)
			return models[i];
	}
	return NULL;
}
