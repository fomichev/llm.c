#pragma once

#include "tensor.h"

void layer_norm(
	tensor_t *ln,
	tensor_t *tmp_mat,
	const tensor_t *weight,
	const tensor_t *bias);

void softmax_1d(tensor_t *t);
void softmax_2d(tensor_t *t);
void gelua(tensor_t *t);
