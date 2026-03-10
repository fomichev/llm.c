#!/usr/bin/env python3
#
# Convert huggingface GPT-2 weights into llm.c-compatible format:
# 1. gpt2.llmc - key-value pairs of configuration
# 2. gpt2_param.data - weights
# 3. gpt2_param.meta - metadata about weight sizes/offsets in the data file
# 4. gpt2_vocab.data - vocabulary
# 5. gpt2_vocab.meta - metadata about vocabulary sizes/offsets in the data file

from itertools import chain
import struct
import sys
import torch
import json
from transformers import GPT2Tokenizer, GPT2Model

tensor_padding = 64 # SIMD
name = sys.argv[3]

transpose = [
    #'attn.c_attn.weight',
    #'attn.c_proj.weight',
    #'mlp.c_fc.weight',
    #'mlp.c_proj.weight',
]

model = GPT2Model.from_pretrained(sys.argv[1])
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(model)
print(tokenizer)

def write_tensor(file_meta, file_data, key, val):
    transposed = ''
    for i in transpose:
        if key.endswith(i):
            val = val.transpose(-2, -1).contiguous()
            transposed = '.T'

    pos = file_data.tell()
    flat = val.numpy().flat
    for x in flat:
        file_data.write(struct.pack('f', x))
    size = file_data.tell() - pos

    size_padded = (size + tensor_padding - 1) // tensor_padding * tensor_padding
    padding = size_padded - size
    for x in range(padding):
        file_data.write(struct.pack('c', bytes([0])))

    file_meta.write(struct.pack('QQ', pos, size))

    print(key + transposed, pos, size, val.shape, padding)

def convert_weights(model):
    state_dict = model.state_dict()
    C = model.wpe.num_embeddings
    num_layers = len(model.h)

    with open(f'{name}_param.meta', 'wb') as file_meta, open(f'{name}_param.data', 'wb') as file_data:
        write_tensor(file_meta, file_data, 'wte.weight', state_dict['wte.weight'])
        write_tensor(file_meta, file_data, 'wpe.weight', state_dict['wpe.weight'])

        for i in range(num_layers):
            p = f'h.{i}.'
            write_tensor(file_meta, file_data, p + 'ln_1.weight', state_dict[p + 'ln_1.weight'])
            write_tensor(file_meta, file_data, p + 'ln_1.bias', state_dict[p + 'ln_1.bias'])

            # Newer transformers removed attn.bias (causal mask) and
            # attn.masked_bias from the state dict. Generate them if missing.
            attn_bias_key = p + 'attn.bias'
            if attn_bias_key in state_dict:
                write_tensor(file_meta, file_data, attn_bias_key, state_dict[attn_bias_key])
            else:
                causal_mask = torch.tril(torch.ones(1, 1, C, C, dtype=torch.float32))
                write_tensor(file_meta, file_data, attn_bias_key + ' (generated)', causal_mask)

            masked_bias_key = p + 'attn.masked_bias'
            if masked_bias_key in state_dict:
                write_tensor(file_meta, file_data, masked_bias_key, state_dict[masked_bias_key])
            else:
                write_tensor(file_meta, file_data, masked_bias_key + ' (generated)', torch.tensor([-1e4], dtype=torch.float32))

            write_tensor(file_meta, file_data, p + 'attn.c_attn.weight', state_dict[p + 'attn.c_attn.weight'])
            write_tensor(file_meta, file_data, p + 'attn.c_attn.bias', state_dict[p + 'attn.c_attn.bias'])
            write_tensor(file_meta, file_data, p + 'attn.c_proj.weight', state_dict[p + 'attn.c_proj.weight'])
            write_tensor(file_meta, file_data, p + 'attn.c_proj.bias', state_dict[p + 'attn.c_proj.bias'])
            write_tensor(file_meta, file_data, p + 'ln_2.weight', state_dict[p + 'ln_2.weight'])
            write_tensor(file_meta, file_data, p + 'ln_2.bias', state_dict[p + 'ln_2.bias'])
            write_tensor(file_meta, file_data, p + 'mlp.c_fc.weight', state_dict[p + 'mlp.c_fc.weight'])
            write_tensor(file_meta, file_data, p + 'mlp.c_fc.bias', state_dict[p + 'mlp.c_fc.bias'])
            write_tensor(file_meta, file_data, p + 'mlp.c_proj.weight', state_dict[p + 'mlp.c_proj.weight'])
            write_tensor(file_meta, file_data, p + 'mlp.c_proj.bias', state_dict[p + 'mlp.c_proj.bias'])

        write_tensor(file_meta, file_data, 'ln_f.weight', state_dict['ln_f.weight'])
        write_tensor(file_meta, file_data, 'ln_f.bias', state_dict['ln_f.bias'])

def convert_vocab(json):
    with open(f'{name}_vocab.meta', 'wb') as file_meta, open(f'{name}_vocab.data', 'wb') as file_data:
        i = 0
        for k, v in json.items():
            pos = file_data.tell()

            # https://github.com/openai/gpt-2/blob/master/src/encoder.py
            rg = chain(range(0, 32+1), range(127, 160+1), range(173, 173+1))
            for x in rg:
                u = chr(256 + x)
                k = k.replace(u, chr(x))

            file_data.write(bytes(k, encoding='utf=8'))
            file_data.write(struct.pack('c', bytes([0])))
            size = file_data.tell() - pos
            file_meta.write(struct.pack('QQ', pos, size))
            if v != i:
                raise Exception(f'unexpected token index {v}, expected {i}')
            i = i + 1

convert_weights(model)
with open(sys.argv[2]) as f:
    convert_vocab(json.load(f))

with open(f'{name}.llmc', 'w') as config:
    print('version', f'1', file=config);
    print('param_data', f'{name}_param.data', file=config);
    print('param_meta', f'{name}_param.meta', file=config);
    print('vocab_data', f'{name}_vocab.data', file=config);
    print('vocab_meta', f'{name}_vocab.meta', file=config);
    print('transposed', '0' if len(transpose) == 0 else 1, file=config);
    print('context', model.wpe.num_embeddings, file=config);
    print('head_len', model.h[0].attn.head_dim, file=config);
    print('heads', model.h[0].attn.num_heads, file=config);
    print('layers', len(model.h), file=config);
    print('embeddings', model.wte.embedding_dim, file=config);
    print('vocab_len', model.wte.num_embeddings, file=config);
