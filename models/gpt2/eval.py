#!/usr/bin/env python3
#
# Compare the output of three GPT-2 models:
# 1. Completely stock one (model_default)
# 2. Stock one with it's modules called explicitly (model_expanded)
# 3. Custom one using only the weights from the stock one (model_fully_expanded)
#
# Evaluate 10 tokens. Always pick the ones with the highest probability
# to make the results stable.

import sys
import torch
import math
import time
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import pipeline, set_seed

def generate_tok_k():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    m = GPT2LMHeadModel.from_pretrained('gpt2')

    inp = tokenizer("In the morning I was able to", return_tensors='pt')

    begin = time.time()
    sample_output = m.generate(
            **inp,
            do_sample=True,
            max_length=500,
            top_k=5)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
    end = time.time()
    print(f'total={(end-begin)}s')

def generate_pipeline():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    begin = time.time()
    text = generator("In the morning I was able to", max_length=500, num_return_sequences=1)
    end = time.time()
    print(text)
    print(f'total={(end-begin)}s')

#generate_tok_k()
#generate_pipeline()
#sys.exit(0)

model = GPT2Model.from_pretrained(sys.argv[1])
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

profiler_last = 0
profiler_points = None
profiler_names = [ "" for _ in range(16) ]
profiler_started = False

def profiler_start():
    global profiler_last
    global profiler_points
    global profiler_names
    global profiler_started

    profiler_last = time.time()
    profiler_started = True
    profiler_points = [ 0 for _ in range(16) ]

def profiler_report():
    global profiler_last
    global profiler_points
    global profiler_names
    global profiler_started

    if not profiler_started:
        return

    total = 0
    for i in range(16):
        if profiler_points[i] == 0:
            continue

        val = profiler_points[i]
        total += val
        print('%.9f %s' % (val, profiler_names[i]))
    print('total: %.9f' % (total))

def profiler_record(index, name):
    global profiler_last
    global profiler_points
    global profiler_names
    global profiler_started

    if not profiler_started:
        return

    now = time.time()
    profiler_points[index] += now - profiler_last
    profiler_names[index] = name
    profiler_last = now

def embeddings_to_vocab(output):
    last_tok = output[:, -1, :]             # [1, 768]
    wte = model.wte.weight                  # [50257, 768]
    wteT = wte.T                            # [768, 50257]
    logits = last_tok @ wteT                # [1, 50257]
    profiler_record(12, "wte")
    return torch.softmax(logits, dim=-1)    # [1, 50257]

def infer_max(inp, num, m):
    for _ in range(num):
        output = m(inp)
        probs = embeddings_to_vocab(output)
        next_tok = torch.argmax(probs, dim=-1).item()
        profiler_record(13, "max")
        inp.append(next_tok)
    return inp

def model_default(inp):
    encoded_input = {
            'input_ids': torch.tensor([inp]),
            'attention_mask': torch.ones((1, len(inp)), dtype=int),
    }
    return model(**encoded_input).last_hidden_state

def model_expanded(inp):
    tok = torch.tensor(inp)
    pos = torch.arange(len(inp))

    t = model.wte(tok)
    p = model.wpe(pos)
    hidden_state = t + p
    hidden_state = hidden_state.unsqueeze(0)

    for i in range(12):
        h = model.h[i]
        ln1 = h.ln_1(hidden_state)
        attn, _ = h.attn(ln1)
        attn = attn + hidden_state
        ln2 = h.ln_2(attn)
        hidden_state = h.mlp(ln2)
        hidden_state = hidden_state + attn

    return model.ln_f(hidden_state)

def model_fully_expanded(inp):
    tok = torch.tensor(inp)
    pos = torch.arange(len(inp))

    E = model.wte.embedding_dim
    H = model.h[0].attn.num_heads
    L = len(model.h)

    t = model.wte.weight[tok]
    p = model.wpe.weight[pos]
    hidden_state = t + p
    hidden_state = hidden_state.unsqueeze(0)

    profiler_record(0, "pick")

    def layer_norm(inp, gamma, beta):
        mean = inp.mean(dim=-1, keepdims=True)
        var = ((inp - mean) ** 2).mean(dim=-1, keepdims=True)
        lh = (inp - mean) / (var + 1e-5).sqrt()
        return lh * gamma + beta

    for i in range(L):
        h = model.h[i]
        ln1 = layer_norm(hidden_state, h.ln_1.weight, h.ln_1.bias)
        profiler_record(1, "ln1")
        
        def split_head(t):
            new_shape = t.size()[:-1] + (H, 64) # 12 * 64 = 768
            t2 = t.view(*new_shape)
            return t2.permute(0, 2, 1, 3) # (batch, head, T, HLEN)

        def merge_heads(t):
            t = t.permute(0, 2, 1, 3).contiguous()
            new_shape = t.size()[:-2] + (H * 64,)
            return t.view(new_shape)

        a = h.attn

        # attn {{{
        lnx = ln1 @ a.c_attn.weight + a.c_attn.bias
        profiler_record(2, "c_attn")

        #  z=T y=12 x=64  -> z=12 y=T x=64
        #  essentially groups heads by token by flipping the cube
        #
        #      +---------+
        #     /         /|
        #    T tokens  / |
        #   /         /  |
        #  +---------+   |
        #  |         |   |
        #  |         |   +
        # 12 heads   |  /
        #  |         | /
        #  |         |/
        #  +----64---+
        # 
        #
        # FROM:
        # | t0                 | t1 .. tn | <- outer dimension
        # | h0      | h1 .. hn |            <- middle dimension
        # | 0 .. 64 |                       <- inner dimension
        #
        # TO:
        # | h0                 | h1 .. hn | <- outer dimension
        # | t0      | t1 .. tn |            <- middle dimension
        # | 0 .. 64 |                       <- inner dimension

        q, k, v = lnx.split(E, dim=-1)
        q = split_head(q) # softmax needs to happen over heads!
        k = split_head(k) # softmax needs to happen over heads!
        v = split_head(v) # softmax needs to happen over heads!

        ql, kl = q.size(-2), k.size(-2)
        attn = q @ k.transpose(-1, -2)
        attn = attn / (float(v.size(-1)) ** 0.5)

        mask = torch.tril(torch.ones(ql, kl)) # when allocating, max dimension block_size x block_size
        attn = attn.masked_fill(mask == 0, float(-1.0000e+04))
        
        attn = F.softmax(attn, dim=-1)
        attn = attn @ v
        attn = merge_heads(attn)
        profiler_record(3, "xformer")
        attn = attn @ a.c_proj.weight + a.c_proj.bias
        profiler_record(4, "c_proj")
        attn = attn + hidden_state # residual
        profiler_record(5, "c_proj residual")
        # }}}

        ln2 = layer_norm(attn, h.ln_2.weight, h.ln_2.bias)
        profiler_record(6, "ln2")
        
        # mlp {{{
        def gelua(inp):
            return 0.5 * inp * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (inp + 0.044715 * torch.pow(inp, 3.0))))

        mlp = h.mlp
        a = ln2 @ mlp.c_fc.weight + mlp.c_fc.bias
        profiler_record(7, "c_fc")
        act = gelua(a)
        profiler_record(8, "geula")
        hidden_state = act @ mlp.c_proj.weight + mlp.c_proj.bias
        profiler_record(9, "c_proj")
        hidden_state = hidden_state + attn # residual
        profiler_record(10, "c_proj residual")
        # }}}

    ret = layer_norm(hidden_state, model.ln_f.weight, model.ln_f.bias)
    profiler_record(11, "ln2 residual")
    return ret

if len(model.h) == 12:
    # encoded_input = tokenizer("In the morning I was able to", return_tensors='pt')
    inp = [818,  262, 3329,  314,  373, 1498,  284]
    exp = [818, 262, 3329, 314, 373, 1498, 284, 651, 257, 922, 804, 379, 262, 2615, 290, 262, 2615]

    print('exp', exp)
    for m in [model_default, model_expanded, model_fully_expanded]:
        if m == model_fully_expanded:
            profiler_start()
        begin = time.time()
        got = infer_max(inp.copy(), 10, m)
        end = time.time()
        print(f'{m} duration {end - begin}s ')
        if m == model_fully_expanded:
            profiler_report()
        assert got == exp

    print('inp:', tokenizer.decode(inp))
    print('exp:', tokenizer.decode(exp))

inp = [818,  262, 3329,  314,  373, 1498,  284]
print(tokenizer.decode(inp), end='')
i = len(inp)
begin = time.time()
while True:
    output = model_default(inp)
    probs = embeddings_to_vocab(output)
    next_tok = torch.multinomial(probs, num_samples=1).item()
    inp.append(next_tok)
    inp = inp[-1024:]
    print(tokenizer.decode(next_tok), end='')
    sys.stdout.flush()

    if i % 100 == 0:
        end = time.time()
        print(f'[{i} tokens, {(end-begin)/100} sec/tok]')
        begin = end
    i += 1
