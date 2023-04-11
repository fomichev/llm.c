# Large language model(s) in C

GPT-2 inference implementation in pure C. The only external dependency
is CBLAS (Intel MKL is what I've been using).

The project is mostly (self) educational because original OpenAI TensorFlow
implementation is not super comprehensible.

## Implementation details

Transformer implementation is not super optimal:

* Instead of head-qkv permutation (in python case), I'm doing a copy to the per-head buffer.
* No threading (besides Intel MKL for matmul), all transformer heads run in sequence.
* KV cache is not rotated, the program crashes upon reaching 1024 tokens.

OTOH, KV-cache is implemented which seems to bring performance close to
the PyTorch Hugging Face implementation.

The model parameters are converted into mmap-able format so even the models
that don't fit into RAM should be able to run (albeit super slowly).

All GPT-2 flavors (124M/355M/774M/1558M) are supported and run perfectly fine.

## HOWTO

Build container with all dependencies and download GPT-2 124M model:

```
$ git clone git@github.com:fomichev/llm.c.git
$ cd llm.c
$ export BUILDAH_LAYERS=true
$ buildah bud --build-arg M=124M --volume ${PWD}:/host:rw --tag llmc
```

Start container and it will tell a story (500 tokens) starting with
the following phrase: "In the morning I was able to". top-k-5 sampling
is used to make sure each time the story is a little bit different.

```
$ podman run --volume ${PWD}:/host:rw localhost/llmc:latest ./llmc gpt2_124M.llmc In the morning I was able to
```
```
loading model from gpt2_124M.llmc
runtime memory: 10MB + 18MB KV cache
In the morning I was able to get to the airport and was taken back to work. My flight was taken
back by a taxi, so it took about 30 minutes for my luggage. It is now over the border and my wife
and son are in Canada and are going there on Monday, June 1. I was told to take my passport out for
the flight back. The flight has already departed and it has been taken by another plane which will
take my family. The next day I went to see if[100 tokens, 0.038295785 sec/tok] they were willing
to give the passport out, they told the flight manager to give it out. I had a lot of anxiety because
I was going to be going to Canada, I had been told that it was going out and they were going out
of business. I had to take it home, so that was very difficult. My husband and I are in Vancouver
and are going to see the Canadian consulate in Canada and we will get our passport out soon and then
I can see the rest. The flight[200 tokens, 0.041043856 sec/tok] is now over and the flight is now in
the airport for me to go home. I was also given a passport that says I am going to go on holiday.
My passport says that the next flight will arrive from Montreal on Monday and that is when we will be in
Canada, we just need to make arrangements. We are not in the US yet and it will be about three days before
the trip. It is going well, it is going really well for us. I will be in Canada[300 tokens, 0.044696700 sec/tok]
on June 11 to meet our son. He has been a little busy so it has been a long time but we will get him
back. He is going to go to a couple of different places so we need his passport and his name and we need
a place to get a picture so I can see the whole thing, we can go to a hotel in Canada. He has been in
this country for about three weeks. It is very nice to have the chance to meet him and get some good
pictures[400 tokens, 0.046139368 sec/tok] of what it feels. We will have our picture taken, I have been
told that he is a little shy about this and I have told him to come and we are very happy that he did,
he is going well. It has been an incredible journey for me as he had been so kind and supportive, it
has been really amazing for me. He was very supportive of our journey and I really want to thank him and
our friends at our local hotel, the hotel in Montreal where I am from[500 tokens, 0.047711053 sec/tok].
It is so great for him
total=22.272080s
```

Running on 4-core 8th gen Intel i5:

```
$ grep "model name" /proc/cpuinfo | head -n1
model name      : Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
```

## Other GPT-2 sizes

Possible bigger GPT2-2 sizes (M= argument):

* 124M, 523M on disk, 50ms per token
* 355M, 1.5G on disk, 100ms per token
* 774M, 3.1G on disk, 200ms per token
* 1558M, 6.0G on disk, 500ms per token

Note that about 3x the amount of disk space is required. The script
downloads OpenAI TensorFlow snapshot, converts it to PyTorch and then
coverts it to the mmap-able format that llm.c can understand.
