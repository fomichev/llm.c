#BLAS_CFLAGS=$(shell pkg-config --cflags cblas)
#BLAS_LDFLAGS=$(shell pkg-config --libs cblas) -lm

#export SRAND48_SEED=1337

BLAS_CFLAGS=$(shell pkg-config --cflags mkl-dynamic-lp64-iomp)
BLAS_LDFLAGS=$(shell pkg-config --libs mkl-dynamic-lp64-iomp)

O=3
GPT2_EVAL_ROUNDS?=10

CFLAGS=$(BLAS_CFLAGS) -I. -O$(O) -march=native -DGPT2_EVAL_ROUNDS=$(GPT2_EVAL_ROUNDS) -rdynamic
LDFLAGS=$(BLAS_LDFLAGS)
CC=clang

M=124M
#M=355M
#M=774M
#M=1558M

FG=/home/sdf/src/FlameGraph

all:
	$(MAKE) build
	./llmc gpt2_$(M).llmc In the morning I was able to

build:
	$(CC) $(LDFLAGS) $(CFLAGS) -g main.c gpt2.c snapshot.c vocab.c tensor.c -o llmc

check:
	$(MAKE) build
	./llmc gpt2_$(M).llmc
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/tensor.c tensor.c && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/simd.c && ./a.out

flamegraph:
	$(MAKE) build O=0 GPT2_EVAL_ROUNDS=100
	perf record -F 99 -g -- ./llmc gpt2_$(M).llmc In the morning I was able to
	perf script | $(FG)/stackcollapse-perf.pl > out.perf-folded
	$(FG)/flamegraph.pl out.perf-folded > perf.svg

convert:
	./gpt2_convert.py ~/src/gpt-2/pytorch_$(M) ~/src/gpt-2/models/$(M)/encoder.json gpt2_$(M)

eval:
	./gpt2_eval.py ~/src/gpt-2/pytorch_$(M)
