LIBS=-lm

#BLAS_CFLAGS=$(shell pkg-config --cflags cblas)
#BLAS_LDFLAGS=$(shell pkg-config --libs cblas)

# OpenBLAS
BLAS_CFLAGS=$(shell pkg-config --cflags openblas)
BLAS_LDFLAGS=$(shell pkg-config --libs openblas)

# Intel MKL
#BLAS_CFLAGS=$(shell pkg-config --cflags mkl-dynamic-lp64-iomp)
#BLAS_LDFLAGS=$(shell pkg-config --libs mkl-dynamic-lp64-iomp)

# ROCm
#BLAS_CFLAGS=-I/opt/rocm/include
#BLAS_LDFLAGS=-L/opt/rocm/lib -lrocblas -Dcblas_sgemm=rocblas_sgemm

SLEEF_CFLAGS=-I$(HOME)/src/sleef/install/include -D USE_SLEEF
SLEEF_LDFLAGS=-L$(HOME)/src/sleef/install/lib
LIBS+=-lsleef

#export SRAND48_SEED=1337

O=3
GPT2_EVAL_ROUNDS?=10

CFLAGS=$(SLEEF_CFLAGS) $(BLAS_CFLAGS) -I. -O$(O) -march=native -DGPT2_EVAL_ROUNDS=$(GPT2_EVAL_ROUNDS) -rdynamic
LDFLAGS=$(SLEEF_LDFLAGS) $(BLAS_LDFLAGS)
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
	$(CC) $(LDFLAGS) $(CFLAGS) -g main.c gpt2.c snapshot.c vocab.c tensor.c $(LIBS) -o llmc
	$(CC) $(LDFLAGS) $(CFLAGS) -g main.c gpt2.c snapshot.c vocab.c tensor.c $(LIBS) -o llmc

check:
	$(MAKE) build
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/tensor.c tensor.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/simd.c $(LIBS) && ./a.out
	./llmc gpt2_$(M).llmc

flamegraph:
	$(MAKE) build O=0 GPT2_EVAL_ROUNDS=100
	perf record -F 99 -g -- ./llmc gpt2_$(M).llmc In the morning I was able to
	perf script | $(FG)/stackcollapse-perf.pl > out.perf-folded
	$(FG)/flamegraph.pl out.perf-folded > perf.svg

convert:
	./gpt2_convert.py ~/src/gpt-2/pytorch_$(M) ~/src/gpt-2/models/$(M)/encoder.json gpt2_$(M)

eval:
	./gpt2_eval.py ~/src/gpt-2/pytorch_$(M)
