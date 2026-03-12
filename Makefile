LIBS=-lm

#BLAS_CFLAGS=$(shell pkg-config --cflags cblas)
#BLAS_LDFLAGS=$(shell pkg-config --libs cblas)

# OpenBLAS
BLAS_CFLAGS=-I/usr/include/openblas
BLAS_LDFLAGS=-lopenblas

# Intel MKL
#BLAS_CFLAGS=$(shell pkg-config --cflags mkl-dynamic-lp64-iomp)
#BLAS_LDFLAGS=$(shell pkg-config --libs mkl-dynamic-lp64-iomp)

# ROCm
#BLAS_CFLAGS=-I/opt/rocm/include
#BLAS_LDFLAGS=-L/opt/rocm/lib -lrocblas -Dcblas_sgemm=rocblas_sgemm

#SLEEF_CFLAGS=-I$(HOME)/src/sleef/install/include -D USE_SLEEF
#SLEEF_LDFLAGS=-L$(HOME)/src/sleef/install/lib
#LIBS+=-lsleef

O=3
MODEL?=gpt2

CFLAGS=$(SLEEF_CFLAGS) $(BLAS_CFLAGS) -I. -Imodels/$(MODEL) -O$(O) -march=native -rdynamic
LDFLAGS=$(SLEEF_LDFLAGS) $(BLAS_LDFLAGS)
CC=clang

M=124M
#M=355M
#M=774M
#M=1558M

FG=/home/sdf/src/FlameGraph

all:
	$(MAKE) build
	./llmc gpt2_$(M).gguf In the morning I was able to

build:
	$(CC) $(LDFLAGS) $(CFLAGS) -g main.c models/$(MODEL)/$(MODEL).c model.c nn.c kvcache.c gguf.c vocab.c tensor.c quant.c profiler.c $(LIBS) -o llmc

check:
	$(MAKE) build
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/tensor.c tensor.c quant.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/simd.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/nn.c tensor.c quant.c nn.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/gguf.c gguf.c tensor.c quant.c $(LIBS) && ./a.out gpt2_$(M).gguf
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/quant.c tensor.c quant.c $(LIBS) && ./a.out
	SRAND48_SEED=1337 ./llmc gpt2_$(M).gguf < models/gpt2/test/prefill_$(M).txt > models/gpt2/test/got_$(M).txt
	diff models/gpt2/test/expected_$(M).txt models/gpt2/test/got_$(M).txt
	SRAND48_SEED=1337 ./llmc gpt2_$(M)-Q8_0.gguf < models/gpt2/test/prefill_$(M).txt > models/gpt2/test/got_$(M)-Q8_0.txt
	diff models/gpt2/test/expected_$(M)-Q8_0.txt models/gpt2/test/got_$(M)-Q8_0.txt

flamegraph:
	$(MAKE) build O=0
	perf record -F 99 -g -- ./llmc gpt2_$(M).gguf In the morning I was able to
	perf script | $(FG)/stackcollapse-perf.pl > out.perf-folded
	$(FG)/flamegraph.pl out.perf-folded > perf.svg

eval:
	models/gpt2/eval.py ~/src/gpt-2/pytorch_$(M)
