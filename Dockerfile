FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip \
		   libmkl-dev \
		   clang

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir jupyter \
                                          tensorflow-cpu \
                                          torch

RUN python3 -m pip install 'transformers[torch]'
RUN python3 -m pip install 'transformers[tf-cpu]'

RUN git clone https://github.com/openai/gpt-2.git
WORKDIR /gpt-2

ARG M

RUN python3 download_model.py $M
RUN mkdir pytorch_$M
RUN transformers-cli convert \
	--model_type gpt2 \
	--tf_checkpoint models/$M \
	--pytorch_dump_output pytorch_$M \
	--config models/$M/hparams.json

WORKDIR /host
RUN ./gpt2_convert.py /gpt-2/pytorch_$M /gpt-2/models/$M/encoder.json gpt2_$M
RUN echo 2
RUN make build
