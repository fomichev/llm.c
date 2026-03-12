#!/bin/bash
set -e

M=${1:-124M}
VENV_DIR=".venv"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Setting up virtual environment for $M..."
pip install --quiet torch transformers safetensors

# Install llama.cpp's gguf-py if needed
if ! python3 -c "import gguf" 2>/dev/null; then
    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        echo "Cloning llama.cpp..."
        git clone --depth=1 --filter=blob:none --sparse "$LLAMA_CPP_DIR" 2>/dev/null || \
            git clone --depth=1 --filter=blob:none --sparse https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
        (cd "$LLAMA_CPP_DIR" && git sparse-checkout set --skip-checks convert_hf_to_gguf.py gguf-py)
    fi
    pip install --quiet "$LLAMA_CPP_DIR/gguf-py"
fi

MODEL_NAME="openai-community/gpt2"
case "$M" in
    124M)  MODEL_NAME="openai-community/gpt2" ;;
    355M)  MODEL_NAME="openai-community/gpt2-medium" ;;
    774M)  MODEL_NAME="openai-community/gpt2-large" ;;
    1558M) MODEL_NAME="openai-community/gpt2-xl" ;;
    *) echo "Unknown model size: $M (expected 124M, 355M, 774M, 1558M)" >&2; exit 1 ;;
esac

# Short name used in HF repo filenames (strip "openai-community/" prefix)
HF_SHORT="${MODEL_NAME#openai-community/}"

CONVERTER="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [ ! -f "$CONVERTER" ]; then
    echo "Error: $CONVERTER not found" >&2
    echo "Set LLAMA_CPP_DIR to your llama.cpp checkout" >&2
    exit 1
fi

# GPT2LMHeadModel.save_pretrained strips the attn.bias/masked_bias
# buffers that the converter doesn't handle, so we save locally first.
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Downloading $MODEL_NAME..."
python3 -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
m = GPT2LMHeadModel.from_pretrained('$MODEL_NAME')
t = GPT2Tokenizer.from_pretrained('$MODEL_NAME')
m.save_pretrained('$TMPDIR', safe_serialization=True)
t.save_pretrained('$TMPDIR')
"

echo "Converting to GGUF f32..."
python3 "$CONVERTER" "$TMPDIR" --outtype f32 --outfile "gpt2_${M}.gguf"
echo "Done: gpt2_${M}.gguf"

# Download pre-quantized variants from Hugging Face
echo "Downloading Q8_0..."
curl -fSL "https://huggingface.co/DevQuasar/openai-community.${HF_SHORT}-GGUF/resolve/main/openai-community.${HF_SHORT}.Q8_0.gguf" \
    -o "gpt2_${M}-Q8_0.gguf"
echo "Done: gpt2_${M}-Q8_0.gguf"

echo "Downloading Q4_0..."
curl -fSL "https://huggingface.co/RichardErkhov/openai-community_-_${HF_SHORT}-gguf/resolve/main/${HF_SHORT}.Q4_0.gguf" \
    -o "gpt2_${M}-Q4_0.gguf"
echo "Done: gpt2_${M}-Q4_0.gguf"
