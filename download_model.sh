#!/bin/bash
set -e

M=${1:-124M}
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Setting up virtual environment for $M..."
pip install --quiet torch transformers

# gpt2_convert.py downloads the model from HuggingFace via
# GPT2Model.from_pretrained and converts it to llm.c format.
#
# It expects: gpt2_convert.py <pretrained_name> <encoder.json> <output_name>
# but encoder.json is only used for vocab, and the tokenizer is loaded
# separately from HuggingFace. We download encoder.json from OpenAI.

MODEL_NAME="gpt2"
case "$M" in
    124M) MODEL_NAME="gpt2" ;;
    355M) MODEL_NAME="gpt2-medium" ;;
    774M) MODEL_NAME="gpt2-large" ;;
    1558M) MODEL_NAME="gpt2-xl" ;;
    *) echo "Unknown model size: $M (expected 124M, 355M, 774M, 1558M)" >&2; exit 1 ;;
esac

ENCODER_URL="https://openaipublic.blob.core.windows.net/gpt-2/models/${M}/encoder.json"
ENCODER_FILE="encoder_${M}.json"

if [ ! -f "$ENCODER_FILE" ]; then
    echo "Downloading encoder.json for $M..."
    curl -sL "$ENCODER_URL" -o "$ENCODER_FILE"
fi

echo "Converting $MODEL_NAME to llm.c format..."
./gpt2_convert.py "$MODEL_NAME" "$ENCODER_FILE" "gpt2_${M}"

echo "Done. Model files written to gpt2_${M}.llmc"
