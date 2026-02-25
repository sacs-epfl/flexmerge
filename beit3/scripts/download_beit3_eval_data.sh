#!/bin/bash

# This requires wget and remotezip python packages to be installed: pip install wget remotezip

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi
OUTPUT_PATH="$1"

echo "Output path: $OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH" || (echo "Error: Failed to create output directory $OUTPUT_PATH" && exit 1)

SENTENCEPIECE_MODEL_PATH="$SCRIPT_DIR/../finetuned_checkpoints/beit3.spm"


COCO_PATH="$OUTPUT_PATH/coco"
echo "Downloading COCO datasets..."
mkdir -p "$COCO_PATH"
python "$SCRIPT_DIR/download_coco.py" "$COCO_PATH" "$SENTENCEPIECE_MODEL_PATH"

download_and_unpack() {
        local url=$1 destdir=$2
        mkdir -p "$destdir"
        wget -O- "$url" \
                | bsdtar -xf- -C "$destdir"
}

IMAGENET_PATH="$OUTPUT_PATH/imagenet"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "Downloading ImageNet datasets..."
    mkdir -p "$IMAGENET_PATH"
    mkdir -p "$IMAGENET_PATH/val"

    download_and_unpack "https://www.kaggle.com/api/v1/datasets/download/sautkin/imagenet1kvalid" "$IMAGENET_PATH/val"

    python "$SCRIPT_DIR/make_imagenet_index.py" "$IMAGENET_PATH"
else
    echo "ImageNet datasets already exist at $IMAGENET_PATH. Skipping download."
fi

NLVR2_PATH="$OUTPUT_PATH/nlvr2"
if [ ! -d "$NLVR2_PATH" ]; then
    echo "Downloading NLVR2 datasets..."
    mkdir -p "$NLVR2_PATH"

    download_and_unpack "https://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip" "$NLVR2_PATH"
    git clone https://github.com/lil-lab/nlvr.git "$NLVR2_PATH/nlvr"

    python "$SCRIPT_DIR/make_nlvr2_index.py" "$NLVR2_PATH" "$SENTENCEPIECE_MODEL_PATH"
else
    echo "NLVR2 datasets already exist at $NLVR2_PATH. Skipping download."
fi