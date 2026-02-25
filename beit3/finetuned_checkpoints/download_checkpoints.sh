#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Download beit3.spm separately
echo "Downloading beit3.spm..."
wget -O "beit3.spm" "https://github.com/addf400/files/releases/download/beit3/beit3.spm" &

echo "Downloading pretrained beit3-base model"
wget -O "beit3_base_patch16_224.pth" "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth" &

# Read the CSV file and download files
csv_file="checkpoints_info.csv"
if [[ -f "$csv_file" ]]; then
    skip_headers=1
    while IFS=, read -r filename url task_name model_name || [[ -n "$filename" ]];
    do
        if ((skip_headers))
        then
            ((skip_headers--))
        else
            echo "Downloading $url into $filename..."
            wget -O "$filename" "$url" &
        fi
    done < "$csv_file"
else
    echo "Error: $csv_file not found!"
    exit 1
fi

# Wait for all background jobs to finish
wait
