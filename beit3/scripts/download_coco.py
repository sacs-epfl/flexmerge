import os
import shutil
import sys
import json
import wget
import math
import zipfile

from remotezip import RemoteZip
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Check for correct usage and input arguments
if len(sys.argv) != 3:
    print("Usage: python download_coco_captioning.py <output_dir> <sentencepiece_model_path>")
    sys.exit(1)

# Define output directory and sentencepiece model path
output_dir = os.path.abspath(sys.argv[1])
sentencepiece_model_path = os.path.abspath(sys.argv[2])

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


for folder_to_download in ["train2014", "val2014", "test2015"]:
    folder_output_path = os.path.join(output_dir, folder_to_download)
    if not os.path.exists(folder_output_path):
        folder_zip_path = os.path.join(output_dir, f"{folder_to_download}.zip")
        folder_download_url = f"http://images.cocodataset.org/zips/{folder_to_download}.zip"
        print(f"Downloading {folder_to_download} from {folder_download_url}...")
        wget.download(folder_download_url, out=folder_zip_path)
        print(f"\nUnzipping {folder_to_download}...")
        with zipfile.ZipFile(folder_zip_path, 'r') as z:
            z.extractall(output_dir)
        os.remove(folder_zip_path)
    else:
        print(f"{folder_to_download} already exists, skipping download.")


# Download and process COCO captioning dataset
file_to_extract = "dataset_coco.json"
if not os.path.exists(os.path.join(output_dir, file_to_extract)):
    # Download and extract the dataset
    with RemoteZip("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip") as zf:
        with zf.open(file_to_extract) as f:
            data_split = json.load(f)
    # Filter for test split
    data_split["images"] = list(filter(lambda x: x["split"] == "test", data_split["images"]))
    print("Number of images in the selected coco captioning splits:", len(data_split["images"]))
    # Save the filtered dataset
    with open(os.path.join(output_dir, file_to_extract), "w") as f:
        json.dump(data_split, f)
else:
    print(f"{file_to_extract} already exists, skipping download.")


# Download and extract VQA dataset
vqa_dir = os.path.join(output_dir, "vqa")
if not os.path.exists(vqa_dir):
    os.makedirs(vqa_dir, exist_ok=True)
    print("Downloading VQA dataset questions and annotations...")
    with RemoteZip("https://www.kaggle.com/api/v1/datasets/download/biminhco/vqa-v2-question") as zf:
        zf.extractall(vqa_dir)
else:
    print("vqa directory already exists, skipping download.")

# Import custom dataset classes and tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from beit3.datasets import CaptioningDataset, RetrievalDataset, VQAv2Dataset
from transformers import XLMRobertaTokenizer

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer(sentencepiece_model_path)

# Generate COCO captioning dataset index
if not os.path.exists(os.path.join(output_dir, "coco_captioning.test.jsonl")):
    CaptioningDataset.make_coco_captioning_dataset_index(
        data_path=output_dir,
        tokenizer=tokenizer,
    )
else:
    print("coco_captioning.test.jsonl already exists, skipping download.")

# Generate COCO retrieval dataset index
if not os.path.exists(os.path.join(output_dir, "coco_retrieval.test.jsonl")):
    RetrievalDataset.make_coco_dataset_index(
        data_path=output_dir,
        tokenizer=tokenizer,
    )
else:
    print("coco_retrieval.test.jsonl already exists, skipping download.")

# Generate VQA dataset index
if not os.path.exists(os.path.join(output_dir, "vqa.test.jsonl")):
    VQAv2Dataset.make_dataset_index(
        data_path=output_dir,
        tokenizer=tokenizer,
        annotation_data_path=vqa_dir,
    )
else:
    print("vqa.test.jsonl already exists, skipping download.")

coco_capt_test_gt = os.path.join(output_dir, "coco_karpathy_test_gt.json")
if not os.path.exists(coco_capt_test_gt):
    wget.download(
        "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
        out=coco_capt_test_gt,
    )
else:
    print("coco_karpathy_test_gt.json already exists, skipping download.")