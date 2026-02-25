import os
import sys

if len(sys.argv) != 2:
    print("Usage: python make_imagenet_index.py <data_path>")
    sys.exit(1)

data_path = os.path.abspath(sys.argv[1])

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from beit3.datasets import ImageNetDataset

ImageNetDataset.make_dataset_index(
    train_data_path = None,
    val_data_path = os.path.join(data_path, "val"),
    index_path = data_path
)