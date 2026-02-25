import os
from torch.utils.data import Dataset
import torch
from torchvision.datasets.folder import default_loader

### follows the same class ordering as the .txt files
garbage_classnames = [
    'glass', 
    'paper', 
    'cardboard', 
    'plastic', 
    'metal', 
    'trash'
]

class CustomDatasetReader(Dataset):
    def __init__(self, images_dir, split_file, transform=None):
        """
        Args:
            images_dir (str): Path to the 'images' folder.
            split_file (str): Path to the txt file containing <image_name> <label>.
            transform (callable, optional): Transform to apply to images.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.default_loader = default_loader

        # Step 1: Build a lookup: image_name -> full_path
        self.image_lookup = self._build_image_lookup()

        # Step 2: Read split file
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, label = parts
                    if image_name not in self.image_lookup:
                        raise FileNotFoundError(f"{image_name} not found in {images_dir}")
                    image_path = self.image_lookup[image_name]
                    self.samples.append((image_path, int(label)-1))

    def _build_image_lookup(self):
        """Build a lookup dict from image name to full path."""
        lookup = {}
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg')):  # Accept common formats
                    lookup[file] = os.path.join(root, file)
        return lookup

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = self.default_loader(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

class Garbage:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=8,
    ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = garbage_classnames

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, "garbage_classification", "images")
        split_file = os.path.join(self.location, "garbage_classification", "one-indexed-files-notrash_train.txt")
        self.train_dataset = CustomDatasetReader(traindir, split_file, transform=self.preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def populate_test(self):
        testdir = os.path.join(self.location, "garbage_classification", "images")
        split_file = os.path.join(self.location, "garbage_classification", "one-indexed-files-notrash_test.txt")
        self.test_dataset = CustomDatasetReader(testdir, split_file, transform=self.preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def name(self):
        return "intel_images"

class GarbageVal(Garbage):
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=8):
        super().__init__(preprocess, location, batch_size, num_workers)

    def populate_test(self):
        testdir = os.path.join(self.location, "garbage_classification", "images")
        split_file = os.path.join(self.location, "garbage_classification", "one-indexed-files-notrash_val.txt")
        self.test_dataset = CustomDatasetReader(testdir, split_file, transform=self.preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )