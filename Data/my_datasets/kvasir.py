import os
import torch
from torchvision import datasets

class Kvasir:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):
        # Load the entire dataset
        full_dataset = datasets.ImageFolder(root=os.path.join(location, "kvasir"), transform=preprocess)

        # Compute split sizes
        train_split = 0.9
        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Create generator for reproducibility
        generator = torch.Generator().manual_seed(42)

        # Split into train and test
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        # Create DataLoaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Classes come from ImageFolder dataset
        self.classnames = full_dataset.classes
