import os
import torch
from torchvision.datasets import ImageFolder

intel_images_classnames = [
    'buildings', 
    'forest', 
    'glacier', 
    'mountain', 
    'sea', 
    'street'

]

class IntelImages:
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
        self.classnames = intel_images_classnames

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, "intel_images", "seg_train")
        self.train_dataset = ImageFolder(traindir, transform=self.preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def populate_test(self):
        testdir = os.path.join(self.location, "intel_images", "seg_test")
        self.test_dataset = ImageFolder(testdir, transform=self.preprocess)
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