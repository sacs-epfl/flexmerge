import os
import torch
from torchvision.datasets import ImageFolder

beans_classnames = [
    'angular_leaf_spot',
    'bean_rust',
    'healthy'
]

class BeansBase:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        split="train",
        batch_size=32,
        num_workers=8,
    ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = beans_classnames
        self.split = split

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, "beans", "train")
        self.train_dataset = ImageFolder(traindir, transform=self.preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def populate_test(self):
        testdir = os.path.join(self.location, "beans", self.split)
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
        return "beans"

class Beans(BeansBase):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=8,
    ):
        super().__init__(
            preprocess, 
            location, 
            split="test", 
            batch_size=batch_size, 
            num_workers=num_workers
        )

class BeansVal(BeansBase):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=8,
    ):
        super().__init__(
            preprocess, 
            location, 
            split="validation", 
            batch_size=batch_size, 
            num_workers=num_workers
        )

