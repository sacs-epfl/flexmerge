import os
import torch
from torchvision.datasets import ImageFolder

vegetables_classnames = [
    'Bean',
    'Bitter_Gourd',
    'Bottle_Gourd',
    'Brinjal',
    'Broccoli',
    'Cabbage',
    'Capsicum',
    'Carrot',
    'Cauliflower',
    'Cucumber',
    'Papaya',
    'Potato',
    'Pumpkin',
    'Radish',
    'Tomato'
]

class VegetablesBase:
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
        self.classnames = vegetables_classnames
        self.split = split

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, "vegetables", "train")
        self.train_dataset = ImageFolder(traindir, transform=self.preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def populate_test(self):
        testdir = os.path.join(self.location, "vegetables", self.split)
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
        return "vegetables"

class Vegetables(VegetablesBase):
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

class VegetablesVal(VegetablesBase):
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

