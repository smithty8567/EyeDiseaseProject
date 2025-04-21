import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm

class CNN(Dataset):
    def __init__(self):
        fileDir = "normalizedSize"

        # Transforms images in grayscale and to a tensor to be able to load into dataloader
        transform = v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.RandomHorizontalFlip(p=0.25),
            v2.RandomVerticalFlip(p=0.25),
            v2.RandomRotation(degrees=360),
            #v2.RandomInvert(p=0.2),
            v2.ToTensor(),  # Convert PIL image to Tensor
            v2.Normalize(mean=[0], std=[1])
            # Add other transforms here, like normalization
        ])
        df = datasets.ImageFolder(fileDir, transform=transform)

        loader = DataLoader(df, batch_size=len(df), shuffle=False)
        images, labels = next(iter(loader))

        self.train_images = images.view(-1,1,128,128)
        #print(self.train_numbers.size())
        self.train_labels = labels

        # # Standard deviation of all columns
        # std_all = self.train_images.std()
        # print("Standard deviation of all columns:\n", std_all)
        #print(self.train_labels.size())
        #exit(10)

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_images[item], self.train_labels[item]

    def __len__(self):
        return self.len


class EyeDisease(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(EyeDisease, self).__init__()
        self.in_to_h1 = nn.Conv2d(1, 16, (5, 5), padding=(2, 2))  # 16 x 128 x 128
        # Maxpool2d -> 16 x 64 x 64
        self.h1_to_h2 = nn.Conv2d(16, 8, (3, 3), padding=(1, 1))  # 8 x 64 x 64
        # Maxpool2d -> 8 x 32 x 32
        self.h2_to_h3 = nn.Linear(8*32*32, 8) #
        self.h3_to_out = nn.Linear(8, 4)  #

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))  # 16 x 128 x 128
        x = F.max_pool2d(x, (2, 2))  # 16 x 64 x 64
        x = F.relu(self.h1_to_h2(x))  # 8 x 64 x 64
        x = F.max_pool2d(x, (2, 2))  # 8 x 32 x 32
        x = torch.flatten(x, 1)
        x = F.relu(self.h2_to_h3(x))  # 8
        return (self.h3_to_out(x)) # 4


def trainNN(epochs=15, batch_size=16, lr=0.002, display_test_acc=False):
    # load dataset
    cnn = CNN()

    # create data loader
    cnn_loader = DataLoader(cnn, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    number_classify = EyeDisease()#.to(device)
    print(f"Total parameters: {sum(param.numel() for param in number_classify.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(number_classify.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(cnn_loader)):

            x, y = data

            # x = x.to(device)
            # y = y.to(device)

            optimizer.zero_grad()

            output = number_classify(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        # if display_test_acc:
        with torch.no_grad():
            predictions = torch.argmax(number_classify(cnn.train_images), dim=1)  # Get the prediction
            correct = (predictions == cnn.train_labels).sum().item()
            print(f"Accuracy on train set: {correct / len(cnn.train_labels):.4f}")

trainNN()