import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm

class CNN(Dataset):
    def __init__(self):
        fileDir = "normalizedSize/normalizedSize"

        # Transforms images in grayscale and to a tensor to be able to load into dataloader
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Convert PIL image to Tensor
            # Add other transforms here, like normalization
        ])
        df = datasets.ImageFolder(fileDir, transform=transform)
        loader = DataLoader(df, batch_size=len(df), shuffle=False)
        images, labels = next(iter(loader))
        self.train_numbers = images.view(-1,1,128,128)
        print(self.train_numbers.size())
        self.train_labels = labels
        print(self.train_labels.size())
        #exit(10)
        # self.test_numbers = torch.tensor(df[3500:][0]).view(-1, 1, 128, 128)
        # self.test_labels = torch.tensor(df[3500:][1])

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_numbers[item], self.train_labels[item]

    def __len__(self):
        return self.len


class NumberClassify(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(NumberClassify, self).__init__()

        self.in_to_out = nn.Linear(128*128, 4) #

    def forward(self, x):

        x = torch.flatten(x,1)
        return self.in_to_out(x)


def trainNN(epochs=10, batch_size=16, lr=0.001, display_test_acc=False):
    # load dataset
    cnn = CNN()

    # create data loader
    cnn_loader = DataLoader(cnn, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    number_classify = NumberClassify()#.to(device)
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

            input = number_classify(x)

            loss = cross_entropy(input, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        # if display_test_acc:
        with torch.no_grad():
            predictions = torch.argmax(number_classify(cnn.train_numbers), dim=1)  # Get the prediction
            correct = (predictions == cnn.train_labels).sum().item()
            print(f"Accuracy on train set: {correct / len(cnn.train_labels):.4f}")

trainNN()