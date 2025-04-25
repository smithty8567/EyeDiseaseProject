import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
import SaveLoad


class CNN(Dataset):
    def __init__(self):

        ####Check fileDir name to be correct
        fileDir = "normalizedSizeColor/normalizedSizeColor"

        # Transforms images in grayscale and to a tensor to be able to load into dataloader
        transform = v2.Compose([
            v2.ToImage(),  # Convert PIL image to Tensor
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[.4209,.2807,.1728], std=[.2931,.2173,.1644])


        ])
        df = datasets.ImageFolder(fileDir, transform=transform)

        loader = DataLoader(df, batch_size=len(df), shuffle=True)
        images, labels = next(iter(loader))

        self.unique_labels = ["cataract","diabetic_retinopathy","glaucoma","normal"]

        self.train_images = images.view(-1,3,128,128)
        self.train_labels = labels

        self.train_images, self.valid_images, self.train_labels, self.valid_labels = train_test_split(self.train_images, self.train_labels, test_size=0.2)


        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_images[item], self.train_labels[item]

    def __len__(self):
        return self.len


class EyeDisease(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(EyeDisease, self).__init__()
        self.in_to_h1 = nn.Conv2d(3, 16, (5, 5), padding=(2, 2))  # 16 x 128 x 128
        # Maxpool2d -> 16 x 64 x 64
        self.h1_to_h2 = nn.Conv2d(16, 8, (3, 3), padding=(1, 1))  # 8 x 64 x 64
        # Maxpool2d -> 8 x 32 x 32

        self.h3_to_h4 = nn.Linear(8*32*32, 8) # 8
        self.h4_to_out = nn.Linear(8, 4)  # 4

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))  # 16 x 128 x 128
        x = F.max_pool2d(x, (2, 2))  # 16 x 64 x 64
        x = F.relu(self.h1_to_h2(x))  # 8 x 64 x 64
        x = F.max_pool2d(x, (2, 2))  # 8 x 32 x 32
        x = torch.flatten(x, 1)
        x = F.relu(self.h3_to_h4(x))  # 8
        return self.h4_to_out(x) # 4


def trainNN(epochs=10, batch_size=32, lr=0.001, display_test_acc=True):
    # load dataset
    cnn = CNN()

    # create data loader
    cnn_loader = DataLoader(cnn, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    disease_classify = EyeDisease()#.to(device)
    print(f"Total parameters: {sum(param.numel() for param in disease_classify.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(disease_classify.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(cnn_loader)):
            disease_classify.train()

            x, y = data

            # x = x.to(device)
            # y = y.to(device)

            optimizer.zero_grad()

            output = disease_classify(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")

        running_loss = 0.0

        predictions = torch.argmax(disease_classify(cnn.train_images), dim=1)  # Get the prediction
        correct = (predictions == cnn.train_labels).sum().item()
        print(f"Accuracy on train set: {correct / len(cnn.train_labels):.4f}")
        if display_test_acc:
            with torch.no_grad():
                disease_classify.eval()
                predictions = torch.argmax(disease_classify(cnn.valid_images), dim=1)  # Get the prediction
                correct = (predictions == cnn.valid_labels).sum().item()
                print(f"Accuracy on test set: {correct / len(cnn.valid_labels):.4f}")
    cm = confusion_matrix(cnn.valid_labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=cnn.unique_labels)
    disp.plot()
    plt.show()
    return disease_classify

CNN = trainNN(epochs = 20, batch_size=32)
# CNN = SaveLoad.load()
# CNN.eval()
# print(CNN)
