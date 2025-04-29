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
        fileDir = "normalizedSizeColor256"
        # fileDir = "normalizedSizeColor256Binary"

        # Transforms images in grayscale and to a tensor to be able to load into dataloader
        transform = v2.Compose([
            v2.ToImage(),  # Convert PIL image to Tensor
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[.4209,.2807,.1728], std=[.2931,.2173,.1644])


        ])
        df = datasets.ImageFolder(fileDir, transform=transform)

        loader = DataLoader(df, batch_size=len(df), shuffle=True)
        images, labels = next(iter(loader))

        # Labels for 4 type classification
        self.unique_labels = ["cataract","diabetic_retinopathy","glaucoma","normal"]

        # # Labels for Binary classification
        # self.unique_labels = ["glaucoma", "normal"]

        self.train_images = images.view(-1,3,256,256)
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
        self.in_to_h1 = nn.Conv2d(3, 32, (5, 5), padding=(2, 2))  # 32 x 256 x 256
        # Maxpool2d -> 32 x 128 x 128
        self.h1_to_h2 = nn.Conv2d(32, 16, (3, 3), padding=(1, 1))  # 16 x 128 x 128
        # Maxpool2d -> 16 x 64 x 64
        self.h3_to_h4 = nn.Linear(16*64*64, 8) # 8
        self.h4_to_out = nn.Linear(8, 4)  # 4 Layer for 4 classes
        # self.h4_to_out = nn.Linear(8, 2)  # 2 Layer for Binary Classification

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))  # 32 x 256 x 256
        x = F.dropout(x, p=.1)
        x = F.max_pool2d(x, (2, 2))  # 32 x 128 x 128
        x = F.relu(self.h1_to_h2(x))  # 16 x 128 x 128
        x = F.max_pool2d(x, (2, 2))  # 16 x 64 x 64
        x = torch.flatten(x, 1)
        x = F.relu(self.h3_to_h4(x))  # 8
        return self.h4_to_out(x) # 4


def trainNN(epochs=10, batch_size=32, lr=0.0005, display_test_acc=True):
    # load dataset
    cnn = CNN()

    # create data loader
    cnn_loader = DataLoader(cnn, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    disease_classify = EyeDisease().to(device)
    print(f"Total parameters: {sum(param.numel() for param in disease_classify.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(disease_classify.parameters(), lr=lr, weight_decay=1e-5)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(cnn_loader)):
            disease_classify.train()

            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            output = disease_classify(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")

        running_loss = 0.0
        if epoch == epochs-1:
            sums = 0
            for i in range(len(cnn.train_images)):
                image = cnn.train_images[i].to(device)
                result = torch.argmax(disease_classify(image.view(-1, 3, 256, 256)).to(device),dim=1)
                if result == cnn.train_labels[i]:
                    sums += 1/len(cnn.train_images)
            print(f"Accuracy on train set: {sums:.4f}")
        if display_test_acc:
            with torch.no_grad():
                disease_classify.eval()
                sums = 0
                all_results = []
                for i in range(len(cnn.valid_images)):
                    image = cnn.valid_images[i].to(device)
                    result = torch.argmax(disease_classify(image.view(-1, 3, 256, 256)).to(device),dim=1)
                    all_results.extend(result.cpu()) #adding the calculated predictions(result) from the batch to the array
                    if result == cnn.valid_labels[i]:
                        sums += 1 / len(cnn.valid_images)
                print(f"Accuracy on validation set: {sums:.4f}")
    cm = confusion_matrix(cnn.valid_labels, all_results)
    disp = ConfusionMatrixDisplay(cm, display_labels=cnn.unique_labels)
    disp.plot()
    plt.show()
    return disease_classify

CNN = trainNN(epochs = 25, batch_size=32)
SaveLoad.save(CNN, path = "4typeClassification.pth")
# CNN = SaveLoad.load()
# CNN.eval()
# print(CNN)
