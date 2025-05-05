import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import SaveLoad
import canny


class CNN(Dataset):
    def __init__(self):

        # Check file directory name to be correct
        fileDir = "normalizedSizeColor256"

        # Transforms images to a tensor to be able to load into dataloader
        transform = v2.Compose([
            v2.ToImage(),  # Convert PIL image to Tensor
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0,0,0], std=[1,1,1]),
            # v2.RandomVerticalFlip(.1),
            # v2.RandomHorizontalFlip(.1),
            # v2.RandomRotation(15)

        ])
        df = datasets.ImageFolder(fileDir, transform=transform)

        loader = DataLoader(df, batch_size=len(df), shuffle=True)
        images, labels = next(iter(loader))

        # Labels for the eye classification
        self.unique_labels = ["cataract","diabetic_retinopathy","glaucoma","normal"]

        # adds a canny channel
        processed_images = torch.stack([canny.addCannyLayer(image) for image in images])
        self.train_images = processed_images
        self.train_labels = labels

        # train/test split for the calidation set to have 20% of the original images
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
        self.in_to_h1 = nn.Conv2d(4, 72, (5, 5), padding=(2, 2))  # 72 x 256 x 256
        # Maxpool2d -> 72 x 128 x 128
        self.h1_to_h2 = nn.Conv2d(72, 16, (3, 3), padding=(1, 1))  # 16 x 128 x 128
        # Maxpool2d -> 16 x 64 x 64
        # Maxpool2d -> 16 x 32 x 32
        self.h3_to_h4 = nn.Linear(16*32*32, 32) # 32
        self.h4_to_out = nn.Linear(32, 4) # 4

    def forward(self, x):
        x = self.in_to_h1(x)  # 32 x 256 x 256
        x = F.dropout(x, p=.1)
        x = F.max_pool2d(x, (2, 2))  # 32 x 128 x 128
        x = self.h1_to_h2(x)  # 16 x 128 x 128
        x = F.max_pool2d(x, (2, 2))  # 16 x 64 x 64
        x = F.max_pool2d(x, (2, 2))  # 16 x 32 x 32
        x = torch.flatten(x, 1)
        x = F.relu(self.h3_to_h4(x))  # 32
        return self.h4_to_out(x) # 4


def trainNN(epochs=10, batch_size=32, lr=0.001, display_test_acc=True):
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
    optimizer = torch.optim.Adam(disease_classify.parameters(), lr=lr)

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
        # runs accuracy check in batches to not overload memory
        if epoch == epochs-1:
            sums = 0
            for i in range(len(cnn.train_images)):
                image = cnn.train_images[i].to(device)
                result = torch.argmax(disease_classify(image.view(-1, 4, 256, 256)).to(device),dim=1)
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
                    result = torch.argmax(disease_classify(image.view(-1, 4, 256, 256)).to(device),dim=1)
                    all_results.extend(result.cpu()) #adding the calculated predictions(result) from the batch to the array
                    if result == cnn.valid_labels[i]:
                        sums += 1 / len(cnn.valid_images)
                print(f"Accuracy on validation set: {sums:.4f}")
    cm = confusion_matrix(cnn.valid_labels, all_results)
    disp = ConfusionMatrixDisplay(cm, display_labels=cnn.unique_labels)
    disp.plot()
    # plt.savefig("confMatrix.png") uncomment to save the confusion matrix
    plt.show()
    return disease_classify






CNN = trainNN(epochs = 10, batch_size=32)
# SaveLoad.save(CNN, path = "4typeClassification.pth") #uncomment to save the neural network
