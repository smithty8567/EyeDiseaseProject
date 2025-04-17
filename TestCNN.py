# loading in data from folders

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



#This is the directory where the folders of images are stored, could be different directory for each person
fileDir = "../EyeDiseaseProject/normalizedSize/normalizedSize"

#Transforms images in grayscale and to a tensor to be able to load into dataloader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert PIL image to Tensor
    # Add other transforms here, like normalization
])
dataset = datasets.ImageFolder(fileDir, transform=transform)

#print(dataset)

# Printing to see what number corresponds to what label

# {'cataract': 0, 'diabetic_retinopathy': 1, 'glaucoma': 2, 'normal': 3}
# print(dataset.class_to_idx)


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Testing to see images are random in the data loader

# images, labels = next(iter(dataloader))
#
#
#
# img_tensor = images[0]  # Shape: [1, H, W]
# img = img_tensor.squeeze(0).numpy()  # Shape: [H, W]
#
# plt.imshow(img, cmap='gray')  # Use grayscale colormap
# plt.title(f"Label: {labels[0]}")
# plt.axis('off')
# plt.show()



exit(0)