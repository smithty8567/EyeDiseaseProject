import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

size = [128,128]
folderPath = "dataset"
def resizeImage(im, size):
    im = cv2.resize(im, (size[0], size[1]))
    return im


for folderName in os.listdir(folderPath):
    print(f"new folder:: {folderName}")
    folder_path = os.path.join(folderPath, folderName)
    index = 0

    output_folder = os.path.join("normalizedSize", folderName)
    os.makedirs(output_folder, exist_ok=True)

    for file_path in os.listdir(folder_path):
        im = cv2.cvtColor(cv2.imread(folder_path + "/" + file_path), cv2.COLOR_BGR2GRAY)
        cv2.imwrite("normalizedSize/" + folderName + "/" + file_path, resizeImage(im, size))
        if index % 25 == 0:
            print(index)
        index += 1

print('done')