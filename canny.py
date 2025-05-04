import numpy as np
import cv2
import torch

def addCannyLayer(img_tensor):
    #transform tensor image to proper image
    np_img = img_tensor.numpy().transpose(1, 2, 0)

    #grayscale image
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    #do edge detection on grayscale image
    edges = cv2.Canny(gray, 0, 15).astype(np.float32) / 255.0

    #convert our numpy image to a tensor image
    edge_tensor = torch.from_numpy(edges).unsqueeze(0)

    #return new edge channel with original image
    return torch.cat((img_tensor, edge_tensor), dim=0)