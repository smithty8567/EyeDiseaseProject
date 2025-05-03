import numpy as np
import cv2
import torch
def addCannyLayer(img_tensor):
    # img_tensor: [3, H, W], float32
    np_img = img_tensor.numpy().transpose(1, 2, 0)  # [H, W, 3]
    gray = np.clip((np_img * 255), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 0, 15).astype(np.float32) / 255.0
    edge_tensor = torch.from_numpy(edges).unsqueeze(0)  # [1, H, W]
    return torch.cat((img_tensor, edge_tensor), dim=0)  # [4, H, W]