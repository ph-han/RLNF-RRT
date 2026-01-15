import torch
import numpy as np
from PIL import Image

def load_cspace_img_to_np(path):
    return np.array(Image.open(path).convert('L'))

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")