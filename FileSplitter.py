import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil

train_dir = Path("training/")
test_dir = Path("testing/")

folders_path_list = [p for p in train_dir.iterdir()]
for folder in folders_path_list:
    amount_of_files_training = len(os.listdir(folder)) * 0.8
    for i, img_path in enumerate(os.listdir(folder)):
        if i >= amount_of_files_training:
            os.makedirs(test_dir.joinpath(folder.name), exist_ok=True)
            shutil.move(train_dir.joinpath(folder.name).joinpath(img_path), test_dir.joinpath(folder.name))













































