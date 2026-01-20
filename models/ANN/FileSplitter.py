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
validation_dir = Path("validation/")

folders_path_list = [p for p in train_dir.iterdir()]
for folder in folders_path_list:
    files = os.listdir(folder) 
    val_end = int(len(files) * 0.1)
    test_end = val_end + int(len(files) * 0.2)
    for i, img_path in enumerate(files):
        if i < val_end:
            os.makedirs(validation_dir.joinpath(folder.name), exist_ok=True)
            shutil.move(train_dir.joinpath(folder.name).joinpath(img_path), validation_dir.joinpath(folder.name))
        elif val_end <= i < test_end:
            os.makedirs(test_dir.joinpath(folder.name), exist_ok=True)
            shutil.move(train_dir.joinpath(folder.name).joinpath(img_path), test_dir.joinpath(folder.name))
        else:
            continue













































