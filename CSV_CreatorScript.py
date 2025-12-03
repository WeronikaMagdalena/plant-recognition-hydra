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
from torchvision.datasets import ImageFolder
from tqdm import tqdm

train_dir = Path("training/")
test_dir = Path("testing/")
validation_dir = Path("validation/")

# -------------------- Transforms --------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- Dataset and Loader --------------------
train_dir = Path("training/")
test_dir = Path("testing/")
validation_dir = Path("validation/")

data_train = ImageFolder(train_dir, transform=transform)
data_test = ImageFolder(test_dir, transform=transform)
data_validation = ImageFolder(validation_dir, transform=transform)

batch_size = 32 
train_loader = DataLoader(
    data_train, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,  
)

test_loader = DataLoader(
    data_test, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4, 
)

validation_loader = DataLoader(
    data_validation, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4, 
)

# -------------------- Model --------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Identity()
model.eval()

if __name__ == '__main__':
    # ------------------- Images to Pixel data conversion -------------------
    all_embeddings = []
    all_labels = []

    with torch.inference_mode():  # Disable gradient calculation for inference
        for images, labels in tqdm(train_loader, desc="Processing images"):
            embeddings = model(images)
            all_embeddings.append(embeddings.numpy())
            all_labels.append(labels.numpy())

    # Convert to numpy arrays
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    result = pd.DataFrame(all_embeddings)
    result['label'] = all_labels

    # Save to CSV
    result.to_csv("Train.csv", index=False)
























