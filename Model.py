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
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder

# ---------------------- CONFIG ----------------------
EPOCHS = 100
LR = 1e-4

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
train_path = Path("training/")
test_path = Path("testing/")

data_train = ImageFolder(train_path, transform=transform)
data_test = ImageFolder(test_path, transform=transform)

# -------------------- Model --------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Identity()

# ------------------- Images to Pixel data conversion --------------------
embeddings = model(data_train[0][0].unsqueeze(0))  #model requires 4D tensor so from [3, 224, 224] this makes it [1, 3, 224, 224] since no batches are used it goes for 1 batch each
print(embeddings.T.shape)

def initParams():
    w1 = np.random.rand(1536, 2048) - 0.5
    b1 = np.random.rand(1536, 1) - 0.5
    w2 = np.random.rand(1200, 1536) - 0.5
    b2 = np.random.rand(1200, 1) - 0.5
    w3 = np.random.rand(1000, 1200) - 0.5
    b3 = np.random.rand(1000, 1) - 0.5
    return w1, b1, w2, b2, w3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def Softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0

def forwardPropagation(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = Softmax(z3)
    return z1, a1, z2, a2, z3, a3

def oneHot(Y):    # Y are maybe classes?
    emptyMatrix = np.zeros((Y.size, Y.max() + 1))
    emptyMatrix[np.arange(Y.size, Y)] = 1
    emptyMatrix = emptyMatrix.T
    return emptyMatrix
    





# Plan jest taki -> przepuść tensor [3, 224, 224] przez model (spłaszczony do vectora 3x224x224 size), return 2048 embeddings
# 2048 Embeddings -> przepuść przez NN i wypluj klasyfikację
# NN -> Input 2048 -> Hidden 1536 -> Hidden 1200 -> Output 1000 (bo 1000 klas w datasecie)


































# -------------------- Optimizer & Loss --------------------
# optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
# criterion = nn.BCELoss()  # We'll apply mask manually
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# -------------------- Training Loop --------------------
# for epoch in range(EPOCHS):
#   model.train()
#    epoch_loss = 0
#    batches = 0
#
#    for img, label in data_train:
#        if label is None:
#            continue
#
#        optimizer.zero_grad()
#        outputs = model(img)
#        loss = criterion(outputs, label)
#        loss.backward()
#        optimizer.step()
#
#    if batches > 0:
#        avg_loss = epoch_loss / batches
#        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
#        scheduler.step(avg_loss)
#    else:
#        print(f"Epoch {epoch+1}: No valid data processed.")






