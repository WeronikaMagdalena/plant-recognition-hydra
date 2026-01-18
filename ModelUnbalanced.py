import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------- CONFIG ----------------------
EPOCHS = 100
LR = 0.05

# -------------------- Dataset and Loader --------------------
class EmbeddingDataset(Dataset):
    def __init__(self, csv_path=None, X=None, y=None):
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            self.X = df.iloc[:, :-1].values.astype(np.float32)
            self.y = df.iloc[:, -1].values.astype(np.int64)
        else:
            self.X = X
            self.y = y
            
        self.num_classes = int(self.y.max() + 1) if len(self.y) > 0 else 0
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


train_data = EmbeddingDataset("first_10_classes.csv")
test_data = EmbeddingDataset("Test.csv")
val_data = EmbeddingDataset("Validation.csv")


NUM_CLASSES_TRAIN = train_data.num_classes
NUM_CLASSES_TEST = test_data.num_classes
NUM_CLASSES_VALIDATION = val_data.num_classes
INPUT_DIM = 2048

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


def initParams(input_dim, num_classes):
    w1 = np.random.rand(1536, input_dim) - 0.5
    b1 = np.random.rand(1536, 1) - 0.5
    w2 = np.random.rand(1200, 1536) - 0.5
    b2 = np.random.rand(1200, 1) - 0.5
    w3 = np.random.rand(num_classes, 1200) - 0.5
    b3 = np.random.rand(num_classes, 1) - 0.5
    return w1, b1, w2, b2, w3, b3

def ReLU(Z):
    return np.maximum(Z, 0) 

def Softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_vals = np.exp(Z_shifted)
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

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

def oneHot(Y, num_classes):
    num_samples = Y.size
    one_hot = np.zeros((num_classes, num_samples))
    one_hot[Y, np.arange(num_samples)] = 1
    return one_hot
    
def backPropagation(Z1, A1, Z2, A2, Z3, A3, W3, W2, X, Y):
    m = Y.size
    one_hot_Y = oneHot(Y, NUM_CLASSES_TRAIN)

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def updateParams(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1

    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2

    w3 = w3 - alpha * dW3
    b3 = b3 - alpha * db3
    return w1, b1, w2, b2, w3, b3

def getPredictions(A3):
    return np.argmax(A3, 0)

def getAccuracy(predictions, Y):
    print('Predictions: \n')
    print(predictions)
    print('\n')
    print('Labels: \n')
    print(Y)
    return np.sum(predictions == Y) / Y.size

def train_custom_nn(train_loader, iterations, alpha):
    W1, B1, W2, B2, W3, B3 = initParams(INPUT_DIM, NUM_CLASSES_TRAIN)

    for epoch in range(iterations):  
        for X_batch, y_batch in train_loader:

            # Convert to NumPy
            X_np = X_batch.numpy().T     # shape: [2048, batch]
            Y_np = y_batch.numpy()       # shape: [batch]

            Z1, A1, Z2, A2, Z3, A3 = forwardPropagation(W1, B1, W2, B2, W3, B3, X_np)
            dW1, db1, dW2, db2, dW3, db3 = backPropagation(Z1, A1, Z2, A2, Z3, A3, W3, W2, X_np, Y_np)
            W1, B1, W2, B2, W3, B3 = updateParams(W1, B1, W2, B2, W3, B3,dW1, db1, dW2, db2, dW3, db3, alpha)

        print(f"=== Epoch {epoch+1}/{iterations} ===")
        predictions = getPredictions(A3)
        
        acc = getAccuracy(predictions, Y_np)
        print("  Accuracy:", acc)

    return W1, B1, W2, B2, W3, B3


W1, B1, W2, B2, W3, B3 = train_custom_nn(train_loader, EPOCHS, alpha=LR)





