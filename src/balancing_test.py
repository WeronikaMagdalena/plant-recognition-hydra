import numpy as np

from src.load_embeddings import load_split
from src.balancer import filter_and_balance
from collections import Counter

# Load embeddings
X_train, y_train = load_split("../embeddings/Train.csv")

# Filter + balance
X_train_bal, y_train_bal = filter_and_balance(
    X_train,
    y_train,
    min_samples_per_class=50
)

# -------------------- Check results --------------------
classes, counts = np.unique(y_train_bal, return_counts=True)
print("Number of classes after filtering + balancing:", len(classes))
print("Samples per class:", counts[:10])
print("All classes equal:", np.all(counts == counts[0]))
