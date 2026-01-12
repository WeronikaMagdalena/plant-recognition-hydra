import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from c45 import build_tree
from balancer import filter_and_balance
from predict import predict


# -----------------------------
# 1. Load all data
# -----------------------------
df_all = pd.concat([
    pd.read_csv("../embeddings/Train.csv"),
    pd.read_csv("../embeddings/Validation.csv"),
    pd.read_csv("../embeddings/Test.csv"),
], ignore_index=True)

print("Total samples:", len(df_all))
print("Total classes:", df_all["label"].nunique())

# -----------------------------
# 2. Select top 5 most frequent classes
# -----------------------------
class_counts = df_all["label"].value_counts()
top_classes = class_counts.head(5).index.tolist()

print("Top 5 classes:", top_classes)
print(class_counts.loc[top_classes])

df_filtered = df_all[df_all["label"].isin(top_classes)].copy()

# -----------------------------
# 3. Split BEFORE balancing (important)
# -----------------------------
df_train, df_temp = train_test_split(
    df_filtered,
    test_size=0.30,
    stratify=df_filtered["label"],
    random_state=42
)

df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    stratify=df_temp["label"],
    random_state=42
)

# -----------------------------
# 4. Balance TRAIN ONLY
# -----------------------------
X_train = df_train.drop(columns=["label"]).values.astype(float)
y_train = df_train["label"].values

X_train_bal, y_train_bal = filter_and_balance(X_train, y_train)

print("Balanced train class counts:")
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique, counts)))

# -----------------------------
# 5. Train decision tree
# -----------------------------
tree = build_tree(
    X_train_bal,
    y_train_bal,
    max_depth=15,
    min_samples=5
)

# -----------------------------
# 6. Evaluate (sanity check)
# -----------------------------

X_val = df_val.drop(columns=["label"]).values.astype(float)
y_val = df_val["label"].values

preds = predict(X_val, tree)

print("\nSample predictions vs truth:")
for p, t in zip(preds[:10], y_val[:10]):
    print(f"pred={p}  true={t}")

accuracy = (preds == y_val).mean()
print("\nValidation accuracy:", accuracy)
