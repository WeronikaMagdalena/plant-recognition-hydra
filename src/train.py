import pandas as pd
from sklearn.metrics import accuracy_score
from load_embeddings import load_split
from src.balancer import filter_and_balance
from src.c45 import build_tree
from src.predict import predict

df_train = pd.read_csv("../embeddings/Train.csv")
df_val   = pd.read_csv("../embeddings/Validation.csv")
df_test  = pd.read_csv("../embeddings/Test.csv")

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

print(df_all["label"].value_counts())

X = df_all.drop(columns=["label"]).values.astype(float)
y = df_all["label"].values

X_balanced, y_balanced = filter_and_balance(X, y)

feature_cols = df_all.drop(columns=["label"]).columns

df_balanced = pd.DataFrame(X_balanced, columns=feature_cols)
df_balanced["label"] = y_balanced

print(df_balanced["label"].value_counts())


from sklearn.model_selection import train_test_split

df_train, df_temp = train_test_split(
    df_balanced,
    test_size=0.30,
    stratify=df_balanced["label"],
    random_state=42
)

df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    stratify=df_temp["label"],
    random_state=42
)

X_train = df_train.drop(columns=["label"]).values.astype(float)
y_train = df_train["label"].values

X_val = df_val.drop(columns=["label"]).values.astype(float)
y_val = df_val["label"].values

X_test = df_test.drop(columns=["label"]).values.astype(float)
y_test = df_test["label"].values

tree = build_tree(X_train, y_train, depth=0, max_depth=5, min_samples=10)

y_pred = predict(X_val, tree)
print("Predictions:", y_pred[:10])
print("True labels:", y_val[:10])

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

def print_tree(node, depth=0):
    indent = "  " * depth
    if node.is_leaf_node():
        print(f"{indent}Leaf: class={node.value}")
    else:
        print(f"{indent}Feature {node.feature} <= {node.threshold}")
        print_tree(node.left, depth+1)
        print_tree(node.right, depth+1)
