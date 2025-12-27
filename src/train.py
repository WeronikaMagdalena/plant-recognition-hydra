from load_embeddings import load_split

X_train, y_train = load_split("../embeddings/Train.csv")
X_val, y_val     = load_split("../embeddings/Validation.csv")
X_test, y_test   = load_split("../embeddings/Test.csv")

print(f"Train samples: {X_train[0]}, Validation samples: {X_val[0]}, Test samples: {X_test[0]}")
