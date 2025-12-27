import pandas as pd

def load_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(float)
    y = df["label"].values
    return X, y
