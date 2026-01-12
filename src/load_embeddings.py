import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_split(path):
    logging.info(f"Loading data from {path}")

    df = pd.read_csv(path)
    logging.info(f"CSV loaded with shape {df.shape}")

    if "label" not in df.columns:
        logging.error("Column 'label' not found in dataframe")
        raise ValueError("Missing 'label' column")

    X = df.drop(columns=["label"]).values.astype(float)
    y = df["label"].values

    logging.info(f"Split complete: X shape {X.shape}, y length {len(y)}")

    return X, y
