import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# Get the folder where this script is located
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "MNIST-full"
train_csv = DATA_DIR / "gt-train.tsv"

def get_training_data():
    df_train = pd.read_csv(
        os.path.join(DATA_DIR, "gt-train.tsv"),
        sep="\t", header=None, names=["image_path", "label"]
    )
    image_paths_train = df_train["image_path"].values
    labels_train = df_train["label"].values
    train_images = np.array(
        [load_image(os.path.join(DATA_DIR, path)) for path in image_paths_train]
    )
    return train_images, labels_train


def get_test_data():
    df_test = pd.read_csv(
        os.path.join(DATA_DIR, "gt-test.tsv"),
        sep="\t", header=None, names=["image_path", "label"]
    )
    image_paths_test = df_test["image_path"].values
    labels_test = df_test["label"].values
    test_images = np.array(
        [load_image(os.path.join(DATA_DIR, path)) for path in image_paths_test]
    )
    return test_images, labels_test


def load_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    img_array = img_array.flatten()
    img_array = img_array / 255.0  # normalize
    return img_array


def get_train_and_validation_set(val_size=0.2, random_state=42):
    training_data, training_labels = get_training_data()
    X_train, X_val, y_train, y_val = train_test_split(
        training_data, training_labels, test_size=val_size, random_state=random_state
    )
    return X_train, y_train, X_val, y_val
