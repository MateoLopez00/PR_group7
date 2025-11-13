from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

class MNISTFashionDataLoader:
    def __init__(self, data_subpath="competition01/unifr-pr2025-competition-fashionmnist/Fashion-MNIST", random_state=42):
        # automatically find repo root (folder containing .git)
        self.root_dir = self._find_project_root()
        self.file_path = self.root_dir / data_subpath
        self.random_state = random_state

    def _find_project_root(self):
        """Find the git repo root dynamically."""
        path = Path(__file__).resolve()
        for parent in path.parents:
            if (parent / ".git").exists():
                return parent
        return Path.cwd()  # fallback

    def load_image(self, image_path):
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32).flatten() / 255.0
        return img_array

    def get_training_data(self):
        df_train = pd.read_csv(
            self.file_path / "gt-train.tsv",
            sep="\t", header=None, names=["image_path", "label"]
        )
        image_paths_train = df_train["image_path"].values
        labels_train = df_train["label"].values
        train_images = np.array([
            self.load_image(self.file_path / path) for path in image_paths_train
        ])
        return (train_images, labels_train)

    def get_test_data(self):
        df_test = pd.read_csv(
            self.file_path / "test-files.tsv",
            sep="\t", header=None, names=["image_path", "label"]
        )
        image_paths_test = df_test["image_path"].values
        test_images = np.array([
            self.load_image(self.file_path / path) for path in image_paths_test
        ])
        return test_images

    def get_train_and_validation_set(self, validation_size=0.2):
        X, y = self.get_training_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=self.random_state
        )
        return X_train, y_train, X_val, y_val
    def create_submission_file(self, predictions, file_name="submission.csv"):
        df_sample_submission = pd.read_csv(self.file_path / "sample-submission.csv")
        assert len(df_sample_submission) == len(predictions), "Prediction length mismatch"
        df_sample_submission["Class"] = predictions
        output_path = self.root_dir / "competition01/submissions" / file_name
        df_sample_submission.to_csv(output_path, index=False)

