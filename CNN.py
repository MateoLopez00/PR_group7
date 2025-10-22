#from tensorflow import datasets, layers, models
import matplotlib.pyplot as plt
from keras import layers, models
import pandas as pd
from PIL import Image
import numpy as np

df_test = pd.read_csv('MNIST-full/gt-test.tsv', sep='\t', header=None, names=['image_path', 'label'])
df_train = pd.read_csv('MNIST-full/gt-train.tsv', sep='\t', header=None, names=['image_path', 'label'])


def load_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array
# Load all images and labels
image_paths_test = df_test['image_path'].values
labels_test = df_test['label'].values
image_paths_train = df_train['image_path'].values
labels_train = df_train['label'].values

# Load images
test_images = np.array([load_image("MNIST-full/" + path) for path in image_paths_test])
train_images = np.array([load_image("MNIST-full/" + path) for path in image_paths_train])

# TODO make a val set from the train set

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for MNIST digits (0-9)
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

# Train the model
model.fit(train_images, labels_train, validation_data=(test_images, labels_test), epochs=5, batch_size=32)
# accuracy: 0.9949 - loss: 0.0170 - val_accuracy: 0.9906 - val_loss: 0.0291

# TODO validate on the test set and record findings / loss and accuracy curves for report