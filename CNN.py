#from tensorflow import datasets, layers, models
import matplotlib.pyplot as plt
from keras import layers, models
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

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

val_images, train_images, val_labels, train_labels = train_test_split(
    train_images, labels_train, test_size=0.8, random_state=42, stratify=labels_train
)

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
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=5, batch_size=32)

# Save the trained model
model.save('cnn_model.keras')

test_loss, test_acc = model.evaluate(test_images, labels_test, verbose=0)

train_acc = history.history.get('accuracy') or history.history.get('acc')
val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
train_loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

epochs_range = range(1, len(train_acc) + 1)

# Plot accuracy and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axs[0].plot(epochs_range, train_acc, 'o-', label='Train accuracy')
axs[0].plot(epochs_range, val_acc, 's-', label='Validation accuracy')
axs[0].hlines(test_acc, epochs_range[0], epochs_range[-1], colors='r', linestyles='--',
              label=f'Test accuracy = {test_acc:.4f}')
axs[0].set_title('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend()
axs[0].grid(True)

# Loss plot
axs[1].plot(epochs_range, train_loss, 'o-', label='Train loss')
axs[1].plot(epochs_range, val_loss, 's-', label='Validation loss')
axs[1].set_title('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('accuracy_loss_cnn.png')
plt.show()