#from tensorflow import datasets, layers, models
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.data_loader as data_loader




X_train, y_train, X_val, y_val = data_loader.get_train_and_validation_set() 
X_test, y_test = data_loader.get_test_data()

def ensure_4d(X):
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 28*28:
        X = X.reshape((-1, 28, 28, 1))
    elif X.ndim == 3 and X.shape[1] == 28 and X.shape[2] == 28:
        X = X.reshape((-1, 28, 28, 1))
    return X.astype('float32')

X_train = ensure_4d(X_train)
X_val   = ensure_4d(X_val)
X_test  = ensure_4d(X_test)

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
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=5, batch_size=32)

# Save the trained model
model.save('CNN/cnn_model.keras')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

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
plt.savefig('CNN/accuracy_loss_cnn.png')
plt.show()