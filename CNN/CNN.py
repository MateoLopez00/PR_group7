#from tensorflow import datasets, layers, models
import matplotlib.pyplot as plt
from keras import layers, models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.data_loader as data_loader

config = {
    "num_conv_layers": 3,        # number of Conv2D + MaxPool blocks
    "kernel_size": (7, 7),       # kernel size for all conv layers
    "learning_rate": 1e-4,       # optimizer learning rate
    "epochs": 5,
}


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

def build_cnn(num_conv_layers=3, kernel_size=(3,3), input_shape=(28,28,1), num_classes=10, filters_start=32):
    """
    Build a Sequential CNN with `num_conv_layers` Conv2D+MaxPool blocks.
    Each block doubles the number of filters starting from filters_start.
    """
    model = models.Sequential()
    for i in range(num_conv_layers):
        filters = int(filters_start * (2 ** i))
        if i == 0:
            model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Build and compile the model using config values
model = build_cnn(num_conv_layers=config["num_conv_layers"],
                  kernel_size=config["kernel_size"],
                  input_shape=(28,28,1))

optimizer = Adam(learning_rate=config["learning_rate"])
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=config["epochs"],
                    batch_size=32)

# Save the trained model
model.save(f'CNN/cnn_modelk{config["kernel_size"]}l{config["num_conv_layers"]}.keras')

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
plt.savefig(f'CNN/accuracy_loss_cnn{config["kernel_size"]}l{config["num_conv_layers"]}.png')
plt.show()