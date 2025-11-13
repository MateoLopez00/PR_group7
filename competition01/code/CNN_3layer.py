import tensorflow as tf
from tensorflow import keras
from data_loader import MNISTFashionDataLoader


loader = MNISTFashionDataLoader()
X_train, y_train, X_val, y_val = loader.get_train_and_validation_set()
X_test = loader.get_test_data()

# rehape data for CNN input
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 2. Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=7.5,
    width_shift_range=0.078,
    height_shift_range=0.075,
    zoom_range=0.085
)

# 3. Model definition (CNN-3-128 style)
model = keras.Sequential([
    keras.layers.Conv2D(128, (3,3), activation="relu", padding="same", input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

# 4. Compile and train
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=50,                     # longer training is key
    validation_data=(X_val, y_val),
    verbose=2
)

predictions = model.predict(X_test)
predicted_classes = tf.argmax(predictions, axis=1).numpy()

loader.create_submission_file(predicted_classes, file_name="submission_cnn_3layer_adam50.csv")

