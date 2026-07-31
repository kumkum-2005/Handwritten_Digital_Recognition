import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ===========================
# Load MNIST Dataset
# ===========================

print("Loading MNIST Dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ===========================
# Preprocessing
# ===========================

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

# ===========================
# Build CNN Model
# ===========================

model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation="relu",
    input_shape=(28,28,1)
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation="relu"
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(
    128,
    activation="relu"
))

model.add(Dropout(0.3))

model.add(Dense(
    10,
    activation="softmax"
))

# ===========================
# Compile Model
# ===========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===========================
# Train Model
# ===========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# ===========================
# Evaluate Model
# ===========================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy : {:.2f}%".format(accuracy * 100))

# ===========================
# Save Model
# ===========================

os.makedirs("models", exist_ok=True)

model.save("models/digit_model.keras")

print("\nModel Saved Successfully!")
print("Location : models/digit_model.keras")