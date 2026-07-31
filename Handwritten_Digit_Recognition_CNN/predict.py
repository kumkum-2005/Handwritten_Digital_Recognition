import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================================
# Load Trained Model
# =====================================

MODEL_PATH = "models/digit_model.keras"

if not os.path.exists(MODEL_PATH):
    print("Error: Trained model not found!")
    print("Please run train_model.py first.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

print("\n====================================")
print(" Handwritten Digit Recognition")
print("====================================")

# =====================================
# Take Image Path from User
# =====================================

image_path = input("\nEnter image path: ")

if not os.path.exists(image_path):
    print("\nError: Image file not found!")
    exit()

# =====================================
# Read Image
# =====================================

# Read original image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("\nError: Unable to load image!")
    exit()

# =====================================
# Resize Image
# =====================================

image = cv2.resize(image, (28, 28))

# =====================================
# Invert Colors
# (MNIST = White Digit on Black Background)
# =====================================

image = 255 - image

# =====================================
# Normalize Image
# =====================================

image = image.astype("float32") / 255.0

# =====================================
# Prepare for Prediction
# =====================================

input_image = image.reshape(1, 28, 28, 1)

# =====================================
# Predict
# =====================================

prediction = model.predict(input_image)

predicted_digit = np.argmax(prediction)

confidence = np.max(prediction) * 100

# =====================================
# Show Result
# =====================================

print("\n========== Prediction ==========")
print("Predicted Digit :", predicted_digit)
print("Confidence      : {:.2f}%".format(confidence))

# =====================================
# Display Image
# =====================================

plt.figure(figsize=(5,5))
plt.imshow(image, cmap="gray", interpolation="nearest")
plt.title(f"Prediction : {predicted_digit}")
plt.axis("off")
plt.show()

# =====================================
# Show All Probabilities
# =====================================

print("\nProbability for each digit:\n")

for i in range(10):
    print(f"{i} : {prediction[0][i] * 100:.2f}%")