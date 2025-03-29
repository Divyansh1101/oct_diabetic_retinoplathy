import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ✅ Set Paths
DATA_PATH = "/Users/divyanshsaxena/Documents/project/processed_data/data.npy"
LABELS_PATH = "/Users/divyanshsaxena/Documents/project/processed_data/labels.npy"
MODEL_PATH = "/Users/divyanshsaxena/Documents/project/model.keras"
ENCODER_PATH = "/Users/divyanshsaxena/Documents/project/label_encoder.pkl"

# ✅ Load Data
print("📂 Loading data...")

if not os.path.exists(DATA_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("❌ data.npy or labels.npy not found! Run preprocess.py first.")

X = np.load(DATA_PATH, allow_pickle=True)  # Shape: (993, 224, 224, 3)
y = np.load(LABELS_PATH, allow_pickle=True)  # Shape: (993,)

# ✅ Encode Labels (Fix for 'CNV' Error)
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)  # Converts 'CNV', 'DME' → 0,1,2
y_categorical = keras.utils.to_categorical(y_numeric)

# ✅ Save Label Encoder for Predictions
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ✅ Model Architecture
model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(label_encoder.classes_), activation="softmax")  # Output Layer
])

# ✅ Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Train Model
print("🚀 Training Model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Save Model in Recommended Format
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")