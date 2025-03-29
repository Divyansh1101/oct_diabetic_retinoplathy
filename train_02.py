import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ✅ Set Paths
PROCESSED_PATH = "/Users/divyanshsaxena/Documents/project/processed_data"
LABELS_PATH = os.path.join(PROCESSED_PATH, "labels.npy")
MODEL_PATH = "/Users/divyanshsaxena/Documents/project/model.keras"
ENCODER_PATH = "/Users/divyanshsaxena/Documents/project/label_encoder.pkl"

# ✅ Load Labels
print("📂 Loading labels...")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("❌ labels.npy not found! Run preprocess.py first.")

y = np.load(LABELS_PATH, allow_pickle=True)  # Shape: (total_samples,)
print(f"🔍 Unique Labels Found: {np.unique(y)}")
print(f"✅ Number of Unique Labels: {len(np.unique(y))}")

# ✅ Encode Labels
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)
num_classes = np.max(y_numeric) + 1  # ✅ Get correct number of classes
y_categorical = keras.utils.to_categorical(y_numeric, num_classes=num_classes)  # ✅ Fix one-hot encoding

# ✅ Save Label Encoder for Predictions
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

# ✅ Get List of Processed Data Batches
batch_files = sorted([f for f in os.listdir(PROCESSED_PATH) if f.startswith("data_batch_") and f.endswith(".npy")])

if not batch_files:
    raise FileNotFoundError("❌ No processed batch files found! Run preprocess.py first.")

# ✅ Print class info
print(f"🔹 Number of classes: {num_classes}")

# ✅ Model Architecture (Handles Grayscale Input)
model = keras.Sequential([
    keras.Input(shape=(224, 224, 1)),  # Grayscale images (1 channel)
    layers.Conv2D(32, (3, 3)),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3)),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")  # Output Layer
])

# ✅ Compile Model with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ✅ Train Model Batch-by-Batch
print("🚀 Training Model...")
total_samples = len(y_categorical)
print(f"✅ y_categorical shape: {y_categorical.shape}")

for batch_index, batch_file in enumerate(batch_files):
    batch_path = os.path.join(PROCESSED_PATH, batch_file)
    print(f"🔄 Loading {batch_file}...")

    # Load batch
    X_batch = np.load(batch_path)
    X_batch = X_batch.astype("float32") / 255.0  # ✅ Normalize

    # Ensure correct label slicing
    start_idx = batch_index * len(X_batch)
    end_idx = min(start_idx + len(X_batch), y_categorical.shape[0])  # ✅ Prevent index overflow

    y_batch = y_categorical[start_idx:end_idx]

    # Validate batch size consistency
    if len(y_batch) != len(X_batch):
        raise ValueError(f"❌ Mismatch: X_batch ({len(X_batch)}) vs y_batch ({len(y_batch)})")

    # Print batch shapes for validation
    print(f"Batch {batch_index}: X_batch shape = {X_batch.shape}, y_batch shape = {y_batch.shape}")

    # Train on batch
    model.fit(X_batch, y_batch, epochs=10, batch_size=32)

# ✅ Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")