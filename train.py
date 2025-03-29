import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import gc

# ✅ Set Paths
DATA_PATH = "/Users/divyanshsaxena/Documents/project/processed_data/final_data.npy"
LABELS_PATH = "/Users/divyanshsaxena/Documents/project/processed_data/labels.npy"
MODEL_PATH = "/Users/divyanshsaxena/Documents/project/model.keras"
ENCODER_PATH = "/Users/divyanshsaxena/Documents/project/label_encoder.pkl"

# ✅ Load Labels (Safe to Load Fully)
y = np.load(LABELS_PATH, allow_pickle=True)

# Encode Labels
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_numeric)

# ✅ Save Label Encoder
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

# ✅ Train-Test Split (Using Indices to Avoid Full Data Loading)
num_samples = len(y_categorical)  # ✅ Now y_categorical is still defined
indices = np.arange(num_samples)
X_train_idx, X_test_idx, y_train, y_test = train_test_split(indices[:5000], y_categorical[:5000], test_size=0.2, random_state=42)

# ✅ Now delete y after it's used
del y, y_numeric, y_categorical
gc.collect()

# ✅ Memory-Mapped Data Loader (Loading Only When Needed)
X_memmap = np.memmap(DATA_PATH, dtype="float32", mode="r", shape=(108309, 224, 224))

print("🚀 Checking for NaN values in dataset...")
print(f"Train NaN count: {np.isnan(X_memmap[X_train_idx]).sum()}")
print(f"Test NaN count: {np.isnan(X_memmap[X_test_idx]).sum()}")

@tf.function
def load_sample(idx):
    img = tf.numpy_function(lambda i: X_memmap[i] / 255.0, [idx], tf.float32)  # Load only required sample
    img = tf.expand_dims(img, axis=-1)  # Add channel dimension
    img.set_shape((224, 224, 1))  # Explicitly define shape
    return img

def tf_data_generator(X_idx, y_data, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((X_idx, y_data))

    def load_data(idx, label):
        img = load_sample(idx)
        return img, label

    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Prefetch for efficiency
    return dataset

batch_size = 1 if len(X_train_idx) > 10000 else 4
train_dataset = tf_data_generator(X_train_idx, y_train, batch_size=batch_size)
test_dataset = tf_data_generator(X_test_idx, y_test, batch_size=batch_size)

# ✅ Further Simplified Model Architecture
model = keras.Sequential([
    keras.Input(shape=(224, 224, 1)),  # Grayscale images
    layers.Conv2D(2, (3, 3), activation=keras.layers.LeakyReLU(alpha=0.1)),  # Reduce filters even more
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(4, (3, 3), activation=keras.layers.LeakyReLU(alpha=0.1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(8, activation=keras.layers.LeakyReLU(alpha=0.1)),  # Further reduced dense layer
    layers.Dense(len(label_encoder.classes_), activation=keras.activations.log_softmax)
])

# ✅ Compile Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0), loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Train Model Using `tf.data.Dataset`
print("🚀 Training Model...")
model.fit(train_dataset, epochs=2)  # Keeping epochs low

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

del train_dataset, test_dataset, model
gc.collect()