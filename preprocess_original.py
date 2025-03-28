import os
import numpy as np
import cv2
import gc  # Garbage Collector
from tqdm import tqdm

# 🔹 Set paths
DATASET_PATH = "/Users/divyanshsaxena/Documents/project/train"
IMAGE_PATH = os.path.join(DATASET_PATH, "img")
LABEL_PATH = os.path.join(DATASET_PATH, "labels.npy")
OUTPUT_PATH = "/Users/divyanshsaxena/Documents/project/processed_data"

# 🔹 Image processing parameters
IMAGE_SIZE = (224, 224)  # Resize images to 224x224
BATCH_SIZE = 5000  # Process images in batches to prevent memory overflow

def extract_label_from_filename(filename):
    """Extract label from filename. Modify based on naming convention."""
    return filename.split("_")[0]  # Assumes "class1_001.jpg" -> "class1"

def generate_labels(image_files):
    """Generate labels from filenames if labels.npy is missing."""
    print("⚠️ No labels.npy found. Generating labels from filenames...")
    labels = [extract_label_from_filename(img) for img in image_files]
    
    # Convert labels to integers if they are class names
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in labels])  # Convert to numbers
    
    print("✅ Generated labels:", label_map)
    return labels

def preprocess_data():
    """Preprocess images in batches to avoid memory overload."""
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"❌ Error: Image folder '{IMAGE_PATH}' not found!")

    image_files = sorted(os.listdir(IMAGE_PATH))
    print(f"📂 Found {len(image_files)} images. Processing in batches...")

    # Load or generate labels
    if os.path.exists(LABEL_PATH):
        print("✅ Labels found! Loading...")
        labels = np.load(LABEL_PATH, allow_pickle=True)
    else:
        labels = generate_labels(image_files)  # Generate labels if missing

    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Ensure output directory exists

    # 🔹 Process in batches
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        processed_images = []

        for img_name in tqdm(batch_files, desc=f"Processing Batch {i//BATCH_SIZE + 1}"):
            img_path = os.path.join(IMAGE_PATH, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, IMAGE_SIZE)  # Resize
            img = img / 255.0  # Normalize to [0,1]
            processed_images.append(img)

        processed_images = np.array(processed_images, dtype=np.float32)

        # Save each batch separately
        np.save(os.path.join(OUTPUT_PATH, f"data_batch_{i//BATCH_SIZE}.npy"), processed_images)
        
        # 🔹 Explicitly release memory
        del processed_images
        gc.collect()  # Force garbage collection

    # Save labels (only once)
    np.save(os.path.join(OUTPUT_PATH, "labels.npy"), labels)
    
    print("✅ All images processed in batches and saved!")

if __name__ == "__main__":
    preprocess_data()