import os
import numpy as np

# Paths
PROCESSED_PATH = "/Users/divyanshsaxena/Documents/project/processed_data"
OUTPUT_FILE = os.path.join(PROCESSED_PATH, "final_data.npy")
LABELS_FILE = os.path.join(PROCESSED_PATH, "labels.npy")

# Get all batch files
batch_files = sorted([f for f in os.listdir(PROCESSED_PATH) if f.startswith("data_batch_") and f.endswith(".npy")])

# Load and merge all batches
merged_data = []
for batch in batch_files:
    batch_path = os.path.join(PROCESSED_PATH, batch)
    print(f"🔄 Loading {batch}...")
    batch_data = np.load(batch_path)
    merged_data.append(batch_data)

# Concatenate into a single array
merged_data = np.concatenate(merged_data, axis=0)

# Save final dataset
np.save(OUTPUT_FILE, merged_data)
print(f"✅ Merged dataset saved at: {OUTPUT_FILE}")

# Check labels
if os.path.exists(LABELS_FILE):
    print(f"✅ Labels file exists: {LABELS_FILE}")
else:
    print("⚠️ Warning: Labels file not found!")

# Print dataset shape for verification
print(f"📊 Final dataset shape: {merged_data.shape}")