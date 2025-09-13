import os
import shutil
import random

# Paths
source_dir = "data/annotations"
train_dir = "data/annotations/train"
val_dir = "data/annotations/validation"

# Make sure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all JSON files
all_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

# Shuffle and split
random.shuffle(all_files)
split_idx = int(len(all_files) * 0.85)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# Copy files to respective folders
for f in train_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")