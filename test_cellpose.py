import os
import cv2
import h5py
import numpy as np
from cellpose import models, io

# --- 1. File Paths ---
# Make sure these match your actual folder structure!
image_path = "images/test/87.png"
pos_h5 = "annotations/test/positive/87.h5"
neg_h5 = "annotations/test/negative/87.h5"

# --- 2. Load the Image ---
print(f"Loading image: {image_path}")
image = io.imread(image_path)
# Ensure it's RGB
if image.ndim == 3 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 3. Load the Pathologist Ground Truth (.h5) ---
def load_coords(path):
    if not os.path.exists(path):
        return []
    with h5py.File(path, 'r') as f:
        return np.asarray(f['coordinates'])

all_coords = []
for c in load_coords(pos_h5): all_coords.append((int(c[0]), int(c[1])))
for c in load_coords(neg_h5): all_coords.append((int(c[0]), int(c[1])))

# --- 4. Run Cellpose Detection ---
print("Downloading/Loading pre-trained Cellpose 'nuclei' model...")
# Updated for Cellpose v4+
model = models.CellposeModel(model_type='nuclei') 

print("Detecting cells blindly... (This might take a few seconds on CPU)")
# Channels=[0,0] processes the image in grayscale for generic shape detection
# We capture the full results tuple and extract just the masks at index 0
results = model.eval(image, diameter=None, channels=[0,0])
masks = results[0]

# --- 5. Cross-Check: Point-in-Polygon ---
true_positives = 0
false_negatives = 0
matched_mask_ids = set()

# Check every human-marked dot
for x, y in all_coords:
    # Ensure the coordinate is inside the image bounds
    if 0 <= y < masks.shape[0] and 0 <= x < masks.shape[1]:
        mask_id = masks[y, x]
        if mask_id > 0:
            # The dot fell inside a cell boundary!
            true_positives += 1
            matched_mask_ids.add(mask_id)
        else:
            # The dot landed on empty background (Cellpose missed it)
            false_negatives += 1
    else:
        false_negatives += 1

# Calculate False Positives (cells Cellpose found, but humans didn't mark)
total_detected_cells = np.max(masks)
false_positives = total_detected_cells - len(matched_mask_ids)

# --- 6. Calculate and Print Metrics ---
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*40)
print(" 🔬 CELLPOSE DETECTION BENCHMARK ")
print("="*40)
print(f"Total Ground Truth Cells (.h5): {len(all_coords)}")
print(f"Total Cells Detected by AI:   {total_detected_cells}")
print("-" * 40)
print(f"✅ True Positives (Correctly found):   {true_positives}")
print(f"❌ False Positives (Mistaken non-cells): {false_positives}")
print(f"⚠️ False Negatives (Missed real cells):  {false_negatives}")
print("-" * 40)
print(f"Precision: {precision:.2%} (When it guesses a cell, how often is it right?)")
print(f"Recall:    {recall:.2%} (Out of all real cells, how many did it find?)")
print(f"F1-Score:  {f1:.2%} (Overall Accuracy Balance)")
print("="*40)