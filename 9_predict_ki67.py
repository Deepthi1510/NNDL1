import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("=========================================")
print("      FINAL KI-67 SCORING INFERENCE      ")
print("=========================================\n")

# --- 1. Configuration ---
# CHANGE THIS to the name of any image in your BCData/images/test/ folder!
TEST_IMAGE_NAME = "18.png" 

IMG_PATH = f"images/test/{TEST_IMAGE_NAME}"
POS_H5_PATH = f"annotations/test/positive/{TEST_IMAGE_NAME.replace('.png', '.h5')}"
NEG_H5_PATH = f"annotations/test/negative/{TEST_IMAGE_NAME.replace('.png', '.h5')}"
PATCH_SIZE = 64

# --- 2. Load the Models ---
print("Loading the Grand Ensemble Models...")
model_1 = tf.keras.models.load_model('resnet_model.keras')
model_2 = tf.keras.models.load_model('efficientnet_model.keras')
model_3 = tf.keras.models.load_model('densenet_model.keras')

# --- 3. Read Cell Coordinates ---
def load_coords(h5_path):
    if not os.path.exists(h5_path):
        return []
    with h5py.File(h5_path, 'r') as f:
        return np.asarray(f['coordinates'])

pos_coords = load_coords(POS_H5_PATH)
neg_coords = load_coords(NEG_H5_PATH)

# Combine them so we can test the AI blind
all_coords = []
for c in pos_coords:
    all_coords.append((c[0], c[1]))
for c in neg_coords:
    all_coords.append((c[0], c[1]))

total_cells = len(all_coords)
print(f"\nGround Truth: {len(pos_coords)} Positive, {len(neg_coords)} Negative ({total_cells} Total).")

# --- 4. Process the Full Image ---
print(f"Loading full image: {TEST_IMAGE_NAME}")
image = cv2.imread(IMG_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

patches = []
valid_coords = []

# Extract patches around every coordinate
half_size = PATCH_SIZE // 2
for x, y in all_coords:
    x, y = int(x), int(y)
    
    # Ensure patch doesn't go outside image boundaries
    x1, x2 = max(0, x - half_size), min(w, x + half_size)
    y1, y2 = max(0, y - half_size), min(h, y + half_size)
    
    patch = image_rgb[y1:y2, x1:x2]
    
    # Pad with white if the cell is touching the very edge of the image
    if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
        pad_y = PATCH_SIZE - patch.shape[0]
        pad_x = PATCH_SIZE - patch.shape[1]
        patch = cv2.copyMakeBorder(patch, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
    patches.append(patch)
    valid_coords.append((x, y))

patches_array = np.array(patches)

# --- 5. AI Prediction (The Ensemble Vote) ---
print("\nRunning AI Ensemble Inference on extracted cells...")
pred_1 = model_1.predict(patches_array, batch_size=32, verbose=0)
pred_2 = model_2.predict(patches_array, batch_size=32, verbose=0)
pred_3 = model_3.predict(patches_array, batch_size=32, verbose=0)

# Soft Voting
ensemble_probs = (pred_1 + pred_2 + pred_3) / 3.0

# Class 0 is Negative, Class 1 is Positive
ai_predictions = (ensemble_probs > 0.5).astype(int).flatten()

predicted_positives = np.sum(ai_predictions == 1)
predicted_negatives = np.sum(ai_predictions == 0)

# --- 6. Calculate Final Ki-67 Score ---
actual_ki67 = (len(pos_coords) / total_cells) * 100 if total_cells > 0 else 0
predicted_ki67 = (predicted_positives / total_cells) * 100 if total_cells > 0 else 0

# --- 7. Print the Final Report ---
print("\n--------------------------------")
print("          Ki-67 Analysis        ")
print("--------------------------------")
print(f"Total cells    : {total_cells}")
print(f"Positive cells : {predicted_positives}")
print(f"Negative cells : {predicted_negatives}")
print("--------------------------------")
print(f"Actual Ki-67   : {actual_ki67:.2f}%")
print(f"AI Ki-67 Score : {predicted_ki67:.2f}%")
print("--------------------------------\n")

# --- 8. Visualization ---
viz_image = image_rgb.copy()
for i, (x, y) in enumerate(valid_coords):
    if ai_predictions[i] == 1:
        # AI says Positive -> Draw Red Circle
        cv2.circle(viz_image, (x, y), radius=5, color=(255, 0, 0), thickness=2)
    else:
        # AI says Negative -> Draw Green Circle
        cv2.circle(viz_image, (x, y), radius=5, color=(0, 255, 0), thickness=2)

plt.figure(figsize=(10, 10))
plt.imshow(viz_image)
plt.title(f"AI Ki-67 Assessment: {predicted_ki67:.2f}%\nRed = Positive | Green = Negative")
plt.axis('off')
plt.tight_layout()
plt.show()