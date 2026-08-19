import os
import cv2
import h5py
import numpy as np

# --- Configuration ---
PATCH_SIZE = 64  # We will cut out 64x64 pixel squares
HALF_PATCH = PATCH_SIZE // 2
BASE_DIR = "." # Current directory
OUTPUT_DIR = "Dataset_Patches"

def create_dirs():
    """Creates the output directory structure."""
    splits = ['train', 'validation', 'test']
    classes = ['positive', 'negative']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def get_coordinates(h5_path):
    """Safely extracts coordinates from an .h5 file."""
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                return np.asarray(f['coordinates'])
        except:
            pass
    return []

def extract_patches_from_image(image_path, pos_h5, neg_h5, split, filename_base):
    """Crops patches around coordinates and saves them."""
    if not os.path.exists(image_path):
        return

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return
    
    img_h, img_w = img.shape[:2]
    
    # Process both positive and negative classes
    classes = {'positive': pos_h5, 'negative': neg_h5}
    
    for cls_name, h5_file in classes.items():
        coords = get_coordinates(h5_file)
        
        for idx, (x, y) in enumerate(coords):
            x, y = int(x), int(y)
            
            # Calculate the bounding box for the crop
            x_min = x - HALF_PATCH
            x_max = x + HALF_PATCH
            y_min = y - HALF_PATCH
            y_max = y + HALF_PATCH
            
            # Boundary check: Ensure the patch doesn't go outside the image edges
            if x_min >= 0 and y_min >= 0 and x_max <= img_w and y_max <= img_h:
                
                # Crop the image (Numpy arrays are sliced [y_start:y_end, x_start:x_end])
                patch = img[y_min:y_max, x_min:x_max]
                
                # Save the patch
                patch_filename = f"{filename_base}_{cls_name}_{idx}.png"
                save_path = os.path.join(OUTPUT_DIR, split, cls_name, patch_filename)
                cv2.imwrite(save_path, patch)

def process_dataset():
    """Main loop to process the entire dataset."""
    splits = ['train', 'validation', 'test']
    
    create_dirs()
    print("Starting Patch Extraction... This might take a few minutes!")
    
    for split in splits:
        img_dir = os.path.join(BASE_DIR, "images", split)
        if not os.path.exists(img_dir):
            continue
            
        images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        print(f"\nProcessing {split.upper()} set ({len(images)} images)...")
        
        for i, img_name in enumerate(images):
            # Print progress every 50 images
            if (i + 1) % 50 == 0:
                print(f"  -> Processed {i + 1}/{len(images)} images")
                
            base_name = os.path.splitext(img_name)[0]
            
            # Construct full paths
            img_path = os.path.join(img_dir, img_name)
            pos_h5 = os.path.join(BASE_DIR, "annotations", split, "positive", f"{base_name}.h5")
            neg_h5 = os.path.join(BASE_DIR, "annotations", split, "negative", f"{base_name}.h5")
            
            extract_patches_from_image(img_path, pos_h5, neg_h5, split, base_name)

    print("\n✅ Extraction Complete! Check the 'Dataset_Patches' folder.")

if __name__ == "__main__":
    process_dataset()