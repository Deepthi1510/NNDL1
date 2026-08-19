import os
import h5py

def count_h5_cells(folder_path):
    """Opens every .h5 file in a folder and counts the total number of coordinates."""
    total_cells = 0
    if not os.path.exists(folder_path):
        return 0
        
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder_path, filename)
            try:
                with h5py.File(filepath, 'r') as f:
                    total_cells += len(f['coordinates'])
            except Exception as e:
                pass
    return total_cells

def analyze_split(base_path, split_name):
    """Analyzes the images and annotations for a specific split (train, test, validation)."""
    img_dir = os.path.join(base_path, "images", split_name)
    pos_dir = os.path.join(base_path, "annotations", split_name, "positive")
    neg_dir = os.path.join(base_path, "annotations", split_name, "negative")
    
    # Count images
    num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]) if os.path.exists(img_dir) else 0
    
    # Count cells
    pos_cells = count_h5_cells(pos_dir)
    neg_cells = count_h5_cells(neg_dir)
    total_cells = pos_cells + neg_cells
    
    print(f"--- {split_name.upper()} SET ---")
    print(f"Total Images: {num_images}")
    print(f"Positive Cells (Brown): {pos_cells}")
    print(f"Negative Cells (Blue): {neg_cells}")
    print(f"Total Cells: {total_cells}\n")
    
    return num_images, pos_cells, neg_cells

print("=========================================")
print("          DATASET SUMMARY REPORT         ")
print("=========================================\n")

base_dataset_path = "." # Looks in the current directory

# Analyze all three splits
train_img, train_pos, train_neg = analyze_split(base_dataset_path, "train")
val_img, val_pos, val_neg = analyze_split(base_dataset_path, "validation")
test_img, test_pos, test_neg = analyze_split(base_dataset_path, "test")

# Print Grand Totals
print("=========================================")
print(f"GRAND TOTAL IMAGES: {train_img + val_img + test_img}")
print(f"GRAND TOTAL CELLS: {train_pos + train_neg + val_pos + val_neg + test_pos + test_neg}")
print("=========================================")