import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import h5py
import os

st.set_page_config(page_title="Ki-67 AI Scorer", layout="wide")
st.title("🔬 Automated Ki-67 Breast Cancer Assessment")
st.markdown("Select a test slide to evaluate the Grand Ensemble's diagnostic accuracy against pathologist ground-truth coordinates.")

# --- 1. Load Models ---
@st.cache_resource
def load_ensemble_models():
    m1 = tf.keras.models.load_model('resnet_model.keras')
    m2 = tf.keras.models.load_model('efficientnet_model.keras')
    m3 = tf.keras.models.load_model('densenet_model.keras')
    return m1, m2, m3

with st.spinner("Loading AI Ensemble (ResNet50, EfficientNetB0, DenseNet121)..."):
    model_1, model_2, model_3 = load_ensemble_models()

# --- 2. Helper to load .h5 coordinates ---
def load_coords(h5_path):
    if not os.path.exists(h5_path):
        return []
    with h5py.File(h5_path, 'r') as f:
        return np.asarray(f['coordinates'])

# --- 3. UI: Select Image ---
test_images_path = "images/test"
if not os.path.exists(test_images_path):
    st.error(f"Cannot find folder: {test_images_path}. Make sure you run this from the BCData folder.")
else:
    image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith('.png')])
    selected_image = st.selectbox("Select a Slide from the Test Dataset:", image_files)

    if st.button("Run AI Diagnostics"):
        with st.spinner(f"Analyzing {selected_image}..."):
            
            # Setup paths
            img_path = os.path.join(test_images_path, selected_image)
            pos_h5 = f"annotations/test/positive/{selected_image.replace('.png', '.h5')}"
            neg_h5 = f"annotations/test/negative/{selected_image.replace('.png', '.h5')}"
            
            # Load Image
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image_rgb.shape
            
            # Load Coordinates
            pos_coords = load_coords(pos_h5)
            neg_coords = load_coords(neg_h5)
            
            all_coords = []
            for c in pos_coords: all_coords.append((int(c[0]), int(c[1])))
            for c in neg_coords: all_coords.append((int(c[0]), int(c[1])))
            
            total_cells = len(all_coords)
            actual_ki67 = (len(pos_coords) / total_cells) * 100 if total_cells > 0 else 0
            
            if total_cells == 0:
                st.warning("No cells annotated in this slide.")
            else:
                # Extract Patches
                PATCH_SIZE = 64
                half_size = PATCH_SIZE // 2
                patches = []
                
                for x, y in all_coords:
                    x1, x2 = max(0, x - half_size), min(w, x + half_size)
                    y1, y2 = max(0, y - half_size), min(h, y + half_size)
                    patch = image_rgb[y1:y2, x1:x2]
                    
                    if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                        pad_y = PATCH_SIZE - patch.shape[0]
                        pad_x = PATCH_SIZE - patch.shape[1]
                        patch = cv2.copyMakeBorder(patch, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    patches.append(patch)
                
                # Predict
                patches_array = np.array(patches)
                pred_1 = model_1.predict(patches_array, batch_size=32, verbose=0)
                pred_2 = model_2.predict(patches_array, batch_size=32, verbose=0)
                pred_3 = model_3.predict(patches_array, batch_size=32, verbose=0)
                
                ensemble_probs = (pred_1 + pred_2 + pred_3) / 3.0
                ai_predictions = (ensemble_probs > 0.5).astype(int).flatten()
                
                predicted_positives = int(np.sum(ai_predictions == 1))
                predicted_negatives = int(np.sum(ai_predictions == 0))
                ai_ki67 = (predicted_positives / total_cells) * 100 if total_cells > 0 else 0
                
                # Draw Output
                viz_image = image_rgb.copy()
                for i, (x, y) in enumerate(all_coords):
                    color = (255, 0, 0) if ai_predictions[i] == 1 else (0, 255, 0)
                    cv2.circle(viz_image, (x, y), radius=5, color=color, thickness=2)
                
                st.success("Diagnostic Assessment Complete!")
                
                # Layout
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Cells", total_cells)
                col2.metric("AI Positives", predicted_positives)
                col3.metric("AI Negatives", predicted_negatives)
                col4.metric("Actual Ki-67", f"{actual_ki67:.2f}%")
                col5.metric("AI Ki-67", f"{ai_ki67:.2f}%")
                
                st.image(viz_image, caption="AI Result Overlaid (Red = Positive, Green = Negative)", width="stretch")