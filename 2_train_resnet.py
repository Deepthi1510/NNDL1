import tensorflow as tf
import matplotlib.pyplot as plt
import os

print("=========================================")
print("      TRAINING RESNET50 NEURAL NETWORK   ")
print("=========================================\n")

# --- 1. Configuration ---
TRAIN_DIR = "Dataset_Patches/train"
VAL_DIR = "Dataset_Patches/validation"
BATCH_SIZE = 32     # How many images the AI looks at before updating its math
IMG_SIZE = (64, 64) # The size of our patches
EPOCHS = 5          # How many times the AI will review the entire textbook (dataset)

# --- 2. Load the Data ---
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

print("\nLoading validation data...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# The folders are alphabetical: 'negative' is 0, 'positive' is 1.
class_names = train_dataset.class_names
print(f"\nClasses detected: {class_names} (0 = negative, 1 = positive)")

# Optimize data loading for speed
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 3. Build the Neural Network ---
print("\nBuilding the ResNet50 Model...")

# Load the pre-trained base model
base_model = tf.keras.applications.ResNet50(
    input_shape=(64, 64, 3),
    include_top=False, # We remove the original classification layer
    weights='imagenet' # Use pre-trained weights
)

# Freeze the base model so we don't destroy its pre-learned features
base_model.trainable = False

# Add our custom classification head
model = tf.keras.Sequential([
    # Standardize pixel values from 0-255 to -1 to 1 for ResNet
    tf.keras.layers.Rescaling(1./127.5, offset=-1), 
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Prevents overfitting by randomly dropping connections
    tf.keras.layers.Dense(1, activation='sigmoid') # Final output: 0 (Negative) or 1 (Positive)
])

# Compile the model (Giving it a learning strategy)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# --- 4. Train the Model ---
print("\nStarting Training! Grab a coffee, this might take a few minutes...\n")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --- 5. Save the Model ---
model.save('resnet_model.keras')
print("\n✅ Model successfully saved as 'resnet_model.keras'")

# --- 6. Visualize the Results ---
def plot_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('Model Loss (Error)')
    plt.legend()

    plt.show()

plot_history(history)