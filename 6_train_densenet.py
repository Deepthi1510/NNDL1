import tensorflow as tf
import matplotlib.pyplot as plt
import os

print("=========================================")
print("    TRAINING DENSENET121 NEURAL NETWORK  ")
print("=========================================\n")

# --- 1. Configuration ---
TRAIN_DIR = "Dataset_Patches/train"
VAL_DIR = "Dataset_Patches/validation"
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS = 5

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

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 3. Build the Neural Network ---
print("\nBuilding the DenseNet121 Model...")

base_model = tf.keras.applications.DenseNet121(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    # DenseNet prefers pixels between 0 and 1. 
    # This takes the 0-255 pixels and divides them by 255.
    tf.keras.layers.Rescaling(1./255),
    
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# --- 4. Train the Model ---
print("\nStarting Training! Let's complete the trio...\n")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --- 5. Save the Model ---
model.save('densenet_model.keras')
print("\n✅ Model successfully saved as 'densenet_model.keras'")

# --- 6. Visualize the Results ---
def plot_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='purple')
    plt.plot(val_acc, label='Validation Accuracy', color='orange')
    plt.title('DenseNet121 Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='purple')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('DenseNet121 Loss')
    plt.legend()

    plt.show()

plot_history(history)