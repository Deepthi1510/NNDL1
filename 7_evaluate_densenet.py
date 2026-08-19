import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

print("=========================================")
print("     EVALUATING DENSENET ON TEST SET     ")
print("=========================================\n")

# 1. Configuration
TEST_DIR = "Dataset_Patches/test"
BATCH_SIZE = 32
IMG_SIZE = (64, 64)

# 2. Load the Test Data (Ensure shuffle=False so predictions match true labels)
print("Loading test data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = test_dataset.class_names

# 3. Load the Trained Model
print("\nLoading saved model 'densenet_model.keras'...")
model = tf.keras.models.load_model('densenet_model.keras')

# 4. Generate Predictions
print("Evaluating on test patches (this may take a couple of minutes)...")
y_pred_probs = model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Extract true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# 5. Print Metrics Report
print("\n=========================================")
print("          CLASSIFICATION REPORT          ")
print("=========================================")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)  # Purple color theme for DenseNet
plt.title('Confusion Matrix - DenseNet121 Test Set')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Annotate numbers in boxes
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label (Actual)')
plt.xlabel('Predicted Label (AI Guess)')
plt.tight_layout()
plt.show()