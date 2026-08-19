import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

print("=========================================")
print("      EVALUATING THE GRAND ENSEMBLE      ")
print("=========================================\n")

# 1. Configuration
TEST_DIR = "Dataset_Patches/test"
BATCH_SIZE = 32
IMG_SIZE = (64, 64)

# 2. Load the Test Data
print("Loading test data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)
class_names = test_dataset.class_names

# 3. Load All Three Models
print("\nLoading Model 1: ResNet50...")
model_1 = tf.keras.models.load_model('resnet_model.keras')

print("Loading Model 2: EfficientNetB0...")
model_2 = tf.keras.models.load_model('efficientnet_model.keras')

print("Loading Model 3: DenseNet121...")
model_3 = tf.keras.models.load_model('densenet_model.keras')

# 4. Generate Predictions for Each Model
print("\nGathering votes from the team (this will take a few minutes)...")
pred_1 = model_1.predict(test_dataset)
pred_2 = model_2.predict(test_dataset)
pred_3 = model_3.predict(test_dataset)

# 5. Soft Voting (Averaging the probabilities)
ensemble_pred_probs = (pred_1 + pred_2 + pred_3) / 3.0
y_pred = (ensemble_pred_probs > 0.5).astype(int).flatten()

# Extract true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# 6. Print Final Metrics Report
print("\n=========================================")
print("       ENSEMBLE CLASSIFICATION REPORT    ")
print("=========================================")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 7. Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Grand Ensemble')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

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