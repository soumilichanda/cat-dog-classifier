import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "dataset"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

BASELINE_MODEL_PATH = "models/baseline_cnn.keras"
MOBILENET_MODEL_PATH = "models/mobilenetv2_final.keras"

# Must use shuffle=True with seed=42 to match the training split
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class_names = val_ds.class_names
print("Classes:", class_names)

# Load Models
print("\nLoading models...")
baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)

# Collect ground truths and predictions batch by batch to preserve alignment
print("Evaluating across validation batches...")
y_true_list = []
baseline_preds_list = []
mobilenet_preds_list = []

for images, labels in val_ds:
    y_true_list.extend(labels.numpy())

    b_probs = baseline_model.predict(images, verbose=0)
    baseline_preds_list.extend((b_probs >= 0.5).astype(int).flatten())

    m_probs = mobilenet_model.predict(images, verbose=0)
    mobilenet_preds_list.extend((m_probs >= 0.5).astype(int).flatten())

y_true = np.array(y_true_list)
baseline_preds = np.array(baseline_preds_list)
mobilenet_preds = np.array(mobilenet_preds_list)

# Print Reports
print("\n" + "=" * 60)
print("             BASELINE CUSTOM CNN REPORT")
print("=" * 60)
print(classification_report(y_true, baseline_preds, target_names=class_names))

print("=" * 60)
print("             MOBILENETV2 (TRANSFER LEARNING) REPORT")
print("=" * 60)
print(classification_report(y_true, mobilenet_preds, target_names=class_names))

# Plot Confusion Matrices
cm_baseline = confusion_matrix(y_true, baseline_preds)
cm_mobilenet = confusion_matrix(y_true, mobilenet_preds)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(
    cm_baseline,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[0],
    cbar=False,
    annot_kws={"size": 13},
)
axes[0].set_title("Baseline Custom CNN\nConfusion Matrix", fontsize=13, pad=10)
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(
    cm_mobilenet,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1],
    cbar=False,
    annot_kws={"size": 13},
)
axes[1].set_title("MobileNetV2 (Transfer Learning)\nConfusion Matrix", fontsize=13, pad=10)
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("models/confusion_matrix_comparison.png", dpi=300)
print("\nSaved Confusion Matrix to models/confusion_matrix_comparison.png")
plt.show()