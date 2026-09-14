import os
import matplotlib.pyplot as plt
import tensorflow as tf

os.makedirs("models", exist_ok=True)

# 1. Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
INITIAL_EPOCHS = 6
FINE_TUNE_EPOCHS = 4
DATASET_PATH = "dataset"
TRANSFER_MODEL_PATH = "models/mobilenetv2_transfer.keras"
FINAL_MODEL_PATH = "models/mobilenetv2_final.keras"

# 2. Data Loading & Pipeline Optimization
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Model Architecture with Pretrained Base
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3),
)
base_model.trainable = False  # Freeze ImageNet feature extractors

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        # Crucial: MobileNetV2 expects input scaled to [-1, 1]
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="MobileNetV2_Classifier",
)

# 4. Phase 1: Feature Extraction Training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

checkpoint_phase1 = tf.keras.callbacks.ModelCheckpoint(
    TRANSFER_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

print("\n================ PHASE 1: FROZEN FEATURE EXTRACTION ================\n")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[checkpoint_phase1],
)

# 5. Phase 2: Unfreeze Top 20 Layers for Fine-Tuning
print("\n================ PHASE 2: FINE-TUNING TOP 20 LAYERS ================\n")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with very low learning rate to avoid destroying weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

checkpoint_phase2 = tf.keras.callbacks.ModelCheckpoint(
    FINAL_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_phase2],
)

print(f"\nTraining Complete! Best model saved to: {FINAL_MODEL_PATH}")