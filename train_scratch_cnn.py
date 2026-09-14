import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# 1. Config
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "models/baseline_cnn.keras"

# 2. Dataset Ingestion
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

# 3. Pipeline Throughput Optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Model Architecture (Custom CNN from Scratch)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# 5. Compile & Checkpoint Setup
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1
)

# 6. Train
print("\n--- Training Custom CNN ---")
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb]
)

# 7. Plot Curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.tight_layout()
plt.show()