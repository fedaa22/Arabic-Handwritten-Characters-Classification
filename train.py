import os

# Hide TensorFlow warnings and limit CPU threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

# Import required libraries and utility functions
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.load_data import load_dataset
from utils.preprocessing import split_data, create_data_generators
from utils.model_utils import build_scratch_cnn

# Create folders to save models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# STEP 1: Load Dataset
# =========================
print("=" * 50)
print("STEP 1: Loading dataset")
print("=" * 50)

# Load images, labels, and class mapping
X, y, label_map = load_dataset("Dataset")

# =========================
# STEP 1.5: Show Sample Images
# =========================
print("\n" + "=" * 50)
print("STEP 1.5: Checking sample images")
print("=" * 50)

# Display random sample images for debugging
fig, axes = plt.subplots(3, 5, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    if i < len(X):
        ax.imshow(X[i].reshape(64, 64), cmap='gray')

        # Get class name from label map
        class_name = list(label_map.keys())[list(label_map.values()).index(y[i])]

        ax.set_title(f'{class_name}', fontsize=8)
        ax.axis('off')

# Save sample visualization
plt.tight_layout()
plt.savefig('results/sample_images.png')

print("-> Sample images saved to results/sample_images.png")
plt.close()

# =========================
# STEP 2: Split Dataset
# =========================
print("\n" + "=" * 50)
print("STEP 2: Splitting data")
print("=" * 50)

# Split dataset into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# =========================
# STEP 3: Create Data Generators
# =========================
print("\n" + "=" * 50)
print("STEP 3: Creating data generators")
print("=" * 50)

# Create generators for model training
train_gen, val_gen, test_gen = create_data_generators(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

# =========================
# STEP 4: Build CNN Model
# =========================
print("\n" + "=" * 50)
print("STEP 4: Building Scratch CNN")
print("=" * 50)

# Define input shape and number of classes
input_shape = (64, 64, 1)
num_classes = len(label_map)

# Build CNN architecture
model = build_scratch_cnn(input_shape, num_classes)

# Show model architecture summary
model.summary()

# =========================
# Training Callbacks
# =========================

# Configure callbacks to improve training
callbacks = [

    # Reduce learning rate if validation loss stops improving
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    # Stop training early to avoid overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    # Save the best model during training
    ModelCheckpoint(
        'models/scratch_cnn_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =========================
# STEP 5: Train Model
# =========================
print("\n" + "=" * 50)
print("STEP 5: Training the model")
print("=" * 50)

# Print training settings
print("Training settings:")
print(f"-> Epochs: 15")
print(f"-> Batch size: 32")
print(f"-> Learning rate: 0.0005 (will reduce automatically if needed)")
print(f"-> Early stopping patience: 10 epochs")

# Start training process
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =========================
# STEP 6: Evaluate Model
# =========================
print("\n" + "=" * 50)
print("STEP 6: Evaluating on Test Set")
print("=" * 50)

# Evaluate trained model on test data
test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# =========================
# STEP 7: Plot Training Results
# =========================
print("\n" + "=" * 50)
print("STEP 7: Saving training history plots")
print("=" * 50)

# Create figure for accuracy and loss plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training and validation accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)

ax1.set_title('Model Accuracy', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)

ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot training and validation loss
ax2.plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)

ax2.set_title('Model Loss', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)

ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Save training history figure
plt.tight_layout()
plt.savefig('results/scratch_cnn_training_history.png', dpi=150)

print("-> Training history plot saved to results/scratch_cnn_training_history.png")

plt.close()

# =========================
# Save Final Model
# =========================

# Save the final trained model
model.save('models/scratch_cnn_final.keras')

print("-> Final model saved to models/scratch_cnn_final.keras")

# =========================
# Training Summary
# =========================

# Get best training results
best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
best_val_acc = max(history.history['val_accuracy'])
best_train_acc = max(history.history['accuracy'])

print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)

print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
print(f"Best Training Accuracy: {best_train_acc:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Total Epochs Trained: {len(history.history['accuracy'])}")

print("\nSaved files:")
print(f"-> Best model: models/scratch_cnn_best.keras")
print(f"-> Final model: models/scratch_cnn_final.keras")
print(f"-> Training plot: results/scratch_cnn_training_history.png")
print(f"-> Sample images: results/sample_images.png")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)