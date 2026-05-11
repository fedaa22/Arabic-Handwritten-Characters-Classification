import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow warnings

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import project utility functions
from utils.load_data import load_dataset
from utils.preprocessing import split_data, create_data_generators
from utils.transfer_learning import build_transfer_learning_model, unfreeze_for_finetuning

# Create folders to save models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("=" * 60)
print("STEP 1: Loading Arabic Handwritten Characters Dataset")
print("=" * 60)

# Load images, labels, and class mapping
X, y, label_map = load_dataset("Dataset")

print(f"Loaded {X.shape[0]} images across {len(label_map)} classes")

# ============================================================================
# STEP 2: Convert Grayscale to RGB
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Converting Images to 3-Channel RGB")
print("=" * 60)

print("Why? MobileNetV2 was trained on RGB images")
print("     We replicate the grayscale channel 3 times")

# Convert images from 1 channel to 3 channels
X_rgb = np.repeat(X, 3, axis=-1)

print(f"Before conversion: {X.shape}")
print(f"After conversion:  {X_rgb.shape}")

# ============================================================================
# STEP 3: Split Data into Train/Validation/Test
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Splitting Dataset")
print("=" * 60)

# Split dataset into training, validation, and testing sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_rgb, y)

# ============================================================================
# STEP 4: Create Data Generators
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Creating Data Generators")
print("=" * 60)

# Create generators with preprocessing and augmentation
train_gen, val_gen, test_gen = create_data_generators(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

# ============================================================================
# STEP 5: Build Transfer Learning Model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building Transfer Learning Architecture")
print("=" * 60)

# Define input shape and number of output classes
input_shape = (64, 64, 3)
num_classes = len(label_map)

# Build MobileNetV2 transfer learning model
model, base_model = build_transfer_learning_model(input_shape, num_classes)

# Display model architecture
print("\nModel Architecture Summary:")
model.summary()

# ============================================================================
# STEP 6: Phase 1 Training (Only Custom Head)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: PHASE 1 - Training Custom Classification Head")
print("=" * 60)

print("Only the custom Dense layers are trainable")
print("The MobileNetV2 base remains frozen\n")

# Callbacks used during Phase 1 training
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/transfer_learning_phase1_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("Starting Phase 1 training...")

# Train only the classification head
history_phase1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks_phase1,
    verbose=1
)

# Store best validation accuracy from Phase 1
phase1_best_acc = max(history_phase1.history['val_accuracy'])

print(f"\nPhase 1 Complete! Best validation accuracy: {phase1_best_acc:.4f}")

# ============================================================================
# STEP 7: Phase 2 Training (Fine-Tuning)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: PHASE 2 - Fine-Tuning Last Layers")
print("=" * 60)

print("Unfreezing part of MobileNetV2 for fine-tuning")
print("Using a smaller learning rate for stable training\n")

# Unfreeze part of the base model
model = unfreeze_for_finetuning(model, base_model)

# Callbacks used during fine-tuning
callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/transfer_learning_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("Starting Phase 2 fine-tuning...")

# Fine-tune the last layers of the model
history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks_phase2,
    verbose=1
)

# ============================================================================
# STEP 8: Combine Training Histories
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: Combining Training History")
print("=" * 60)

# Merge training history from both phases
combined_history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# ============================================================================
# STEP 9: Evaluate on Test Set
# ============================================================================
print("\n" + "=" * 60)
print("STEP 9: Final Test Set Evaluation")
print("=" * 60)

# Evaluate model performance on unseen test data
test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 10: Visualize Training Progress
# ============================================================================
print("\n" + "=" * 60)
print("STEP 10: Creating Training Visualization")
print("=" * 60)

# Create plots for accuracy and loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Determine where fine-tuning starts
phase1_epochs = len(history_phase1.history['accuracy'])

# Plot training and validation accuracy
ax1.plot(combined_history['accuracy'], label='Train Accuracy', marker='o', markersize=3, linewidth=2)
ax1.plot(combined_history['val_accuracy'], label='Validation Accuracy', marker='s', markersize=3, linewidth=2)
ax1.axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
ax1.set_title('Transfer Learning (MobileNetV2) - Accuracy', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot training and validation loss
ax2.plot(combined_history['loss'], label='Train Loss', marker='o', markersize=3, linewidth=2)
ax2.plot(combined_history['val_loss'], label='Validation Loss', marker='s', markersize=3, linewidth=2)
ax2.axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
ax2.set_title('Transfer Learning (MobileNetV2) - Loss', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save training plots
plt.savefig('results/transfer_learning_training_history.png', dpi=150)

print("-> Training history plot saved to: results/transfer_learning_training_history.png")

plt.close()

# ============================================================================
# STEP 11: Save Model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 11: Saving Final Model")
print("=" * 60)

# Save the final trained model
model.save('models/transfer_learning_final.keras')

print("-> Final model saved to: models/transfer_learning_final.keras")

# ============================================================================
# STEP 12: Training Summary
# ============================================================================

# Calculate final training statistics
total_epochs = len(combined_history['accuracy'])
best_val_acc = max(combined_history['val_accuracy'])
best_epoch = combined_history['val_accuracy'].index(best_val_acc) + 1
best_train_acc = combined_history['accuracy'][best_epoch - 1]

print("\n" + "=" * 60)
print("TRANSFER LEARNING TRAINING COMPLETE!")
print("=" * 60)

print(f"Total epochs trained: {total_epochs}")
print(f"  - Phase 1 (head only): {phase1_epochs} epochs")
print(f"  - Phase 2 (fine-tuning): {total_epochs - phase1_epochs} epochs")

print(f"\nBest Performance:")
print(f"  - Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
print(f"  - Training Accuracy: {best_train_acc:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f}")

print(f"\nFiles Saved:")
print(f"  - Best model: models/transfer_learning_best.keras")
print(f"  - Final model: models/transfer_learning_final.keras")
print(f"  - Training plot: results/transfer_learning_training_history.png")