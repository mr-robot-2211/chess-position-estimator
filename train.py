import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set the path to your dataset
dataset_path = '/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/dataset'  # Replace with the actual path to your dataset

# Step 1: Set up ImageDataGenerators for loading and augmenting the dataset

# Training data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale images to [0, 1] range
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shearing
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill in missing pixels after transformation
)

# Validation data generator (without augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Set the batch size (adjust based on your hardware)
batch_size = 32

# Load and preprocess the training and validation data
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),  # Path to training data folder
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=batch_size,
    class_mode='categorical',  # Because it's a multi-class classification problem
    shuffle=True  # Shuffle the dataset
)

validation_generator = valid_datagen.flow_from_directory(
    os.path.join(dataset_path, 'valid'),  # Path to validation data folder
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Step 2: Build the CNN model for piece detection

model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and add fully connected layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes (pieces)
])

# Compile the model
model.compile(
    optimizer='adam',  # Optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Evaluation metric
)

# Display model summary
model.summary()

# Step 3: Train the model

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Number of batches per epoch
    epochs=20,  # Number of epochs (adjust as needed)
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # Number of validation batches
)

# Step 4: Evaluate the model on the validation set

validation_dir = "C:/Users/sohan/Desktop/sop 3-1/my-model/dataset/valid"

# Check if validation data exists
if len(os.listdir(validation_dir)) > 0:
    # Proceed with validation
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"Validation Accuracy: {val_acc:.4f}")
else:
    print("No validation data found, skipping evaluation.")

# Step 5: Save the trained model
model.save('chess_piece_detector.h5')

print("Model training complete and saved as 'chess_piece_detector.h5'.")
