import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
def build_cnn(input_shape=(64, 64, 1), num_classes=13):
    """
    Build a CNN for chess piece classification.
    Args:
    - input_shape: Tuple, dimensions of input image.
    - num_classes: Number of output classes (0, 1, p, n, b, r, q, k).
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Build the CNN
input_shape = (64, 64, 1)  # Grayscale images of size 64x64
num_classes = 13  # 12 pieces + 1 for empty squares
cnn_model = build_cnn(input_shape=input_shape, num_classes=num_classes)
cnn_model.summary()
