import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"Using TensorFlow version: {tf.__version__}")
print(f"Running on {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

# Load and preprocess image
def load_and_preprocess_image(img_path, img_height, img_width):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale (edges only)
    if img is None:
        img = np.zeros((img_height, img_width), dtype=np.uint8)
    img = cv2.resize(img, (img_height, img_width))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=(0, -1))  # Add batch & channel dimension

# Better logic for checking if two pieces are part of the same piece
def check_appending(base_img, neighbor_img, side):
    """Checks if two edge images belong to the same chess piece using histogram correlation."""
    if base_img is None or neighbor_img is None:
        return False

    if base_img.shape != neighbor_img.shape:
        min_size = max(base_img.shape[0], neighbor_img.shape[0])
        base_img = cv2.resize(base_img, (min_size, min_size))
        neighbor_img = cv2.resize(neighbor_img, (min_size, min_size))

    if side == "left":
        base_strip = base_img[:, :5]
        neighbor_strip = neighbor_img[:, -5:]
    elif side == "right":
        base_strip = base_img[:, -5:]
        neighbor_strip = neighbor_img[:, :5]
    elif side == "top":
        base_strip = base_img[:5, :]
        neighbor_strip = neighbor_img[-5:, :]
    else:  # "bottom"
        base_strip = base_img[-5:, :]
        neighbor_strip = neighbor_img[:5, :]

    # Compute histograms of pixel intensities in the strips
    hist_base = cv2.calcHist([base_strip], [0], None, [256], [0, 256])
    hist_neighbor = cv2.calcHist([neighbor_strip], [0], None, [256], [0, 256])

    # Normalize histograms
    hist_base = cv2.normalize(hist_base, hist_base).flatten()
    hist_neighbor = cv2.normalize(hist_neighbor, hist_neighbor).flatten()

    # Compute correlation between histograms
    similarity = cv2.compareHist(hist_base, hist_neighbor, cv2.HISTCMP_CORREL)

    return similarity > 0.7  # Higher threshold for better accuracy

# Predict piece type
def predict_piece(model, img_path, img_height, img_width):
    try:
        img_array = load_and_preprocess_image(img_path, img_height, img_width)
        predictions = model.predict(img_array)
        return predictions[0] / np.sum(predictions[0])  # Normalize softmax output
    except Exception as e:
        print(f"Error predicting piece for {img_path}: {e}")
        return np.zeros(len(class_labels))

# Load trained model
model_path = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/chess_piece_detector.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path, compile=False)

# Define chessboard and class labels
img_height, img_width = 64, 64
output_folder = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/outputs5"
class_labels = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
matrix = {}

# Process images
image_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".jpg")])

for filename in image_files:
    try:
        row_col = filename[1:-4].rstrip(')')
        row, col = map(int, row_col.split(','))
        img_path = os.path.join(output_folder, filename)

        base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale (edges only)
        if base_img is None:
            continue

        weighted_predictions = np.zeros(len(class_labels))
        total_weight = 1.0  # Direct prediction has highest weight

        # Direct piece prediction
        direct_pred = predict_piece(model, img_path, img_height, img_width)
        weighted_predictions += direct_pred * 0.8  # Give more importance to direct prediction

        # Neighbor-based predictions
        neighbors = {
            "left": (row, col - 1),
            "right": (row, col + 1),
            "top": (row - 1, col),
            "bottom": (row + 1, col)
        }

        for side, (n_row, n_col) in neighbors.items():
            neighbor_filename = f"({n_row},{n_col}).jpg"
            neighbor_path = os.path.join(output_folder, neighbor_filename)

            if os.path.exists(neighbor_path):
                neighbor_img = cv2.imread(neighbor_path, cv2.IMREAD_GRAYSCALE)
                if neighbor_img is None:
                    continue

                if check_appending(base_img, neighbor_img, side):
                    weight = 0.5  # Moderate weight for neighbors
                else:
                    weight = 0.1  # Small weight for unconnected neighbors

                pred = predict_piece(model, neighbor_path, img_height, img_width)
                weighted_predictions += weight * pred
                total_weight += weight

        # Normalize weighted predictions
        if total_weight > 0:
            weighted_predictions /= total_weight

        # Assign predicted piece
        predicted_class = np.argmax(weighted_predictions)
        matrix[(row, col)] = class_labels[predicted_class]

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save predicted board matrix
output_file = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/pieces.txt"
with open(output_file, 'w') as f:
    for i in range(8):
        row_data = [matrix.get((i, j), '--') for j in range(8)]
        print(row_data) 
        f.write(' '.join(row_data) + '\n')
