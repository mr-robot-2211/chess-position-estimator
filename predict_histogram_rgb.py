import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"Using TensorFlow version: {tf.__version__}")
print(f"Running on {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

def load_and_preprocess_image(img_path, img_height, img_width):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Ensure RGB
    if img is None:
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img = cv2.resize(img, (img_width, img_height))
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def check_appending(base_img, neighbor_img, side):
    if base_img is None or neighbor_img is None:
        return False

    if base_img.shape != neighbor_img.shape:
        min_size = max(base_img.shape[0], neighbor_img.shape[0])
        base_img = cv2.resize(base_img, (min_size, min_size))
        neighbor_img = cv2.resize(neighbor_img, (min_size, min_size))

    if side == "left":
        base_strip, neighbor_strip = base_img[:, :5], neighbor_img[:, -5:]
    elif side == "right":
        base_strip, neighbor_strip = base_img[:, -5:], neighbor_img[:, :5]
    elif side == "top":
        base_strip, neighbor_strip = base_img[:5, :], neighbor_img[-5:, :]
    else:  # "bottom"
        base_strip, neighbor_strip = base_img[-5:, :], neighbor_img[:5, :]

    hist_base = cv2.calcHist([base_strip], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_neighbor = cv2.calcHist([neighbor_strip], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_base = cv2.normalize(hist_base, hist_base).flatten()
    hist_neighbor = cv2.normalize(hist_neighbor, hist_neighbor).flatten()

    similarity = cv2.compareHist(hist_base, hist_neighbor, cv2.HISTCMP_CORREL)
    print(f"{side} similarity score: {similarity}")  # Debug print
    return similarity > 0.5  # Lowered threshold

def predict_piece(model, img_path, img_height, img_width):
    try:
        img_array = load_and_preprocess_image(img_path, img_height, img_width)
        predictions = model.predict(img_array)
        normalized_predictions = predictions[0] / np.sum(predictions[0])
        return normalized_predictions, np.max(normalized_predictions)
    except Exception as e:
        print(f"Error predicting piece for {img_path}: {e}")
        return np.zeros(len(class_labels)), 0

model_path = "chess_piece_detector.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path, compile=False)

img_height, img_width = 64, 64
output_folder = "outputs5"
class_labels = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
matrix = {}

image_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".jpg")])

for filename in image_files:
    try:
        row_col = filename[1:-4].rstrip(')')
        row, col = map(int, row_col.split(','))
        img_path = os.path.join(output_folder, filename)

        base_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base_img is None:
            continue

        weighted_predictions = np.zeros(len(class_labels))
        total_weight = 1.0

        direct_pred, confidence = predict_piece(model, img_path, img_height, img_width)
        weighted_predictions += direct_pred * 0.95  # Increased direct prediction weight

        if confidence <= 0.7:  # Only consider neighbors if direct prediction is not confident
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
                    neighbor_img = cv2.imread(neighbor_path, cv2.IMREAD_COLOR)
                    if neighbor_img is None:
                        continue

                    weight = 0.1 if check_appending(base_img, neighbor_img, side) else 0.01
                    pred, _ = predict_piece(model, neighbor_path, img_height, img_width)
                    weighted_predictions += weight * pred
                    total_weight += weight

        if total_weight > 0:
            weighted_predictions /= total_weight

        predicted_class = np.argmax(weighted_predictions)
        if weighted_predictions[predicted_class] > 0.5:  # Confidence threshold
            matrix[(row, col)] = class_labels[predicted_class]
        else:
            matrix[(row, col)] = '--'  # Mark as empty if not confident

    except Exception as e:
        print(f"Error processing {filename}: {e}")

output_file = "pieces.txt"
with open(output_file, 'w') as f:
    for i in range(8):
        row_data = [matrix.get((i, j), '--') for j in range(8)]
        print(row_data)
        f.write(' '.join(row_data) + '\n')


