import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("chess_piece_detector.h5")

image = cv2.imread("path_to_chessboard_image.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Chessboard image not found. Check the path!")

grid_corners = {
    'x': [x0, x1, x2, ..., x8],  # x-coordinates of vertical grid lines
    'y': [y0, y1, y2, ..., y8]   # y-coordinates of horizontal grid lines
}

def calculate_grid_corners(grid_lines):
    """
    Extracts x and y coordinates of grid lines from detected lines.
    Returns a dictionary with 'x' and 'y' keys.
    """
    x_coords = sorted(set(line[0] for line in grid_lines['vertical']))
    y_coords = sorted(set(line[1] for line in grid_lines['horizontal']))
    return {'x': x_coords, 'y': y_coords}

# Map output classes to chess pieces
class_labels = {
    0: "0",  # Empty black square
    1: "1",  # Empty white square
    2: "p",  # Pawn
    3: "n",  # Knight
    4: "b",  # Bishop
    5: "r",  # Rook
    6: "q",  # Queen
    7: "k",  # King
    8: "P",  # White pawn
    9: "N",  # White knight
    10: "B",  # White bishop
    11: "R",  # White rook
    12: "Q",  # White queen
    13: "K",  # White king
}

# Function to classify a single square
def classify_square(square_image):
    square_image = cv2.resize(square_image, (64, 64))
    square_image = square_image.astype("float32") / 255.0  # Normalize
    square_image = np.expand_dims(square_image, axis=-1)  # Add channel dimension
    square_image = np.expand_dims(square_image, axis=0)   # Add batch dimension
    
    pred = model.predict(square_image)
    return class_labels[np.argmax(pred)]

# Generate the output matrix for the board
def generate_board_matrix(image, grid_corners, rows, cols):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            x1, x2 = grid_corners['x'][j], grid_corners['x'][j + 1]
            y1, y2 = grid_corners['y'][i], grid_corners['y'][i + 1]
            square = image[y1:y2, x1:x2]
            row.append(classify_square(square))
        matrix.append(row)
    return matrix

# Assume `image` is your input chessboard image
# and `grid_corners` contains the corner coordinates of the grid
# Example structure:
# grid_corners = {'x': [x0, x1, x2, ..., xn], 'y': [y0, y1, y2, ..., ym]}

output_matrix = generate_board_matrix(image, grid_corners, rows=8, cols=8)
print("Chessboard Matrix:")
for row in output_matrix:
    print(" ".join(row))
