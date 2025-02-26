import cv2
import numpy as np
import os

# Create the output folder
output_folder = "C:/Users/sohan/Desktop/sop 3-1/my-model/outputs"

# Load the image
image = cv2.imread("C:/Users/sohan/Desktop/sop 3-1/my-model/output2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptive threshold to detect the board clearly
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Detect the grid using contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Assume the largest contour corresponds to the board
board_contour = contours[0]
x, y, w, h = cv2.boundingRect(board_contour)

# Crop to the detected board
board = gray[y:y + h, x:x + w]

# Estimate grid size (rows and columns)
height, width = board.shape
horizontal_projections = np.sum(board, axis=1)
vertical_projections = np.sum(board, axis=0)

# Count grid lines based on projections
rows = len(np.where(horizontal_projections < 0.9 * np.max(horizontal_projections))[0])
cols = len(np.where(vertical_projections < 0.9 * np.max(vertical_projections))[0])

# Handle edge cases where rows or cols are miscounted
rows = max(1, min(rows, 8))
cols = max(1, min(cols, 8))

# Cell dimensions
cell_height = height // rows
cell_width = width // cols

# Matrix to store results
matrix = []

# Analyze each cell
for row in range(rows):
    matrix_row = []
    for col in range(cols):
        # Define the cell boundaries
        cell_x1 = col * cell_width
        cell_y1 = row * cell_height
        cell_x2 = cell_x1 + cell_width
        cell_y2 = cell_y1 + cell_height

        # Extract the cell ROI
        cell_roi = board[cell_y1:cell_y2, cell_x1:cell_x2]

        # Determine if the square is white or black
        avg_intensity = np.mean(cell_roi)
        square_value = 1 if avg_intensity < 128 else 0  # Black = 1, White = 0

        # Check for a piece in the center of the cell
        center_x1 = cell_x1 + cell_width // 4
        center_y1 = cell_y1 + cell_height // 4
        center_x2 = cell_x2 - cell_width // 4
        center_y2 = cell_y2 - cell_height // 4
        center_roi = board[center_y1:center_y2, center_x1:center_x2]

        # Threshold to detect pieces
        _, binary_center_roi = cv2.threshold(center_roi, 128, 255, cv2.THRESH_BINARY_INV)
        piece_pixels = cv2.countNonZero(binary_center_roi)

        # Adjust the threshold based on testing
        if piece_pixels > 50:  # Piece detected
            square_value = "x"
            # Save the piece image
            piece_image = image[cell_y1:cell_y2, cell_x1:cell_x2]
            output_filename = os.path.join(output_folder, f"({row},{col}).jpg")
            cv2.imwrite(output_filename, piece_image)

        matrix_row.append(square_value)
    matrix.append(matrix_row)

# Print the resulting matrix
print("Board Matrix:")
for row in matrix:
    print(row)

# Show the board with detected pieces (optional)
cv2.imshow("Detected Board", board)
cv2.waitKey(0)
cv2.destroyAllWindows()
