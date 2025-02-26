import cv2
import numpy as np
import os

# Create the output folder
output_folder = "C:/Users/sohan/Desktop/sop 3-1/my-model/outputs100"
os.makedirs(output_folder, exist_ok=True)

# Load the image
image = cv2.imread("C:/Users/sohan/Desktop/sop 3-1/my-model/output2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Use Canny edge detection
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# Use Hough Line Transformation to detect grid lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=5)

# Extract horizontal and vertical lines separately
horizontal_lines = []
vertical_lines = []

for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y2 - y1) < 10:  # Horizontal line
            horizontal_lines.append((y1, y2))
        elif abs(x2 - x1) < 10:  # Vertical line
            vertical_lines.append((x1, x2))

# Sort the lines and remove duplicates
horizontal_lines = sorted(set([line[0] for line in horizontal_lines]))
vertical_lines = sorted(set([line[0] for line in vertical_lines]))

# Detect rows and columns
rows = len(horizontal_lines) - 1
cols = len(vertical_lines) - 1

# If no grid lines were detected, assume it's 1x1
if rows == 0 or cols == 0:
    rows, cols = 1, 1

# Compute square height and width
square_height = gray.shape[0] // rows
square_width = gray.shape[1] // cols

# Matrix to store results
matrix = []

# Analyze each cell
for row in range(rows):
    matrix_row = []
    for col in range(cols):
        # Define the square boundaries
        cell_x1 = vertical_lines[col] if col < len(vertical_lines) else col * square_width
        cell_x2 = vertical_lines[col + 1] if col + 1 < len(vertical_lines) else (col + 1) * square_width
        cell_y1 = horizontal_lines[row] if row < len(horizontal_lines) else row * square_height
        cell_y2 = horizontal_lines[row + 1] if row + 1 < len(horizontal_lines) else (row + 1) * square_height

        # Extract the cell ROI
        cell_roi = gray[cell_y1:cell_y2, cell_x1:cell_x2]

        # Determine if the square is white or black
        avg_intensity = np.mean(cell_roi)
        square_value = 1 if avg_intensity < 128 else 0  # Black = 1, White = 0

        # Check for a piece using contours
        _, binary_cell = cv2.threshold(cell_roi, 128, 255, cv2.THRESH_BINARY_INV)
        piece_contours, _ = cv2.findContours(binary_cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if piece_contours:
            # If a piece is detected, check its center of mass
            largest_piece = max(piece_contours, key=cv2.contourArea)
            M = cv2.moments(largest_piece)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Ensure the center is within the square
                if 0 <= cx < (cell_x2 - cell_x1) and 0 <= cy < (cell_y2 - cell_y1):
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

# Show the board with detected squares and pieces (optional)
for row in range(rows):
    for col in range(cols):
        cell_x1 = vertical_lines[col] if col < len(vertical_lines) else col * square_width
        cell_x2 = vertical_lines[col + 1] if col + 1 < len(vertical_lines) else (col + 1) * square_width
        cell_y1 = horizontal_lines[row] if row < len(horizontal_lines) else row * square_height
        cell_y2 = horizontal_lines[row + 1] if row + 1 < len(horizontal_lines) else (row + 1) * square_height
        cv2.rectangle(image, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 255, 0), 1)

cv2.imshow("Detected Board", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
