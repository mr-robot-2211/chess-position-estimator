import cv2
import numpy as np
import os

# Create the output folder
output_folder = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/outputs4"
os.makedirs(output_folder, exist_ok=True)

# Load the image
image = cv2.imread("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/output2.jpg")
image2= cv2.imread("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/experimented int outputs/output4.jpg")
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Use Canny edge detection
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# Use Hough Line Transformation to detect grid lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Separate and process horizontal and vertical lines
horizontal_lines = []
vertical_lines = []

for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y2 - y1) < 10:  # Horizontal line
            horizontal_lines.append((y1, y2))
        elif abs(x2 - x1) < 10:  # Vertical line
            vertical_lines.append((x1, x2))

# Sort and filter redundant lines
def filter_lines(lines, threshold=10):
    filtered_lines = []
    for i in sorted(lines):
        if not filtered_lines or abs(i - filtered_lines[-1]) > threshold:
            filtered_lines.append(i)
    return filtered_lines

horizontal_lines = filter_lines([line[0] for line in horizontal_lines])
vertical_lines = filter_lines([line[0] for line in vertical_lines])

# If no lines detected, assume single square (1x1 board)
if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
    horizontal_lines = [0, gray.shape[0]]
    vertical_lines = [0, gray.shape[1]]

# Ensure grid covers the image boundaries
horizontal_lines = [0] + [line for line in horizontal_lines if line < gray.shape[0]] + [gray.shape[0]]
vertical_lines = [0] + [line for line in vertical_lines if line < gray.shape[1]] + [gray.shape[1]]

rows = len(horizontal_lines) - 1
cols = len(vertical_lines) - 1

# Matrix to store results
squares = []

# Analyze each cell
for row in range(rows):
    for col in range(cols):
        # Define the square boundaries
        cell_x1 = vertical_lines[col]
        cell_x2 = vertical_lines[col + 1]
        cell_y1 = horizontal_lines[row]
        cell_y2 = horizontal_lines[row + 1]
        squares.append((cell_x1,cell_x2,cell_y1,cell_y2))
        # Validate square dimensions
        if (cell_x2 - cell_x1) < 10 or (cell_y2 - cell_y1) < 10:
            continue

# Loop through each square
for i, (x1, y1, x2, y2) in enumerate(squares):
    square = image2[y1:y2, x1:x2]  # Crop the square
    piece_mask = cv2.bitwise_and(thresh, thresh, mask=square)  # Focus on the piece
    cv2.imwrite(f'square_{i}.png', square)  # Save the square
