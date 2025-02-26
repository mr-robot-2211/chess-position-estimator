import cv2
import numpy as np
import os

# Create the output folder
output_folder = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/outputs4"
os.makedirs(output_folder, exist_ok=True)

# Load the image
image = cv2.imread("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/output2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
matrix = []

# Analyze each cell
for row in range(rows):
    matrix_row = []
    for col in range(cols):
        # Define the square boundaries
        cell_x1 = vertical_lines[col]
        cell_x2 = vertical_lines[col + 1]
        cell_y1 = horizontal_lines[row]
        cell_y2 = horizontal_lines[row + 1]

        # Validate square dimensions
        if (cell_x2 - cell_x1) < 10 or (cell_y2 - cell_y1) < 10:
            # Skip very small rectangles
            matrix_row.append(0)
            continue

        # Extract the cell ROI
        cell_roi = gray[cell_y1:cell_y2, cell_x1:cell_x2]

        # Determine if the square is white or black
        avg_intensity = np.mean(cell_roi)
        square_value = 1 if avg_intensity < 128 else 0  # Black = 1, White = 0

        # Add the square value to the row
        matrix_row.append(square_value)
        
        # Save all squares, whether empty or with a piece
        piece_image = image[cell_y1:cell_y2, cell_x1:cell_x2]
        output_filename = os.path.join(output_folder, f"({row},{col}).jpg")
        cv2.imwrite(output_filename, piece_image)

    matrix.append(matrix_row)

# Detect pieces using contours
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

radii = []
for contour in contours:
    (_, _), radius = cv2.minEnclosingCircle(contour)
    radii.append(radius)

# Calculate the median radius
some_threshold = np.median(radii) * 0.7  # Adjust the multiplier as needed

for contour in contours:
    # Fit a minimum enclosing circle around the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))  # Center of the circle
    radius = int(radius)

    # Filter by size to exclude small or irrelevant contours
    if radius > some_threshold:  # Define an appropriate threshold based on the image
        base_x, base_y = center

        # Check which square the base center belongs to
        for row in range(rows):
            for col in range(cols):
                cell_x1 = vertical_lines[col]
                cell_x2 = vertical_lines[col + 1]
                cell_y1 = horizontal_lines[row]
                cell_y2 = horizontal_lines[row + 1]

                if cell_x1 <= base_x < cell_x2 and cell_y1 <= base_y < cell_y2:
                    # Mark the square as having a piece
                    matrix[row][col] = "x"

# Visualize for debugging (optional)
for row in range(rows):
    for col in range(cols):
        if matrix[row][col] == "x":
            print(f"Piece detected in square: ({row}, {col})")

# Print the resulting matrix
with open("C:/Users/sohan/Desktop/sop 3-1/my-model/pieces2.txt", 'w') as f:
    print("Positional Board Matrix:")
    for row in matrix:
        print(row)
        f.write(' '.join(str(item) for item in row) + '\n')

# Show the board with detected squares and pieces (optional)
for row in range(rows):
    for col in range(cols):
        cell_x1 = vertical_lines[col]
        cell_x2 = vertical_lines[col + 1]
        cell_y1 = horizontal_lines[row]
        cell_y2 = horizontal_lines[row + 1]
        cv2.rectangle(image, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 255, 0), 1)

cv2.imshow("Detected Board", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
