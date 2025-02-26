import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/output2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection to find grid lines
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough line transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Separate horizontal and vertical lines
horizontal_lines = []
vertical_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
        horizontal_lines.append((y1, y2))
    else:  # Vertical line
        vertical_lines.append((x1, x2))

# Sort lines to determine grid intersections
horizontal_lines = sorted(set(y for y1, y2 in horizontal_lines for y in (y1, y2)))
vertical_lines = sorted(set(x for x1, x2 in vertical_lines for x in (x1, x2)))

# Determine grid size
grid_size = min(len(horizontal_lines) - 1, len(vertical_lines) - 1)
if grid_size < 1:
    raise ValueError("Unable to detect a valid grid.")

# Divide the board into grid squares
square_width = image.shape[1] // grid_size
square_height = image.shape[0] // grid_size

# Map pieces to squares
piece_positions = []
for i in range(grid_size):
    for j in range(grid_size):
        # Define square boundaries
        square_x1 = j * square_width
        square_y1 = i * square_height
        square_x2 = square_x1 + square_width
        square_y2 = square_y1 + square_height

        # Extract the square's ROI
        square_roi = gray[square_y1:square_y2, square_x1:square_x2]

        # Check for pieces in the square
        _, binary_roi = cv2.threshold(square_roi, 128, 255, cv2.THRESH_BINARY_INV)
        non_zero_count = cv2.countNonZero(binary_roi)
        if non_zero_count > 0:  # Adjust threshold as needed
            piece_positions.append(((i, j), square_roi))

# Visualize detected pieces
for (pos, roi) in piece_positions:
    print(f"Piece detected at square: {pos}")
    cv2.imshow(f"Piece at {pos}", roi)

cv2.imshow("Chessboard", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
