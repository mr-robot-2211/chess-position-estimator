import cv2
import numpy as np
import os

# Create the output folder
output_folder = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/outputs5"
os.makedirs(output_folder, exist_ok=True)

# Load the images
image = cv2.imread("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/output2.jpg")
image2 = cv2.imread("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/experimented int outputs/output4.jpg")

if image is None or image2 is None:
    raise FileNotFoundError("One or both input images could not be loaded.")

# resized_image1 = cv2.resize(image, 512, interpolation=cv2.INTER_AREA)
# resized_image2 = cv2.resize(image2, 512, interpolation=cv2.INTER_AREA)

# Double the size of the image
new_width = 2 * image2.shape[1]  # Double the width
new_height = 2 * image2.shape[0]  # Double the height

resized_image = cv2.resize(image2, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Use Canny edge detection
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# Use Hough Line Transformation to detect grid lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

if lines is None:
    raise ValueError("No lines detected. Ensure the input image contains a clear chessboard.")

# Separate and process horizontal and vertical lines
horizontal_lines = []
vertical_lines = []

for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y2 - y1) < 10:  # Horizontal line
            horizontal_lines.append((y1, y2))
        elif abs(x2 - x1) < 10:  # Vertical line
            vertical_lines.append((x1, x2))

# Filter and sort lines
def filter_and_sort_lines(lines, threshold=10):
    filtered_lines = []
    for i in sorted(set(line[0] for line in lines)):
        if not filtered_lines or abs(i - filtered_lines[-1]) > threshold:
            filtered_lines.append(i)
    return filtered_lines

horizontal_lines = filter_and_sort_lines(horizontal_lines)
vertical_lines = filter_and_sort_lines(vertical_lines)

# Ensure the grid covers the entire image
if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
    horizontal_lines = [0, gray.shape[0]]
    vertical_lines = [0, gray.shape[1]]

horizontal_lines = [0] + [line for line in horizontal_lines if line < gray.shape[0]] + [gray.shape[0]]
vertical_lines = [0] + [line for line in vertical_lines if line < gray.shape[1]] + [gray.shape[1]]

rows = len(horizontal_lines) - 1
cols = len(vertical_lines) - 1

# Initialize the board matrix
matrix = []

# Analyze each cell in the grid
# Analyze each cell in the grid
for row in range(rows):
    matrix_row = []
    for col in range(cols):
        # Define the cell boundaries
        cell_x1 = vertical_lines[col]
        cell_x2 = vertical_lines[col + 1]
        cell_y1 = horizontal_lines[row]
        cell_y2 = horizontal_lines[row + 1]

        # Validate cell dimensions
        if (cell_x2 - cell_x1) < 10 or (cell_y2 - cell_y1) < 10:
            matrix_row.append(0)
            continue

        # Extract the cell ROI
        cell_roi = gray[cell_y1:cell_y2, cell_x1:cell_x2]
        piece_image = resized_image[cell_y1:cell_y2, cell_x1:cell_x2]

        # Check if ROI is valid
        if cell_roi.size == 0 or piece_image.size == 0:
            print(f"Skipping empty ROI for cell ({row}, {col}): "
                  f"x1={cell_x1}, x2={cell_x2}, y1={cell_y1}, y2={cell_y2}")
            matrix_row.append(0)
            continue

        # Determine cell color
        avg_intensity = np.mean(cell_roi)
        square_value = 1 if avg_intensity < 128 else 0  # Black = 1, White = 0

        # Save the cell image
        output_filename = os.path.join(output_folder, f"({row},{col}).jpg")
        cv2.imwrite(output_filename, piece_image)

        # Add the square value to the matrix
        matrix_row.append(square_value)
    matrix.append(matrix_row)


# Detect pieces using contours
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate a size threshold for pieces
radii = [cv2.minEnclosingCircle(contour)[1] for contour in contours]
size_threshold = np.median(radii) * 0.7  # Adjust multiplier based on image

# Mark pieces on the board
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    if radius > size_threshold:  # Ignore small contours
        for row in range(rows):
            for col in range(cols):
                cell_x1 = vertical_lines[col]
                cell_x2 = vertical_lines[col + 1]
                cell_y1 = horizontal_lines[row]
                cell_y2 = horizontal_lines[row + 1]

                if cell_x1 <= center[0] < cell_x2 and cell_y1 <= center[1] < cell_y2:
                    matrix[row][col] = "x"

# Save the matrix to a file
output_matrix_path = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/pieces2.0.txt"
with open(output_matrix_path, 'w') as f:
    for row in matrix:
        f.write(' '.join(str(item) for item in row) + '\n')

# Visualize the detected grid and pieces
for row in range(rows):
    for col in range(cols):
        cell_x1 = vertical_lines[col]
        cell_x2 = vertical_lines[col + 1]
        cell_y1 = horizontal_lines[row]
        cell_y2 = horizontal_lines[row + 1]

        # Draw the grid
        cv2.rectangle(resized_image, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 255, 0), 1)

        # Mark cells with pieces
        if matrix[row][col] == "x":
            cv2.circle(resized_image, ((cell_x1 + cell_x2) // 2, (cell_y1 + cell_y2) // 2), 5, (0, 0, 255), -1)

cv2.imshow("Detected Board", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
