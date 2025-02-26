import cv2
import numpy as np
import os

# Load the image
image_path = "output2.jpg"  # Update this path if needed
output_dir = "./squares"  # Directory to save the squares
os.makedirs(output_dir, exist_ok=True)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load the image. Check the file path and integrity.")
    exit()

# Preprocess the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

if lines is not None:
    # Draw lines on a blank mask
    line_mask = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(line_mask, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.int0(corners).reshape(-1, 2)

        # Determine the number of rows and columns in the grid
        unique_x = sorted(set([corner[0] for corner in corners]))
        unique_y = sorted(set([corner[1] for corner in corners]))

        rows = len(unique_y) - 1
        cols = len(unique_x) - 1
        print(f"Detected grid size: {rows + 1}x{cols + 1}")

        # Iterate through the grid and save squares with pieces
        square_images = []
        for i in range(rows):
            for j in range(cols):
                x1, x2 = unique_x[j], unique_x[j + 1]
                y1, y2 = unique_y[i], unique_y[i + 1]

                # Extract the square
                square = image[y1:y2, x1:x2]

                # Check if the square contains a piece (variance in intensity)
                if np.std(square) > 10:  # Adjust this threshold as needed
                    square_filename = f"{output_dir}/square_{i}_{j}.jpg"
                    cv2.imwrite(square_filename, square)
                    square_images.append((i, j, square))

        print(f"Saved {len(square_images)} squares with pieces.")

        # Reconstruct the board
        board_reconstructed = np.zeros_like(image)
        for i, j, square in square_images:
            x1, x2 = unique_x[j], unique_x[j + 1]
            y1, y2 = unique_y[i], unique_y[i + 1]
            board_reconstructed[y1:y2, x1:x2] = square

        # Save and display the reconstructed board
        reconstructed_filename = "./reconstructed_board.jpg"
        cv2.imwrite(reconstructed_filename, board_reconstructed)
        print(f"Reconstructed board saved as {reconstructed_filename}")
        cv2.imshow("Reconstructed Board", board_reconstructed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: No corners detected. Check the input image quality.")
else:
    print("Error: No lines detected. Check the input image quality.")
