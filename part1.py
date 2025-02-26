import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "output2.jpg"  
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

        # Convert to RGB for visualization
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Iterate through the grid and highlight squares with pieces
        for i in range(rows):
            for j in range(cols):
                x1, x2 = unique_x[j], unique_x[j + 1]
                y1, y2 = unique_y[i], unique_y[i + 1]

                # Extract the square
                square = image[y1:y2, x1:x2]

                # Check if the square contains a piece (variance in intensity)
                if np.std(square) > 10:  # Adjust this threshold as needed
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print("Error: No corners detected. Check the input image quality.")
else:
    print("Error: No lines detected. Check the input image quality.")
