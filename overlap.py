import cv2
import numpy as np

# Load and preprocess the image
img = cv2.imread('output2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fit ellipses to contours
for contour in contours:
    if len(contour) >= 5:  # fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(contour)
        # Check if the dimensions are valid
        if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # ellipse[1] is (width, height)
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)
            center = ellipse[0]  # Ellipse center
            # Map center to board square
            # (Perform perspective transformation and grid mapping here)

# Show the result
cv2.imshow('Detected Ellipses', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
