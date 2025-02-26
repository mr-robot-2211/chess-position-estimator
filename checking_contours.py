import cv2
import numpy as np
import os

# Load and preprocess the image
img = cv2.imread('output2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fit ellipses to contours
# for contour in contours:
    # if len(contour) >= 5:  # fitEllipse requires at least 5 points
    #     ellipse = cv2.fitEllipse(contour)
    #     # Check if the dimensions are valid
    #     if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # ellipse[1] is (width, height)
    #         cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    #         center = ellipse[0]  # Ellipse center
    #         # Map center to board square
    #         # (Perform perspective transformation and grid mapping here)
# edges = cv2.Canny(img,100,200)

# coloured = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dst=clahe.apply(gray)

thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# cv2.drawContours(dst, contours, -1, (0, 255, 0), 2)
img_resized = cv2.resize(thresh, (600, 600))  # Resize to a fixed size
# Show the result
# cv2.imshow('contours',img_resized)
# cv2.waitKey(0)  # Waits for a key press to close the window
# cv2.destroyAllWindows()  # Destroys all OpenCV windows
output_path = os.path.join('/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/experimented int outputs', 'output4.jpg')
cv2.imwrite(output_path, img_resized)
