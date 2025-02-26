import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/chess_piece_detector.h5")

# Define image dimensions (adjust based on your model input size)
img_height = 64  # Height of the input image
img_width = 64   # Width of the input image

# Folder containing the square images
output_folder = "/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/outputs4"

# List to store class labels for prediction
class_labels = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wk', 'wp', 'wq', 'wr']

# Initialize the matrix to store predicted class labels
matrix = []

# Loop through all the images in the outputs folder
for filename in sorted(os.listdir(output_folder)):
    # Check if the file is a .jpg image
    if filename.endswith(".jpg"):
        try:
            # Remove parentheses and the file extension
            row_col = filename[1:-4]  # Remove the first '(' and the last 4 chars ('.jpg')
            row_col = row_col.rstrip(')')  # Remove trailing ')'
            
            # Now split by comma to get row and column
            row, col = map(int, row_col.split(','))  # Split by comma and convert to integers
            
            # Construct the full path to the image
            img_path = os.path.join(output_folder, filename)

            # Load the image and preprocess it
            img = image.load_img(img_path, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)  # Convert image to numpy array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Rescale to [0, 1] range

            # Predict the class of the image
            predictions = model.predict(img_array)

            # Get the predicted class index
            predicted_class = np.argmax(predictions, axis=1)

            # Ensure the matrix has enough rows
            while len(matrix) <= row:
                matrix.append([])

            # Add the predicted class label to the matrix
            matrix[row].append(class_labels[predicted_class[0]])

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Print the resulting matrix with predicted class labels
with open("/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/my-model/pieces.txt", 'w') as f:
    print("Predicted Board Matrix:")
    for row in matrix:
        print(row)
        f.write(' '.join(str(item) for item in row) + '\n')

