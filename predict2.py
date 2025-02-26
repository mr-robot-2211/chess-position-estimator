from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('chess_piece_detector.h5')

# Define image dimensions (adjust based on your model input size)
img_height = 64  # Height of the input image
img_width = 64   # Width of the input image

# Load an image to predict
img_path = "output2.jpg"
img = image.load_img(img_path, target_size=(img_height, img_width))

# Preprocess the image (convert to array, rescale)
img_array = image.img_to_array(img)  # Convert image to numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Rescale to [0, 1] range

# Predict the class of the image
predictions = model.predict(img_array)
# List of class labels (make sure this matches your training setup)
class_labels = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wk', 'wp', 'wq', 'wr']

# Get the predicted class number (e.g., class 11)
predicted_class = np.argmax(predictions, axis=1)

# Print the corresponding class name
print(f"Predicted class index: {predicted_class}")
print(f"Predicted class name: {class_labels[predicted_class[0]]}")