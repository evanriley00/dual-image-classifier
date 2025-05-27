import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import cv2

# Load trained model (make sure it's saved as 'fashion_model.h5')
model = load_model('fashion_model.h5')

# Labels for Fashion MNIST
labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # normalize
    img = img.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img)
    pred_label = labels[np.argmax(prediction)]

    return render_template('result.html', label=pred_label)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
