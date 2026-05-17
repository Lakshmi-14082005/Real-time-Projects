import os
import json
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model("cnn_model.h5")

# ✅ Load correct class order
with open("classes.json", "r") as f:
    classes = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Create folder if not exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # Read image safely
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return "Image loading failed ❌"

    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    # Prediction
    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result = classes[index]

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence,2),
        image=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)