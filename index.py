import os
from flask import Flask, request, render_template_string, redirect, send_from_directory, url_for, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pyngrok import ngrok
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
import cv2

import sys
print(f"Python version: {sys.version}")


# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load your pre-trained model
model_path = 'model/Klasifikasi_ikan.h5'
loaded_model = load_model(model_path)


def preprocess_image(img_path, target_size):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return img_array
    except UnidentifiedImageError:
        logging.error(f"Cannot identify image file {img_path}")
        return None


def predict_fish_species(img_path):
    img_array = preprocess_image(img_path, target_size=(224, 224))
    if img_array is None:
        return "Unknown Fish", 0

    img_array = np.expand_dims(img_array, axis=0)
    prediction = loaded_model.predict(img_array)
    class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet',
                   'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']

    max_probability = np.max(prediction)
    if max_probability < 0.85:
        predicted_class = "Unknown Fish"
    else:
        predicted_class = class_names[np.argmax(prediction)]

    return predicted_class, max_probability


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# HTML template with webcam detection
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FishDetect</title>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&display=swap');

    body {
        font-family: "EB Garamond", serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 80vh;
        background: radial-gradient(#1cb5e0, #000066);
        margin: 0;
        color: #fff;
        padding-top: 20px;
    }

    .container {
        background: #FFF;
        color: #333;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 100%;
        max-width: 800px;
        margin-top: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    h1 {
        font-size: 3em;
        margin-bottom: 20px;
        margin-top: 40px;
    }

    h2 {
        text-align: center;
    }

    h3 {
        text-align: center;
        margin-top: 60px;
    }

    form {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    input[type="file"] {
        margin: 20px 0;
        padding: 10px;
        border: 2px solid #000066;
        border-radius: 5px;
        width: 100%;
    }

    input[type="submit"] {
        background: #000066;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background 0.3s;
        width: 100%;
        margin-top: 10px;
    }

    input[type="submit"]:hover {
        background: #150754;
    }

    .result-container {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    img {
        max-width: 400px;
        max-height: 400px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin-right: 20px;
        width: auto;
        height: auto;
    }

    .text-container {
        text-align: left;
    }

    .text-container p {
        font-size: 20px;
        color: #333;
        margin-bottom: 20px;
    }

    a {
        text-decoration: none;
        color: #000066;
        border: 1px solid #000066;
        margin-top: 20px;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background 0.3s, color 0.3s;
        display: inline-block;
    }

    a:hover {
        background: #007bff;
        color: white;
    }

    ul {
        text-align: left;
        margin-bottom: 20px;
    }

    li {
        font-size: 1.0em;
        color: #333;
    }

    .fish-card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 40px;
    }

    .fish-card {
        background: #fff;
        color: #333;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 10px;
        padding: 20px;
        width: 180px;
    }

    .fish-card img {
        width: 100%;
        height: auto;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .fish-card p {
        margin: 0;
        font-size: 1em;
        font-weight: bold;
        color: #000066;
    }
    
    .realtime-container {
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    #videoElement {
        width: 400px;
        height: auto;
        margin-bottom: 10px;
        border-radius: 10px;
    }

    </style>

</head>
<body>
    <h1>Fish Detect</h1>
    <div class="container">
        {% if not image_url %}
        <h2>Upload an Image or Use the Camera</h2>
        <form action="/" method="post" enctype="multipart/form-data">
            <label for="file">Upload your image:</label><br>
            <input type="file" name="file" id="file">
            <input type="submit" value="Upload">
        </form>

        <h2>Or use the camera to detect in real-time:</h2>
        <div class="realtime-container">
            <img src="{{ url_for('video_feed') }}" id="videoElement" alt="Video Stream">
            <form action="/capture" method="post">
                <input type="submit" value="Capture and Detect">
            </form>
        </div>

        {% else %}
        <div class="result-container">
            <img src="{{ image_url }}" alt="Uploaded Image">
            <div class="text-container">
                <p>The predicted fish species is: {{ predicted_class }}</p>
                <p>Confidence: {{ confidence }}%</p>
                <a href="/">Upload another image</a>
            </div>
        </div>
        {% endif %}
    </div>
     <h3>Jenis Ikan yang dapat di deteksi</h3>
        <div class="fish-card-container">
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/BlackSeaSprat.png') }}" alt="Black Sea Spar">
                <p>Black Sea Spar</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/GiltHeadBream.JPG') }}" alt="Gilt Head Bream">
                <p>Gilt Head Bream</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/HorseMackerel.png') }}" alt="Horse Mackerel">
                <p>Horse Mackerel</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/RedMullet.png') }}" alt="Red Mullet">
                <p>Red Mullet</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/RedSeaBream.JPG') }}" alt="Red Sea Bream">
                <p>Red Sea Bream</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/SeaBass.JPG') }}" alt="Sea Bass">
                <p>Sea Bass</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/Shrimp.png') }}" alt="Shrimp">
                <p>Shrimp</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/StripedRedMullet.png') }}" alt="Striped Red Mullet">
                <p>Striped Red Mullet</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/Trout.png') }}" alt="Trout">
                <p>Trout</p>
            </div>
        </div>
</body>
</html>
'''


@app.route('/static/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Video streaming route for webcam
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Create an HTTP response with a multipart content type (for streaming)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag.
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture_image():
    success, frame = camera.read()
    if success:
        # Save the captured frame to a file
        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'], 'realtime_capture.jpg')
        cv2.imwrite(filepath, frame)

        # Predict the fish species in the captured frame
        predicted_class, confidence = predict_fish_species(filepath)
        image_url = url_for('send_image', filename='realtime_capture.jpg')

        confidence_formatted = "{:.1f}".format(confidence * 100)

        return render_template_string(template, image_url=image_url, predicted_class=predicted_class, confidence=confidence_formatted)
    else:
        logging.error("Failed to read from camera")
    return redirect('/')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            predicted_class, confidence = predict_fish_species(filepath)
            image_url = url_for('send_image', filename=file.filename)

            confidence_formatted = "{:.1f}".format(confidence * 100)

            return render_template_string(template, image_url=image_url, predicted_class=predicted_class, confidence=confidence_formatted)
    return render_template_string(template)


# Start ngrok and Flask server
public_url = ngrok.connect(5000)
print(f" * Public URL: {public_url}")
app.run()
