import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image
import PIL.ImageOps
import tools.tools as t
import numpy as np
import joblib

app = Flask(__name__)


cnn_model = load_model("models/modelCNN/size100/trainedModelE40.h5")
rf_model = joblib.load("models/modelRF/rf_model_final.joblib")


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cnn_predict(img_path, model=cnn_model):
    img = Image.open(img_path)
    img = PIL.ImageOps.invert(img)
    img = img.resize((100, 100))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    predictions = model.predict(img, verbose=0, batch_size=1)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return img_path, t.classes[predicted_class]


def rf_predict(org_path, cropped_path, clf=rf_model):
    feature = np.zeros(30000)
    probabilities = clf.predict_proba(feature.reshape(1, -1))
    predicted_class = np.argmax(probabilities)

    return org_path, t.classes[predicted_class]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        cnn_prediction = cnn_predict(file_path, cnn_model)
        rf_prediction = rf_predict(file_path, file_path)

        return jsonify({
            'cnn_prediction': cnn_prediction,
            'rf_prediction': rf_prediction
        })

    return jsonify({'error': 'File format not allowed'})


if __name__ == '__main__':
    app.run(debug=True)
