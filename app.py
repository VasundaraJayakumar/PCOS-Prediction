import requests
import json
from io import BytesIO
from PIL import Image
from keras.applications.vgg16 import VGG16
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the label dictionary
label = {0: 'PCOS', 1: 'NORMAL'}

vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in vggmodel.layers:
    layer.trainable = False

model = joblib.load('xray.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_url = request.json.get('url')
        if image_url:
            try:
                image_data = requests.get(image_url).content
                image = Image.open(BytesIO(image_data))
                image = image.resize((256, 256))
                image = np.expand_dims(image, axis=0)
                image = image / 255.0
                feature_extractor = vggmodel.predict(image)
                features = feature_extractor.reshape(feature_extractor.shape[0], -1)
                prediction = model.predict(features)[0]
                final = label[prediction]
                return jsonify({'prediction': final})
            except Exception as e:
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'No image URL provided'})
    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)