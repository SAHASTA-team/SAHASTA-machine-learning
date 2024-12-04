from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import io
from PIL import Image

app = Flask(__name__)

# Label klasifikasi
labels = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
    'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Load model
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Proses file gambar
        image = Image.open(io.BytesIO(file.read())).convert('RGB')  # Konversi ke RGB
        image = image.resize((224, 224))  # Resize ke ukuran input model
        img_array = img_to_array(image) / 255.0  # Normalisasi ke [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch (1, 224, 224, 3)
        
        # Prediksi
        probabilities = model.predict(img_array)
        predicted_label = labels[np.argmax(probabilities)]
        confidence = probabilities[0][np.argmax(probabilities)]  # Probabilitas tertinggi

        return jsonify({'prediction': predicted_label, 'confidence': float(confidence)})

        # return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)