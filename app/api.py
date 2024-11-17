from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'F:\Multiclass_Image_Classification\model\model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
     
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))  
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        

        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        class_names = sorted(os.listdir('F:/Multiclass_Image_Classification/dataset'))
        predicted_label = class_names[predicted_class]
        
        return jsonify({
            "predicted_class": predicted_class,
            "predicted_label": predicted_label
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads') 
    app.run(debug=True)
