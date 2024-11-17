# inference.py

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model



def preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

def classify_image(model, img_path, class_indices):
    img = preprocess_image(img_path)

    # Predict the class probabilities
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indices[predicted_class_idx]

    return predicted_class

if __name__ == "__main__":
    MODEL_PATH = "F:\Multiclass_Image_Classification\model\model.keras"
    IMAGE_PATH = "F:\Multiclass_Image_Classification\dataset\AbyssinicaSIL-Regular\AbyssinicaSIL-Regular_augmented_1.png" 
    DATA_PATH = "F:\Multiclass_Image_Classification\dataset"  

    model = load_model(MODEL_PATH)


    class_names = sorted(os.listdir(DATA_PATH)) 
    class_indices = {idx: name for idx, name in enumerate(class_names)}

    predicted_class = classify_image(model, IMAGE_PATH, class_indices)

    print(f"The image is classified as: {predicted_class}")
