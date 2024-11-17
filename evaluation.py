import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from data_loader import create_datasets

def evaluate_model(model_path, data_path, batch_size=8):
    model = tf.keras.models.load_model(model_path)

    _, test_ds = create_datasets(data_path, batch_size=batch_size)

    # Loss, Accuracy
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


    y_true = test_ds.classes 
    y_pred = model.predict(test_ds, verbose=1)  
    y_pred_classes = np.argmax(y_pred, axis=1)  

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=test_ds.class_indices.keys()))

    #Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:\n", cm)



if __name__ == "__main__":
    MODEL_PATH = "F:\Multiclass_Image_Classification\model\model.keras"  
    DATA_PATH = "F:\Multiclass_Image_Classification\dataset" 
    BATCH_SIZE = 16


    evaluate_model(MODEL_PATH, DATA_PATH, batch_size=BATCH_SIZE)
