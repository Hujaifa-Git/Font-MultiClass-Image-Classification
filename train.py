from model import build_model
from data_loader import create_datasets
import tensorflow as tf
import os
from datetime import datetime

def train_model(data_path, save_path, log_dir, epochs=20, batch_size=32):
    train_ds, val_ds = create_datasets(data_path, batch_size=batch_size)
    model = build_model()

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",  # Minimize validation loss
        verbose=1,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,  # Stop if no improvement for 5 epochs
        verbose=1,
        restore_best_weights=True,
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )

    model.save(save_path)
    print("Model training complete. Model saved at:", save_path)


if __name__ == "__main__":
    DATA_PATH = "F:\Multiclass_Image_Classification\dataset"
    SAVE_PATH = "F:\Multiclass_Image_Classification\model\model.keras"
    LOG_DIR = 'logs'
    EPOCHS = 20
    BATCH_SIZE = 32

    train_model(DATA_PATH, SAVE_PATH, LOG_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
