import tensorflow as tf

def create_datasets(data_path, img_size=(224, 224), batch_size=32, validation_split=0.2):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0, 
        validation_split=validation_split,  
        rotation_range=20,  
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
    )

    train_ds = datagen.flow_from_directory(
        directory=data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_ds = datagen.flow_from_directory(
        directory=data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=True,
    )

    return train_ds, val_ds
