import tensorflow as tf 
from tensorflow.keras import models,layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
VERSION = 2

if __name__ == "__main__":
    
    data_augmentation = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.2

    ) 
    augmented_train_dataset = data_augmentation.flow_from_directory(
        'dataset/train',
        target_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse"
    )
    augmented_validation_dataset = data_augmentation.flow_from_directory(
        'dataset/val',
        target_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse"
    )
    augmented_test_dataset = data_augmentation.flow_from_directory(
        'dataset/test',
        target_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse"
    )
    print("len",len(augmented_train_dataset))

    input_shape = (IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation="relu"),
        layers.Dense(3,activation="softmax")
    ])
    #model.build(shape=input_shape);
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    model.fit(augmented_train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=int(len(augmented_train_dataset)),
              batch_size=BATCH_SIZE,
              validation_data=augmented_validation_dataset,
              validation_steps=int(len(augmented_validation_dataset))
    )
    scores = model.evaluate(augmented_test_dataset)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save("models/model.{}.h5".format(VERSION))




    
    



