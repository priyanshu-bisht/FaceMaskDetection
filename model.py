#imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#data gens
TRAINING_DIR = "./Face Mask Dataset/Train"
VALIDATION_DIR = "./Face Mask Dataset/Validation"
TEST_DIR = "./Face Mask Dataset/Test"

training_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
                                                    TRAINING_DIR,
                                                    target_size=(50,50),
                                                    class_mode='binary',
                                                    batch_size=126)

validation_generator = validation_datagen.flow_from_directory(
                                                            VALIDATION_DIR,
                                                            target_size=(50,50),
                                                            class_mode='binary',
                                                            batch_size=126)

test_generator = test_datagen.flow_from_directory(
                                                VALIDATION_DIR,
                                                target_size=(50,50),
                                                class_mode='binary',
                                                batch_size=126)

#Keras Model - mask01

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(50, 50, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=25, validation_data = validation_generator)

model.evaluate(test_generator)

model.save('mask01.h5')

#Keras Model - mask099

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(50, 50, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=25, validation_data = validation_generator)

model.evaluate(test_generator)

model.save('mask099.h5')
