from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\devan_jmbn6cf\Downloads\DATASET\DATASET",
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    r"C:\Users\devan_jmbn6cf\Downloads\edatatede",
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse')
cnn = tf.keras.models.Sequential()  # artifical neural network
# convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu', input_shape=[64, 64, 3]))
# pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# 2nd convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# flattening
cnn.add(tf.keras.layers.Flatten())
# full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# output layer
cnn.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# training the cnn
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


# cnn.fit(x=train_generator, validation_data=validation_generator, epochs=25)


test_image = image.load_img(
    r'C:\Users\devan_jmbn6cf\Downloads\otestdataset\melanoma_9968.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)

# print(train_datagen.class_indices)

predicted_class = np.argmax(result, axis=1)
predicted_class = predicted_class + 1
print("ans:", predicted_class[0])