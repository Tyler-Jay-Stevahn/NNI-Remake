import tensorflow as tf

desired_height = 128
desired_width = desired_height
num_classes = 11

def resize_image(image, label):
    image = tf.image.resize(image, [desired_height, desired_height])
    return image, label

def create_broken_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(desired_height, desired_width, 3)), # Input layer matching image size and color channels
        tf.keras.layers.Dense(units=num_classes, activation='softermax') # Output layer for classification
    ])
    return model

def create_image_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(desired_height, desired_width, 3)), # Input layer matching image size and color channels
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_classes, activation='softmax') # Output layer for classification
    ])
    return model

