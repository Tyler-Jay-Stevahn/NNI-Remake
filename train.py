import tensorflow as tf
from models import create_image_model
from datasets import recycling_data
from models import resize_image

model = create_image_model()

data_folder = recycling_data

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_folder,
    labels='inferred',
    image_size=(128, 128),
    batch_size=32
)

resized_train_ds = train_ds.map(resize_image)

dataset = train_ds.shuffle(buffer_size=1000) # Shuffle the dataset

dataset_size = tf.data.experimental.cardinality(dataset).numpy() # Get the dataset size

train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)
val_ds = test_ds.skip(test_size)
test_ds = test_ds.take(test_size)

model.compile(
    optimizer='adam', # or tf.keras.optimizers.Adam()
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # use from_logits=True if your last layer doesn't have softmax
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10, # Number of training epochs
    verbose=2
)