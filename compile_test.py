import tensorflow as tf
from models import create_image_model

try:
    
    model = create_image_model()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    print("Model Compiles")
except ValueError as e: # Catch the specific ValueError
    print(f"Model failed to compile or define: {e}")
except Exception as e: # Catch other potential errors during compilation
    print(f"An unexpected error occurred: {e}")

