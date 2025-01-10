# Install necessary package
#!pip install streamlit

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Custom loss function to handle deserialization
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class CustomSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __init__(self, reduction='auto', name='sparse_categorical_crossentropy', from_logits=False):
        super().__init__(reduction=reduction, name=name, from_logits=from_logits)

# Register the custom loss function
tf.keras.utils.get_custom_objects().update({
    'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy
})

# Load the pre-trained model
model = tf.keras.models.load_model('my food101model.h5', custom_objects={'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy})

# Define class labels
class_names = [
    'burger', 'butter naan', 'chai', 'chapati', 'chole_bhature', 'dal makhani',
    'dhokla', 'fried_rice', 'idli', 'jalebi', 'kathi roll', 'kadhai paneer',
    'kulfi', 'momos', 'paani puri', 'pakode', 'pav bhaji', 'pizza', 'samosa'
]

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = Image.open(uploaded_file).resize((224, 224))  # Resize to match model's input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Streamlit App
st.title('BCS Food101 Classifier')

uploaded_image = st.file_uploader("Upload an image of a food item (limit large file sizes)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Classify'):
        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_image)

        # Make a prediction using the pre-trained model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = class_names[predicted_class]

        st.success(f'Prediction: {predicted_label}')
