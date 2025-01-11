import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('my food101model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = Image.open(image_file).resize((224, 224)).convert('RGB')  # Ensure RGB and resize
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=-1)[0]  # Get the class index with the highest probability
    pred_confidence = np.max(pred)  # Confidence score for the predicted class

    food_classes = [
        'burger', 'butter naan', 'chai', 'chapati', 'chole_bhature', 'dal makhani',
        'dhokla', 'fried_rice', 'idli', 'jalebi', 'kathi roll', 'kadhai paneer',
        'kulfi', 'butter naan', 'momos', 'paani puri', 'pakode', 'pav bhaji', 'pizza', 'samosa'
    ]

    return pred_confidence, food_classes[pred_class]

# Streamlit App
st.title('BCS Food101 Classifier')

uploaded_image = st.file_uploader("Upload an image of a food item ", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', width=224)
    st.write("")
    
st.subheader('Prediction:')
confidence, predicted_class = preprocess_image(uploaded_image)
st.write(f'I am {confidence * 100:.2f}% confident that this is an image of **{predicted_class}**.')
