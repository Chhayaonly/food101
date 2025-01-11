import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the saved model
@st.cache_resource
def load_food101_model():
    return load_model('my_food101_model.keras')

model = load_food101_model()

def predict_food101(img_path):
    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match model's input shape
    img = img / 255.0  # Normalize the image (since ImageDataGenerator used rescale=1./255)

    # Add a batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict the class
    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=-1)[0]  # Get the class index with the highest probability
    pred_confidence = np.max(pred)  # Confidence score for the predicted class

    food_classes = [
        'burger', 'butter naan', 'chai', 'chapati', 'chole_bhature', 'dal makhani',
        'dhokla', 'fried_rice', 'idli', 'jalebi', 'kathi roll', 'kadhai paneer',
        'kulfi', 'butter naan', 'momos', 'paani puri', 'pakode', 'pav bhaji', 'pizza', 'samosa'
    ]

    return pred_confidence, food_classes[pred_class]

def show_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, channels="RGB")

# Streamlit interface
st.title('Food101 Image Classification')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with open('temp_image.jpg', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.subheader('Uploaded Image:')
    show_rgb('temp_image.jpg')

    st.subheader('Prediction:')
    confidence, predicted_class = predict_food101('temp_image.jpg')
    st.write(f'I am {confidence * 100:.2f}% confident that this is an image of **{predicted_class}**.')
