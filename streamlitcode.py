# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = Image.open(image_file).resize((224, 224)).convert('RGB')  # Ensure RGB and resize
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
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
        pred_confidence = np.max(prediction) * 100  # Confidence score as a percentage

        st.success(f'I am {pred_confidence:.2f}% sure that this is an image of {predicted_label}')
