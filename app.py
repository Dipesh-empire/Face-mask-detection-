import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Title
st.title("😷 Face Mask Detection App")
st.write("Upload an image and the model will predict if the person is wearing a mask.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("face_mask_detector.h5")
    return model

model = load_model()

# Set class names as per your model
class_names = ["Mask", "No Mask", "Improper Mask"]

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.convert("RGB").resize((150, 150))
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
