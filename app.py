import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("multi_crop_weed_detector.h5")

# Set class names
class_names = ['Crop', 'Weed']

# Title
st.title("🌿 Multi-Crop Weed Detection")
st.subheader("Upload a field image to predict whether it's a crop or weed")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    confidence = round(prediction * 100 if predicted_class == "Weed" else (1 - prediction) * 100, 2)

    # Output
    st.markdown("---")
    st.subheader(f"🧠 Prediction: **{predicted_class}**")
    st.markdown(f"📊 Confidence: **{confidence}%**")
