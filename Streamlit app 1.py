# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Config ---
st.set_page_config(page_title="Soil Type Classifier", layout="centered")
st.title("🌱 Soil Type Classifier using MobileNetV2")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("soil_classifier_mobilenetv2.h5")
    return model

model = load_model()

# --- Define Class Labels ---
class_names = ['Alluvial Soil', 'Black Soil', 'Red Soil', 'Sandy Soil']  # Adjust if needed

# --- Preprocess Uploaded Image ---
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a soil image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🧠 Predict Soil Type"):
        with st.spinner("Classifying..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.success(f"🌍 **Predicted Soil Type:** `{predicted_class}`")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")
