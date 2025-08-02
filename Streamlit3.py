import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Crop Recommendation System", layout="centered")
st.title("🌱 Crop Recommendation by Soil Type ")

# --- Load Model & Encoders ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("soil_multimodal_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

@st.cache_data
def load_crop_data():
    return pd.read_csv("Crop_recommendation_with_soil_type.csv")

model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()
crop_df = load_crop_data()

# --- Image Preprocessing ---
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Tabular Preprocessing ---
def preprocess_tabular(ph, n, p, k, humidity):
    features = np.array([[ph, n, p, k, humidity]])
    scaled = scaler.transform(features)
    return scaled

# --- Input Interface ---
uploaded_file = st.file_uploader("📷 Upload a soil image", type=["jpg", "jpeg", "png"])

# --- Prediction ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🧠 Predict Soil Type"):
        with st.spinner("Classifying..."):
            img_input = preprocess_image(image)
            tab_input = preprocess_tabular(ph, n, p, k, humidity)

            prediction = model.predict([img_input, tab_input])
            predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction) * 100

        st.success(f"🌍 **Predicted Soil Type:** `{predicted_class}`")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")

        # --- Crop Recommendation from CSV ---
        recommended_crops = crop_df[crop_df['soil type'].str.lower().str.strip() == predicted_class.lower().strip()]['label'].unique()

        if len(recommended_crops) > 0:
            st.markdown("🌾 **Recommended Crops:**")
            for crop in recommended_crops:
                st.markdown(f"- {crop}")
        else:
            st.warning("⚠️ No crop recommendations available for this soil type.")


