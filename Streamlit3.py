import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
import requests
from openai import OpenAI
import os
# -------------------- CONFIG --------------------
st.set_page_config(page_title="Soil Type & AI Crop Recommendation", layout="centered")
st.title("🌱 Soil Type, Crop Recommendation & Location Info")

api_key = st.secrets["OPENAI_API_KEY"].strip()
client = OpenAI(api_key=api_key)
# -------------------- LOAD MODEL & LABEL ENCODER --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("soil_multimodal_model.h5")

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
label_encoder = load_label_encoder()

# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------- STEP 1: MAP CLICK --------------------
st.subheader("📍 Select Location on Map")
m = folium.Map(location=[20, 0], zoom_start=2)
m.add_child(folium.LatLngPopup())
st.write("Click anywhere on the map to get country and elevation:")
output = st_folium(m, width=700, height=500)

country = None
elevation = None
if output and output["last_clicked"]:
    lat = output["last_clicked"]["lat"]
    lon = output["last_clicked"]["lng"]
    st.write(f"**Selected Coordinates:** {lat:.4f}, {lon:.4f}")

    geolocator = Nominatim(user_agent="streamlit-app")
    location = geolocator.reverse((lat, lon), language="en")
    country = location.raw['address'].get('country', 'Unknown')
    st.write(f"**Country:** {country}")

    elev_api = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    response = requests.get(elev_api)
    if response.status_code == 200:
        elevation = response.json()["results"][0]["elevation"]
        st.write(f"**Elevation:** {elevation} meters")
    else:
        st.write("⚠️ Could not retrieve elevation.")

# -------------------- STEP 2: IMAGE UPLOAD --------------------
st.markdown("### 📷 Upload Soil Image")
uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])

# -------------------- STEP 3: PREDICT & AI RECOMMENDATION --------------------
if uploaded_file is not None and country is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🧠 Predict Soil Type & Get AI Crop Recommendations"):
        with st.spinner("Classifying soil type..."):
            img_input = preprocess_image(image)

            # Dummy tabular features (ph, N, P, K, humidity = 5 features)
            dummy_tabular = np.zeros((1, 5))

            prediction = model.predict([img_input, dummy_tabular])
            predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction) * 100

        st.success(f"🌍 **Predicted Soil Type:** `{predicted_class}`")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")
        st.write(f"📌 Country: **{country}**")
        if elevation is not None:
            st.write(f"📌 Elevation: **{elevation} meters**")

        # -------------------- OPENAI RECOMMENDATION --------------------
        with st.spinner("Asking AI for crop recommendations..."):
            prompt = (
                f"You are an agricultural expert. Based on the following details:\n"
                f"- Country: {country}\n"
                f"- Elevation: {elevation} meters\n"
                f"- Soil Type: {predicted_class}\n\n"
                f"Suggest the most suitable crops to grow in this location. "
                f"List them in bullet points with short reasons."
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful agricultural assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )

            ai_recommendations = response.choices[0].message.content

        st.markdown("🌾 **AI Recommended Crops:**")
        st.markdown(ai_recommendations)




