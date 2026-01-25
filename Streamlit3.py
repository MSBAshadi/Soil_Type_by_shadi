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
from streamlit_js_eval import streamlit_js_eval
import os
# -------------------- CONFIG --------------------
st.set_page_config(page_title="Soil Type & AI Crop Recommendation", layout="centered")
st.title("🌱 Soil Type, Crop Recommendation & Location Info")

api_key = st.secrets["OPENAI_API_KEY"].strip()
client = OpenAI(api_key=api_key)

# -------------------- LOAD MODEL & LABEL ENCODER --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"soil_multimodal_model.h5"
    )

@st.cache_resource
def load_label_encoder():
    return joblib.load(
        r"label_encoder.pkl"
    )
model = load_model()
label_encoder = load_label_encoder()

# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------- NAVIGATION STATE --------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Location"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "image" not in st.session_state:
    st.session_state.image = None
if "country" not in st.session_state:
    st.session_state.country = None
if "latlon" not in st.session_state:
    st.session_state.latlon = (None, None)
if "elevation" not in st.session_state:
    st.session_state.elevation = None
if "selected_month" not in st.session_state:
    st.session_state.selected_month = "January"

def go_to(tab):
    st.session_state.active_tab = tab

# -------------------- LOCATION STEP --------------------
if st.session_state.active_tab == "Location":
    st.header("📍 Step 1: Select Location")

    col1, col2 = st.columns(2)
    lat, lon = None, None

    with col1:
        st.write("🌍 Click on the map to select a location")
        m = folium.Map(location=[20, 0], zoom_start=2)
        m.add_child(folium.LatLngPopup())
        output = st_folium(m, width=600, height=400)
        if output and output["last_clicked"]:
            lat = output["last_clicked"]["lat"]
            lon = output["last_clicked"]["lng"]

    with col2:
        st.write("📱 Or use your current location")
        if st.button("Use My Location"):
            coords = streamlit_js_eval(
                js_expressions="""
                new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(
                        (pos) => {
                            resolve({lat: pos.coords.latitude, lon: pos.coords.longitude});
                        },
                        (err) => {
                            reject(err.message);
                        }
                    );
                })
                """,
                key="get_location"
            )
            if coords and "lat" in coords and "lon" in coords:
                lat, lon = coords["lat"], coords["lon"]

    if lat and lon:
        try:
            geolocator = Nominatim(user_agent="streamlit-app")
            location = geolocator.reverse((lat, lon), language="en")
            st.session_state.country = location.raw['address'].get('country', 'Unknown')
        except:
            st.session_state.country = "Unknown"

        try:
            elev_api = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
            response = requests.get(elev_api)
            if response.status_code == 200:
                st.session_state.elevation = response.json()["results"][0]["elevation"]
        except:
            st.session_state.elevation = None

        st.session_state.latlon = (lat, lon)

        st.success(f"📍 Location saved: {lat:.4f}, {lon:.4f} — {st.session_state.country}")

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    st.session_state.selected_month = st.selectbox("📅 Planting Month", months)

    if st.session_state.country:
        if st.button("➡️ Next: Upload Soil Image"):
            go_to("Soil Image")

# -------------------- IMAGE STEP --------------------
elif st.session_state.active_tab == "Soil Image":
    st.header("📷 Step 2: Upload Soil Image")
    uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.image = Image.open(uploaded_file).convert('RGB')
        st.image(st.session_state.image, caption="Uploaded Soil Image", use_column_width=True)

    col1, col2 = st.columns([1,1])
    if col1.button("⬅️ Back"):
        go_to("Location")
    if col2.button("➡️ Next: Prediction") and st.session_state.image is not None:
        go_to("Prediction")

# -------------------- PREDICTION STEP --------------------
elif st.session_state.active_tab == "Prediction":
    st.header("🧠 Step 3: Soil Prediction & Crop Recommendations")

    if st.session_state.image is not None and st.session_state.country is not None:
        if st.button("🔎 Predict & Recommend"):
            with st.spinner("Classifying soil type..."):
                img_input = preprocess_image(st.session_state.image)
                dummy_tabular = np.zeros((1, 5))
                prediction = model.predict([img_input, dummy_tabular])
                predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                confidence = np.max(prediction) * 100

            col1, col2 = st.columns(2)
            col1.metric("🌍 Soil Type", predicted_class)
            

            with st.spinner("Getting AI crop recommendations..."):
                prompt = (
                    f"You are an agricultural expert. Based on the following details:\n"
                    f"- Country: {st.session_state.country}\n"
                    f"- Elevation: {st.session_state.elevation} meters\n"
                    f"- Soil Type: {predicted_class}\n"
                    f"- Planting Month: {st.session_state.selected_month}\n\n"
                    f"Suggest the most suitable crops to grow in this location during that month. "
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

            st.markdown("### 🌾 Recommended Crops")
            st.markdown(
                f"<div style='padding:15px; background:#FFF3E0; border-radius:12px; "
                f"border:1px solid #FFB74D;'>{ai_recommendations}</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please complete previous steps first.")

    if st.button("⬅️ Back"):
        go_to("Soil Image")






