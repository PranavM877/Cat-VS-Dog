import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# ---------------- STREAMLIT CONFIG ---------------- #
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS: Center + remove white strip
st.markdown("""
<style>
.block-container {
    max-width: 900px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
.main-header {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: #2E86C1;
    margin-bottom: 0.5rem;
}
.sub-header {
    text-align: center;
    font-size: 1.1rem;
    color: #5f6a7d;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL DOWNLOAD ---------------- #
MODEL_PATH = "dog_cat_model.h5"
FILE_ID = "1rdLZBi7S_BIDfUbljvbLMrHySnoTKJbo"   # 🔹 Google Drive file ID
URL = "https://drive.google.com/uc?id=1rdLZBi7S_BIDfUbljvbLMrHySnoTKJbo"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- PREPROCESSOR ---------------- #
class InferencePreprocessor:
    def __init__(self, img_size):
        self.img_size = img_size

    def preprocess_image_pil(self, img):
        img_resized = img.resize(self.img_size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

# ---------------- CLASSIFICATION ---------------- #
def classify_image(model, img_array):
    prediction = model.predict(img_array, verbose=0)[0][0]
    if prediction > 0.5:
        return 'Dog', prediction
    else:
        return 'Cat', 1 - prediction

def display_prediction_results(predicted_class, confidence):
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 Prediction", value=predicted_class)
    with col2:
        st.metric(label="🎲 Confidence", value=f"{confidence:.1%}")

    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
    if confidence > 0.8:
        st.success("🎉 High confidence prediction!")
    elif confidence > 0.6:
        st.warning("🤔 Medium confidence prediction.")
    else:
        st.info("😕 Low confidence prediction. Try a clearer image.")

# ---------------- INTERFACES ---------------- #
def upload_image_interface(model, preprocessor):
    st.header("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("🔍 Analyzing image..."):
                img_array = preprocessor.preprocess_image_pil(image)
                predicted_class, confidence = classify_image(model, img_array)
            display_prediction_results(predicted_class, confidence)

def webcam_interface(model, preprocessor):
    st.header("📷 Live Camera Mode")
    st.info("Use your device camera to capture an image.")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Read captured image
        image = Image.open(img_file_buffer).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        with st.spinner("🔍 Classifying..."):
            img_array = preprocessor.preprocess_image_pil(image)
            predicted_class, confidence = classify_image(model, img_array)
        display_prediction_results(predicted_class, confidence)

# ---------------- MAIN ---------------- #
def main():
    st.markdown('<h1 class="main-header">Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a photo or use your camera for instant results</p>', unsafe_allow_html=True)

    model = load_my_model()
    preprocessor = InferencePreprocessor((128, 128))  # adjust if your model uses other size

    method = st.radio(
        "Choose how you want to classify:",
        ["Upload Image", "Use Camera"],
        index=0,
        horizontal=True
    )

    if method == "Upload Image":
        upload_image_interface(model, preprocessor)
    else:
        webcam_interface(model, preprocessor)

if __name__ == "__main__":
    main()
