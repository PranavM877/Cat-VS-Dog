import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import cv2
import os
import gdown

# Google Drive file IDs for the model and pipeline
MODEL_FILE_ID = "1rdLZBi7S_BIDfUbljvbLMrHySnoTKJbo"
PKL_FILE_ID = "1q5JcnZqeolsnrPyvxolgqyu8THdwRBt-"

MODEL_FILE = "dog_cat_model.h5"
PKL_FILE = "preprocess_pipeline.pkl"

def download_file(file_id, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} from Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)

st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐕🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class InferencePreprocessor:
    def __init__(self, img_size):
        self.img_size = img_size

    def preprocess_image_pil(self, img):
        img_resized = img.resize(self.img_size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def preprocess_opencv_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        return frame_batch

@st.cache_resource
def load_model_and_config():
    download_file(MODEL_FILE_ID, MODEL_FILE)
    download_file(PKL_FILE_ID, PKL_FILE)
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        with open(PKL_FILE, 'rb') as f:
            config = pickle.load(f)
        return model, config
    except Exception as e:
        st.error(f"Error loading model or config: {e}")
        return None, None

def classify_image(model, preprocessor, img_array):
    try:
        prediction = model.predict(img_array, verbose=0)[0][0]
        if prediction > 0.5:
            return 'Dog', prediction
        else:
            return 'Cat', 1 - prediction
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None

def display_prediction_results(predicted_class, confidence):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="🎯 Prediction", value=predicted_class)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="🎲 Confidence", value=f"{confidence:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.progress(confidence)
    if confidence > 0.8:
        st.success("🎉 High confidence prediction!")
    elif confidence > 0.6:
        st.warning("🤔 Medium confidence prediction.")
    else:
        st.info("😕 Low confidence prediction. Consider clearer image.")

def upload_image_interface(model, config, preprocessor):
    st.header("📤 Upload Image Classification")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.info(f"Image size: {image.size}")
        with col2:
            with st.spinner("Analyzing image..."):
                img_array = preprocessor.preprocess_image_pil(image)
                predicted_class, confidence = classify_image(model, preprocessor, img_array)
            if predicted_class:
                st.markdown("### 🎯 Classification Results")
                display_prediction_results(predicted_class, confidence)

def webcam_interface(model, config, preprocessor):
    st.header("📷 Live Webcam Classification")
    st.info("Click 'Start Webcam' to begin, then 'Capture & Classify' to analyze the current frame")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_webcam = st.button("Start Webcam", type="primary")
    with col2:
        stop_webcam = st.button("Stop Webcam")
    with col3:
        capture_classify = st.button("Capture & Classify")
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if start_webcam:
        st.session_state.webcam_active = True
    if stop_webcam:
        st.session_state.webcam_active = False
    if st.session_state.webcam_active:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam. Please check your camera connection.")
                st.session_state.webcam_active = False
                return
            video_placeholder = st.empty()
            results_placeholder = st.empty()
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                if capture_classify:
                    with st.spinner("Classifying current frame..."):
                        img_array = preprocessor.preprocess_opencv_frame(frame)
                        predicted_class, confidence = classify_image(model, preprocessor, img_array)
                    if predicted_class:
                        with results_placeholder.container():
                            st.markdown("### Live Classification Results")
                            display_prediction_results(predicted_class, confidence)
                            st.info(f"Frame shape: {frame.shape} → Processed: {img_array.shape}")
                import time
                time.sleep(0.03)
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            st.error(f"Webcam error: {e}")
            st.session_state.webcam_active = False
    else:
        st.info("Webcam is currently stopped. Click 'Start Webcam' to begin.")

def main():
    st.markdown('<h1 class="main-header">🐕🐱 Dog vs Cat Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    model, config = load_model_and_config()
    if model is None or config is None:
        st.stop()
    preprocessor = InferencePreprocessor(config['IMG_SIZE'])
    with st.sidebar:
        st.title("Control Panel")
        method = st.radio("Choose Classification Method:", ["Upload Image", "Use Webcam"])
        st.markdown("---")
        st.markdown("### Model Information")
        st.info(f"Input Size: {config['IMG_SIZE']}")
        st.info("Classes: Cat, Dog")
        st.info("Architecture: CNN")
        st.info("Dataset: opendatasets download")
        if 'class_indices' in config:
            st.markdown("### Class Mapping")
            for cname, idx in config['class_indices'].items():
                st.write(f"• {cname}: {idx}")
        st.markdown("---")
        st.markdown("### Dataset Information")
        st.info("Source: Kaggle (anthonytherrien/dog-vs-cat)")
        st.info("Downloaded via: opendatasets library")
        st.info("Split: 80% train, 20% test")
    if method == "Upload Image":
        upload_image_interface(model, config, preprocessor)
    elif method == "Use Webcam":
        webcam_interface(model, config, preprocessor)
    st.markdown("---")
    st.markdown("""ℹ️ About this classifier:
- Trained on dog and cat images with 80-20 train/test split
- Uses different preprocessing pipeline for inference than training
- Model saved in HDF5 format, preprocessing config in pickle
- Dataset downloaded using opendatasets library
- Built with TensorFlow and Streamlit""")
    with st.expander("Technical Details"):
        st.write("Training Pipeline: ImageDataGenerator with data augmentation")
        st.write("Inference Pipeline: Custom InferencePreprocessor class")
        st.write("Model Format: TensorFlow SavedModel (.h5)")
        st.write("Config Format: Pickle (.pkl)")
        st.write("Dataset Source: Kaggle via opendatasets")
        st.write("Framework: TensorFlow + Streamlit + OpenCV")
        st.write("Model Download: Google Drive, at runtime (using gdown)")

if __name__ == "__main__":
    main()
