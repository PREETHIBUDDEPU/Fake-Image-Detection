import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from streamlit_lottie import st_lottie
import requests

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="🕵‍♂ Fake & Real Image Detector", page_icon="🖼", layout="centered")

# ==========================
# Custom CSS Styles
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
    font-family: 'Poppins', sans-serif;
}

h1, h2, h3 {
    text-align: center;
    font-weight: 600;
}

h1 {
    font-size: 2.2em;
    background: -webkit-linear-gradient(45deg, #00b4d8, #90e0ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Upload box */
div[data-testid="stFileUploader"] {
    border: 2px dashed #00b4d8;
    border-radius: 15px;
    padding: 20px;
    background-color: rgba(255,255,255,0.05);
}

/* Image style */
img {
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

/* Result card */
.result-card {
    border-radius: 15px;
    padding: 25px;
    margin-top: 30px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
    animation: fadeIn 0.8s ease-in-out;
}

.result-card:hover {
    transform: scale(1.05);
}

/* Fade animation */
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}

/* Confidence bar */
.stProgress > div > div {
    background-color: #00b4d8;
    border-radius: 10px;
}

/* Footer */
.footer {
    margin-top: 50px;
    text-align: center;
    color: #bbb;
    font-size: 14px;
}
.footer a {
    color: #00b4d8;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Lottie Loader
# ==========================
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_j1adxtyb.json")

# ==========================
# Cached Model Loader
# ==========================
@st.cache_resource
def load_model_cached():
    return load_model("my_model.keras", compile=False, safe_mode=True)

model = load_model_cached()

# ==========================
# Prediction Function
# ==========================
def predict_image(img, model, img_size=(224, 224)):
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prob = model.predict(img_array, verbose=0)[0][0]

        if prob > 0.5:
            label = "🖼 Real ✅"
            confidence = prob
        else:
            label = "🤖 Fake ❌"
            confidence = 1 - prob

        return label, confidence

    except Exception as e:
        st.error(f"⚠ Error processing image: {str(e)}")
        return None, None

# ==========================
# Tabs Layout
# ==========================
tab1, tab2 = st.tabs(["🔍 Detection", "📘 About the App"])

# ==========================
# Detection Tab
# ==========================
with tab1:
    st.title("🕵‍♂ Fake & Real Image Detector")
    st_lottie(lottie_ai, height=180, key="ai")

    st.markdown("""
    <div style="text-align:center; margin-bottom:30px;">
        <h3>Upload an image to check if it’s Real or AI-generated 🎨</h3>
        <p style="color:#ccc; font-size:16px;">
            Powered by a Deep Learning model trained on thousands of artworks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(img, caption="🖼 Uploaded Image", width=350)

        with col2:
            label, confidence = predict_image(img, model)
            if label is not None:
                # Choose colors dynamically
                if "Real" in label:
                    bg_color = "rgba(40, 167, 69, 0.2)"
                    border_color = "#28a745"
                else:
                    bg_color = "rgba(220, 53, 69, 0.2)"
                    border_color = "#dc3545"

                st.markdown(
                    f"""
                    <div class="result-card" style="background-color:{bg_color}; border: 2px solid {border_color};">
                        🔎 Prediction: {label}<br><br>
                        <span style="font-size:16px; color:#ccc;">Confidence: {confidence:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(int(confidence * 100))

# ==========================
# About Tab
# ==========================
with tab2:
    st.subheader("📘 About This App")
    st.write("""
    This application uses a **Convolutional Neural Network (CNN)** trained on a dataset of real and AI-generated artworks.  
    The model identifies subtle differences in **texture, brush patterns, and color gradients** that often distinguish human art from AI outputs.  

    **Tech Stack:**  
    - 🧠 TensorFlow / Keras  
    - 📊 Streamlit for interactive UI  
    - 💾 ResNet50-based architecture  
    - 🎨 Custom-trained model weights  

    ---
    **Features:**
    - Upload and analyze any image (JPG, JPEG, PNG)
    - High-accuracy prediction with confidence score
    - Elegant, modern, and fully responsive interface

    ---
    """)

    st.success("💡 Tip: For best results, use high-resolution images with clear textures.")

# ==========================
# Footer
# ==========================
st.markdown("""
<div class="footer">
    ✨ Built with ❤️ using Streamlit & TensorFlow ✨<br>
    <a href="https://github.com/yourprofile" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/yourprofile/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
