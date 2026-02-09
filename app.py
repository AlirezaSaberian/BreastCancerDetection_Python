import streamlit as st
from PIL import Image
import numpy as np
import os
import random
from keras.models import model_from_json
from io import BytesIO
import base64
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

st.markdown("""
    <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1 {
            text-align: center;
            color: #2c3e50;
        }

        .stButton>button {
            width: 100%;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: #ffffff;
            padding: 0.6em;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }

        .stButton>button:hover {
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.5);
            transform: scale(1.02);
        }

        .uploaded-img img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        .footer {
            text-align: center;
            font-size: 0.90em;
            color: #aaa;
            margin-top: 25px;
        }

        .stButton {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model.weights.h5")
    return model

model = load_model()

DATASET_DIR = os.path.join("image_processing", "normal", "model_tst", "trainig")

def get_random_image_from_test():
    images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    return random.choice(images)

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.title("🧬 Breast Cancer Detection from Tissue Image")
st.markdown("<div style='text-align: center; margin-bottom: 20px;'>Upload an image or use sample images for testing 👇</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload 50x50 image (jpg/png/jpeg):", type=["jpg", "png", "jpeg"])

test_image_path = None

if st.button("🎲 Random Image from Test Set", use_container_width=True):
    test_image_path = get_random_image_from_test()

if uploaded_file or test_image_path:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB").resize((25, 25))
        caption = "Uploaded Image"
    else:
        image = Image.open(test_image_path).convert("RGB").resize((25, 25))
        caption = f"Sample Image: {os.path.basename(test_image_path)}"


    img_str = img_to_base64(image)
    st.markdown(f"""
        <div class="uploaded-img">
            <img src="data:image/png;base64,{img_str}" width="300" alt="Uploaded Image" />
            <div style="text-align:center; color:#555; margin-top: 8px; font-size: 0.75rem;">{caption}</div>
        </div>
    """, unsafe_allow_html=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    confidence = float(np.max(prediction)) * 100
    label = np.argmax(prediction)
    

    st.markdown("---")
    st.subheader("🔬 Prediction Result")
    if label == 1:
        st.error(f"⚠️ **Cancerous cells detected!**\n\n🧪 **Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

    else:
        st.success(f"✅ **No signs of cancer.**\n\n🧪 **Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))


else:
    st.info("⬆️ Please upload an image or use sample images.")

st.markdown("""
<hr>
<div class="footer">Shahrekord University<br>
Professor Advisor: Dr.Abbas Horri - Student: Alireza Saberian<br>
Summer 1404
</div>
""", unsafe_allow_html=True)
