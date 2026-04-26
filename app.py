import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os
from tensorflow.keras.models import load_model

# download model
if not os.path.exists("model.h5"):
    url = "https://drive.google.com/uc?id=1y52Xofvr40zKXRKyP90U3iVuQGcsGLqa"
    gdown.download(url, "model.h5", quiet=False)

# load model
model = load_model("model.h5")


st.title("🧠 Brain Tumor Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("❌ Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")