# ============================================================
# STREAMLIT GUI FOR YOLOv8 OBJECT DETECTION
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# ============================================================
# Page settings
# ============================================================
st.set_page_config(page_title="Vehicle Detection App", layout="wide")
st.title("Vehicle Detection using YOLOv8")
st.write("Choose an image source from the sidebar and detect vehicles using your trained YOLOv8 model.")

# ============================================================
# Load model only once
# ============================================================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# ============================================================
# Sidebar settings
# ============================================================
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

source_option = st.sidebar.radio(
    "Choose Input Source",
    ["Upload Picture", "Capture Webcam Picture"]
)

# ============================================================
# Function to run prediction and show class and confidence
# ============================================================
def show_prediction(image):
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # ========================================================
    # Run prediction
    # ========================================================
    results = model.predict(source=img_array, conf=conf_threshold)

    # Plot detected output image
    plotted_img = results[0].plot()

    with col2:
        st.subheader("Detected Output")
        st.image(plotted_img, use_container_width=True)

    # ========================================================
    # Show prediction and confidence
    # ========================================================
    st.subheader("Prediction Results")

    boxes = results[0].boxes
    names = results[0].names

    if boxes is None or len(boxes) == 0:
        st.warning("No objects detected.")
    else:
        detected_rows = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            detected_rows.append({
                "Prediction": names[cls_id],
                "Confidence": round(conf, 4)
            })

        df = pd.DataFrame(detected_rows)
        st.dataframe(df, use_container_width=True)

# ============================================================
# Option 1: Upload picture
# ============================================================
if source_option == "Upload Picture":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        show_prediction(image)
    else:
        st.info("Please upload an image to start detection.")

# ============================================================
# Option 2: Capture webcam picture
# ============================================================
elif source_option == "Capture Webcam Picture":
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        show_prediction(image)
    else:
        st.info("Please capture an image to start detection.")