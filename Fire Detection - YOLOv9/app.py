import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os

# Set up directories
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Title and description
st.title("Fire Detection Web App")
st.write("Upload an image and detect fire using YOLOv9 model hosted on Roboflow.")


# Load Roboflow model
@st.cache_resource
def load_model():
    rf = Roboflow(api_key="dlURMCVFA9rzEFf6fQjn")  # Replace with your API key
    project = rf.workspace().project("fire-d6yfv")
    model = project.version(1).model
    return model


model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    with st.spinner("Detecting fire..."):
        result_image_path = os.path.join(RESULT_FOLDER, "result_" + uploaded_file.name)
        prediction = model.predict(image_path, confidence=20)
        prediction.save(result_image_path)

    # Show result
    st.success("Detection complete!")
    st.image(result_image_path, caption="Detected Fire", use_container_width=True)

    # Option to download result
    with open(result_image_path, "rb") as file:
        btn = st.download_button(
            label="Download Result",
            data=file,
            file_name="detected_fire.jpg",
            mime="image/jpg"
        )