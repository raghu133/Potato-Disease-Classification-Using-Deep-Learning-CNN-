import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
MODEL_PATH = r"C:/Users/91984/Documents/models/potato_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (update these according to your dataset)
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Streamlit UI
st.title("🥔 Potato Disease Classification")
st.write("Upload an image of a potato leaf to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Match your IMAGE_SIZE
    img_array = np.array(img) / 255.0  # Rescaling
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Output
    st.markdown(f"**Prediction:** {predicted_class}")
    st.markdown(f"**Confidence:** {confidence}%")