import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load the default pretrained ResNet50 model
model = ResNet50(weights='imagenet')

# --- Streamlit UI ---
st.title("ResNet50 Image Classifier (ImageNet)")
st.write("Upload an image and the model will predict its ImageNet class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image for ResNet50
    img = image.resize((224, 224), Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    decoded = decode_predictions(preds, top=3)[0]

    st.markdown("**Top Predictions:**")
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        st.markdown(f"{i+1}. **{label}** ({prob*100:.2f}%)") 