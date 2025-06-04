import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Define your category mapping ---
category_to_index = {
    'metal': 0,
    'glass': 1,
    'cardboard': 2,
    'trash': 3,
    'paper': 4,
    'plastic': 5
}
index_to_category = {
    0: 'metal',
    1: 'glass',
    2: 'cardboard',
    3: 'trash',
    4: 'paper',
    5: 'plastic'
}

# --- Load the model ---
model = tf.keras.models.load_model('best_model.h5')
# --- For SavedModel directory (TFSMLayer), use this instead: ---
# model = TFSMLayer('my_model_saved', call_endpoint='serving_default')

# --- Streamlit UI ---
st.title("Waste Category Classifier")
st.write("Upload an image and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = image.resize((300, 300), Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0).astype(np.float32)  # This creates shape (1, 300, 300, 3)

     # --- Predict ---
    preds = model.predict(img)
    # --- For TFSMLayer (SavedModel), use this instead: ---
    # preds = model(img)
    # preds = preds['output_0'].numpy()

    pred_class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_class_idx] * 100

    pred_class_name = index_to_category.get(pred_class_idx, "Unknown")

    # --- Add container recommendation ---
    if pred_class_name in ["metal", "plastic"]:
        recommendation = "Debes depositar esto en el contenedor amarillo"
    elif pred_class_name == "glass":
        recommendation = "Debes depositar esto en el contenedor verde"
    elif pred_class_name in ["paper", "cardboard"]:
        recommendation = "Debes depositar esto en el contenedor azul"
    else:
        recommendation = "Debes depositar esto en el contenedor gris"

    st.markdown(f"**Prediction:** {pred_class_name}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown(f"**{recommendation}**")