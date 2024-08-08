import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model("/content/model3.h5")

# Define the class labels
class_labels = ['COVID-19', 'NORMAL', 'VIRAL PNEUMONIA']

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256, 3))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
def main():
    st.title("Lung Disease Classification")
    st.write("Upload an X-ray image to classify the lung condition.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        img = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(img)
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the result
        st.write(f"The predicted lung condition is: {predicted_class}")

if __name__ == "__main__":
    main()