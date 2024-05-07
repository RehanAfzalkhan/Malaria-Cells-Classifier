import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("malaria_detector.h5")


def preprocess_image(image):
    img = cv2.resize(image, (130, 130))
    img_array = img.astype("float32") / 255  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    prediction_score = prediction[0][0]  # Get the prediction score for the first class
    if prediction_score > 0.5:
        label = "Uninfected"
        accuracy = prediction_score * 100
    else:
        label = "Parasitized"
        accuracy = (1 - prediction_score) * 100
    return label, accuracy


def main():
    st.title("Malaria Cell Image Classifier")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict the class and accuracy
        predicted_class, accuracy = predict_image(image)

        st.write("Class:", predicted_class)
        st.write("Accuracy:", f"{accuracy:.2f}%")


if __name__ == "__main__":
    main()
