import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# Function to convert an image to a sketch
def image_to_sketch(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return sketch

# Function to convert an image to grayscale
def image_to_grayscale(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return gray

# Function to apply Gaussian blur to an image
def apply_blur(image):
    blurred = cv2.GaussianBlur(np.array(image), (15, 15), 0)
    return blurred

# Function for edge detection using Canny
def edge_detection(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Function to apply cartoon effect
def cartoonize_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# Streamlit app title and description
st.title("Image Processing App")
st.write("Upload an image and apply various effects like sketch, grayscale, blur, edge detection, or cartoon effect!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Select an effect to apply:")

    # Buttons for each effect
    processed_image = None
    if st.button("Convert to Sketch"):
        processed_image = image_to_sketch(image)
        st.image(processed_image, caption="Pencil Sketch", use_column_width=True, channels="GRAY")

    if st.button("Convert to Grayscale"):
        processed_image = image_to_grayscale(image)
        st.image(processed_image, caption="Grayscale Image", use_column_width=True, channels="GRAY")

    if st.button("Apply Blur"):
        processed_image = apply_blur(image)
        st.image(processed_image, caption="Blurred Image", use_column_width=True)

    if st.button("Edge Detection"):
        processed_image = edge_detection(image)
        st.image(processed_image, caption="Edge Detected Image", use_column_width=True, channels="GRAY")

    if st.button("Cartoonize Image"):
        processed_image = cartoonize_image(image)
        st.image(processed_image, caption="Cartoonized Image", use_column_width=True)

    # Option to download the last processed image
    if processed_image is not None:
        buffer = BytesIO()
        result = Image.fromarray(processed_image if len(processed_image.shape) == 2 else cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        result.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="Download Processed Image",
            data=buffer,
            file_name="processed_image.png",
            mime="image/png",
        )
