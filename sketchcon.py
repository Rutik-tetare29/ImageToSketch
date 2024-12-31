
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# Function to convert an image to a sketch
def image_to_sketch(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur to the inverted image
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred)
    
    # Create the pencil sketch
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    return sketch

# Streamlit app title and description
st.title("Image to Sketch Converter")
st.write("Upload an image, and the app will convert it into a pencil sketch!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Convert to sketch
    sketch = image_to_sketch(image)

    # Display the sketch
    st.image(sketch, caption="Pencil Sketch", use_column_width=True, channels="GRAY")

    # Save sketch to a BytesIO buffer
    buffer = BytesIO()
    result = Image.fromarray(sketch)
    result.save(buffer, format="PNG")  # Save the image in PNG format
    buffer.seek(0)

    # Download option
    st.download_button(
        label="Download Sketch",
        data=buffer,
        file_name="sketch.png",
        mime="image/png",
    )

