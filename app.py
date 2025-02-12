import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('my_model.keras')

# Streamlit app
st.title("Digital Fraud Product Detection")
st.write("Upload an image to detect if it's a fake or genuine product.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Predict the label
    label = model.predict(img_array)

    # Determine the predicted class
    class_names = ['Fake', 'Genuine']
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]

    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class}")

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Plot the image with the predicted label
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(predicted_class)
    ax.axis('off')
    st.pyplot(fig)
