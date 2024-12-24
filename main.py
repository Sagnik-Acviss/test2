import streamlit as st

import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# Function to extract features from a single image
def extract_features_single(img, model):
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        features = model.predict(img_array)  # Predict features
        return features.flatten()
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to check if an image is present in the stored features
def is_image_present(image_features, stored_features, threshold=0.8):
    try:
        max_similarity = 0
        for features in stored_features.values():
            similarity = cosine_similarity([image_features], [features])[0][0]
            max_similarity = max(max_similarity, similarity)

        is_present = max_similarity >= threshold
        return is_present, max_similarity
    except Exception as e:
        st.error(f"Error checking similarity: {e}")
        return False, 0

# Streamlit app definition
def main():
    st.title("Image Presence Checker")
    st.write("Upload a .pkl file with stored features and one or more images to check for similarity.")

    # Upload stored features file
    uploaded_pkl = st.file_uploader("Upload .pkl file", type="pkl")

    # Upload image file(s)
    uploaded_images = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Set similarity threshold
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

    if uploaded_pkl and uploaded_images:
        try:
            # Load the stored features
            stored_features = pickle.load(uploaded_pkl)

            # Load the pre-trained ResNet50 model
            model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

            results = []
            for uploaded_image in uploaded_images:
                # Extract features for the uploaded image
                image_data = image.load_img(uploaded_image)
                image_features = extract_features_single(image_data, model)

                if image_features is not None:
                    # Check similarity against stored features
                    is_present, similarity = is_image_present(image_features, stored_features, threshold)
                    results.append((uploaded_image.name, is_present, similarity))

            # Display results
            st.write("### Results:")
            for filename, is_present, similarity in results:
                st.write(f"- **Image:** {filename}")
                st.write(f"  - **Is Present:** {'Yes' if is_present else 'No'}")
                st.write(f"  - **Highest Similarity:** {similarity:.4f}")

        except Exception as e:
            st.error(f"Error processing files: {e}")

if __name__ == "__main__":
    main()
