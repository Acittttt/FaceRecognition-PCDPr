# app/streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our preprocessing modules
from src.face_preprocessing.alignment import FacePreprocessor
from src.face_preprocessing.normalization import ImageNormalizer

# Set page configuration
st.set_page_config(
    page_title="Face Preprocessing Demo",
    page_icon="👤",
    layout="wide"
)

@st.cache_resource
def load_face_detector():
    """Load dlib's face detector"""
    return dlib.get_frontal_face_detector()

@st.cache_resource
def load_landmark_predictor(path):
    """Load dlib's facial landmark predictor"""
    if os.path.exists(path):
        return dlib.shape_predictor(path)
    else:
        st.error(f"Landmark predictor model not found at {path}")
        return None

def main():
    st.title("Face Preprocessing Demo")
    st.write("""
    This application demonstrates the face preprocessing pipeline for the Face Recognition and Ethnicity Detection System.
    Upload an image containing a face to see the preprocessing steps in action.
    """)
    
    # Sidebar for configuration options
    st.sidebar.title("Configuration")
    
    # Path to landmark predictor model
    landmark_path = st.sidebar.text_input(
        "Path to landmark predictor model",
        "models/shape_predictor_68_face_landmarks.dat"
    )
    
    # Target size for processed images
    target_size = st.sidebar.selectbox(
        "Target Image Size",
        [(224, 224), (160, 160), (96, 96), (299, 299)],
        format_func=lambda x: f"{x[0]}x{x[1]}"
    )
    
    # Preprocessing options
    st.sidebar.subheader("Preprocessing Options")
    do_alignment = st.sidebar.checkbox("Face Alignment", value=True)
    do_color_norm = st.sidebar.checkbox("Color Normalization", value=True)
    do_brightness_norm = st.sidebar.checkbox("Brightness Normalization", value=True)
    do_contrast_norm = st.sidebar.checkbox("Contrast Normalization", value=True)
    do_white_balance = st.sidebar.checkbox("White Balance", value=True)
    
    # Initialize preprocessing objects
    try:
        face_detector = load_face_detector()
        landmark_predictor = load_landmark_predictor(landmark_path)
        
        preprocessor = FacePreprocessor(
            target_size=target_size,
            face_detector=face_detector,
            landmark_predictor_path=landmark_path if os.path.exists(landmark_path) else None
        )
        
        normalizer = ImageNormalizer()
    except Exception as e:
        st.error(f"Error initializing preprocessing tools: {e}")
        return
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Read the image
        image = cv2.imread(tmp_path)
        if image is None:
            st.error("Failed to read the uploaded image.")
            return
        
        # Display original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        # Detect faces
        faces = preprocessor.detect_face(image)
        if len(faces) == 0:
            st.error("No faces detected in the image.")
            return
        
        # Create a copy of the image to draw on
        detected_image = image.copy()
        
        # Draw rectangles around detected faces
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected_image, f"Face {i+1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Get landmarks and draw them
            landmarks = preprocessor.get_landmarks(image, face)
            if landmarks is not None:
                for (x, y) in landmarks:
                    cv2.circle(detected_image, (x, y), 2, (0, 0, 255), -1)
        
        # Display image with face detection
        st.subheader("Face Detection")
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        # Select which face to process if there are multiple
        selected_face_idx = 0
        if len(faces) > 1:
            selected_face_idx = st.selectbox(
                "Select a face to process:",
                range(len(faces)),
                format_func=lambda x: f"Face {x+1}"
            )
        
        selected_face = faces[selected_face_idx]
        
        # Create columns for the preprocessing steps
        col1, col2 = st.columns(2)
        
        # 1. Aligned face
        aligned_face = None
        if do_alignment:
            landmarks = preprocessor.get_landmarks(image, selected_face)
            if landmarks is not None:
                try:
                    aligned_face = preprocessor.align_face(image, landmarks)
                    with col1:
                        st.subheader("Face Alignment")
                        st.image(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
                except Exception as e:
                    st.error(f"Error during face alignment: {e}")
                    # Fallback to cropping
                    x, y, w, h = (selected_face.left(), selected_face.top(),
                                  selected_face.right() - selected_face.left(),
                                  selected_face.bottom() - selected_face.top())
                    aligned_face = image[y:y+h, x:x+w]
                    if aligned_face.size == 0:  # Check if crop is valid
                        aligned_face = image
                    aligned_face = cv2.resize(aligned_face, target_size)
                    with col1:
                        st.subheader("Cropped Face (Alignment Failed)")
                        st.image(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
            else:
                st.warning("Could not detect facial landmarks for alignment.")
                # Just crop the face if landmarks are not available
                x, y, w, h = (selected_face.left(), selected_face.top(),
                              selected_face.right() - selected_face.left(),
                              selected_face.bottom() - selected_face.top())
                aligned_face = image[y:y+h, x:x+w]
                if aligned_face.size == 0:  # Check if crop is valid
                    aligned_face = image
                aligned_face = cv2.resize(aligned_face, target_size)
                with col1:
                    st.subheader("Cropped Face (No Landmarks)")
                    st.image(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        else:
            # Just crop the face if alignment is disabled
            x, y, w, h = (selected_face.left(), selected_face.top(),
                          selected_face.right() - selected_face.left(),
                          selected_face.bottom() - selected_face.top())
            aligned_face = image[y:y+h, x:x+w]
            if aligned_face.size == 0:  # Check if crop is valid
                aligned_face = image
            aligned_face = cv2.resize(aligned_face, target_size)
            with col1:
                st.subheader("Cropped Face")
                st.image(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        # 2. Apply normalization steps
        normalized_face = aligned_face.copy()
        
        # Apply different normalization steps based on user selection
        if do_white_balance:
            normalized_face = normalizer.normalize_white_balance(normalized_face)
            with col2:
                st.subheader("After White Balance")
                st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        if do_contrast_norm:
            normalized_face = normalizer.normalize_contrast(normalized_face)
            with col1:
                st.subheader("After Contrast Normalization")
                st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        if do_color_norm:
            normalized_face = normalizer.normalize_color(normalized_face)
            with col2:
                st.subheader("After Color Normalization")
                st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        if do_brightness_norm:
            # Apply gamma correction with a default gamma of 1.2
            gamma = st.sidebar.slider("Gamma Value", 0.5, 2.0, 1.2, 0.1)
            normalized_face = normalizer.gamma_correction(normalized_face, gamma)
            with col1:
                st.subheader(f"After Gamma Correction (γ={gamma})")
                st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        # Final result
        st.header("Final Preprocessed Result")
        st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), width=300)
        
        # Side-by-side comparison
        st.header("Before and After Comparison")
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.subheader("Original")
            # Get the face from the original image
            x, y, w, h = (selected_face.left(), selected_face.top(),
                         selected_face.right() - selected_face.left(),
                         selected_face.bottom() - selected_face.top())
            original_face = image[y:y+h, x:x+w]
            if original_face.size == 0:  # Check if crop is valid
                original_face = image
            original_face = cv2.resize(original_face, target_size)
            st.image(cv2.cvtColor(original_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        with col_after:
            st.subheader("Preprocessed")
            st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), width=None)  # Changed here
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Add option to download processed image
        if st.button("Download Preprocessed Image"):
            # Save the preprocessed image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, normalized_face)
                tmp_download_path = tmp.name
            
            # Read the file as bytes
            with open(tmp_download_path, "rb") as file:
                btn = st.download_button(
                    label="Click to download",
                    data=file,
                    file_name="preprocessed_face.jpg",
                    mime="image/jpeg"
                )
            
            # Clean up the temporary download file
            os.unlink(tmp_download_path)

if __name__ == "__main__":
    main()