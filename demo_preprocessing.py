# scripts/demo_preprocessing.py

import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from src.alignment import FacePreprocessor
from src.normalization import ImageNormalizer

def display_results(original, detected, aligned, normalized):
    """
    Display preprocessing results in a 2x2 grid.
    
    Args:
        original (numpy.ndarray): Original image
        detected (numpy.ndarray): Image with face detection
        aligned (numpy.ndarray): Aligned face
        normalized (numpy.ndarray): Fully normalized face
    """
    plt.figure(figsize=(12, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Face detection
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    plt.title('Face Detection')
    plt.axis('off')
    
    # Aligned face
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    plt.title('Aligned Face')
    plt.axis('off')
    
    # Normalized face
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    plt.title('Normalized Face')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Preprocessing Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--landmark_model', type=str, default='models/shape_predictor_68_face_landmarks.dat',
                        help='Path to dlib\'s facial landmark predictor model')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output images')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image '{args.image}'")
        return
    
    # Initialize preprocessor and normalizer
    preprocessor = FacePreprocessor(target_size=(224, 224), 
                                  landmark_predictor_path=args.landmark_model)
    normalizer = ImageNormalizer()
    
    # Detect faces
    faces = preprocessor.detect_face(image)
    if len(faces) == 0:
        print("No faces detected in the image.")
        return
    
    # Create a copy of the image to draw on
    detected_image = image.copy()
    
    # Draw rectangles around detected faces
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get landmarks for the first face
        landmarks = preprocessor.get_landmarks(image, face)
        
        # Draw landmarks if available
        if landmarks is not None:
            for (x, y) in landmarks:
                cv2.circle(detected_image, (x, y), 2, (0, 0, 255), -1)
    
    # Process the first face
    face = faces[0]
    aligned_face, info = preprocessor.preprocess_face(image, face)
    
    # Apply full normalization
    normalized_face = normalizer.apply_all_normalizations(aligned_face)
    
    # Save output images
    basename = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_detected.jpg"), detected_image)
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_aligned.jpg"), aligned_face)
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_normalized.jpg"), normalized_face)
    
    # Display results
    display_results(image, detected_image, aligned_face, normalized_face)

if __name__ == "__main__":
    main()