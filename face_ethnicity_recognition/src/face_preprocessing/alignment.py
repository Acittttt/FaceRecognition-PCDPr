# src/face_preprocessing/alignment.py

import cv2
import numpy as np
import dlib
from imutils import face_utils
import os

class FacePreprocessor:
    """
    Class for preprocessing face images: normalization, alignment, and preprocessing.
    """
    
    def __init__(self, target_size=(224, 224), face_detector=None, landmark_predictor_path=None):
        """
        Initialize the face preprocessor.
        
        Args:
            target_size (tuple): Target size for the output images (width, height)
            face_detector: Face detector object (if None, will use dlib's HOG detector)
            landmark_predictor_path (str): Path to dlib's facial landmark predictor model
        """
        self.target_size = target_size
        
        # Initialize face detector if not provided
        if face_detector is None:
            self.face_detector = dlib.get_frontal_face_detector()
        else:
            self.face_detector = face_detector
        
        # Initialize landmark predictor if path is provided
        if landmark_predictor_path and os.path.exists(landmark_predictor_path):
            self.landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
        else:
            print("Warning: Landmark predictor model not found. Face alignment might not work.")
            self.landmark_predictor = None
    
    def detect_face(self, image):
        """
        Detect faces in the image using dlib's HOG detector.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            list: List of dlib rectangles representing detected faces
        """
        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector(gray, 1)
        return faces
    
    def get_landmarks(self, image, face):
        """
        Get facial landmarks for a detected face.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            face (dlib.rectangle): Detected face rectangle
            
        Returns:
            numpy.ndarray: Array of facial landmarks (68 points)
        """
        if self.landmark_predictor is None:
            return None
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get landmarks
        shape = self.landmark_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        return shape
    
    def align_face(self, image, landmarks):
        """
        Align face based on eye landmarks.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            landmarks (numpy.ndarray): Facial landmarks
            
        Returns:
            numpy.ndarray: Aligned face image
        """
        if landmarks is None:
            return image
        
        # Get eye landmarks (assuming 68-point landmark model)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Calculate center of each eye
        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")
        
        # Calculate angle between eye centers
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate desired eye position
        # We want the eyes to be at 45% and 55% of the image width
        desired_left_eye_x = 0.45
        desired_right_eye_x = 0.55
        
        # Calculate desired eye distance
        desired_dist = desired_right_eye_x - desired_left_eye_x
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        scale = desired_dist * self.target_size[0] / dist
        
        # Calculate the center point between the eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Convert eyes_center to a tuple to ensure it's in the correct format
        eyes_center = tuple(map(int, eyes_center))
        
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update the translation component of the matrix
        tX = self.target_size[0] * 0.5
        tY = self.target_size[1] * 0.35  # Place eyes at 35% from the top
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply the affine transformation
        aligned_face = cv2.warpAffine(image, M, self.target_size,
                                    flags=cv2.INTER_CUBIC)
        
        return aligned_face
    
    def normalize_brightness(self, image):
        """
        Normalize the brightness of the image.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            numpy.ndarray: Brightness normalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE-enhanced L channel with the original A and B channels
        merged = cv2.merge([cl, a, b])
        
        # Convert back to BGR color space
        normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def preprocess_face(self, image, face=None):
        """
        Preprocess a face image: detect, align, normalize, and resize.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            face (dlib.rectangle, optional): Pre-detected face rectangle
            
        Returns:
            numpy.ndarray: Preprocessed face image
            dict: Additional information about the preprocessing
        """
        # Make a copy of the image
        img = image.copy()
        
        # Detect face if not provided
        if face is None:
            faces = self.detect_face(img)
            if len(faces) == 0:
                print("Warning: No face detected.")
                # Resize and return the original image
                resized = cv2.resize(img, self.target_size)
                return resized, {"success": False, "message": "No face detected"}
            
            # Use the first detected face
            face = faces[0]
        
        # Get landmarks
        landmarks = self.get_landmarks(img, face)
        
        # Align face if landmarks are available
        if landmarks is not None:
            aligned_face = self.align_face(img, landmarks)
        else:
            # If landmarks are not available, just crop the face
            x, y, w, h = (face.left(), face.top(), 
                          face.right() - face.left(), 
                          face.bottom() - face.top())
            aligned_face = img[y:y+h, x:x+w]
            if aligned_face.size == 0:  # Check if crop is valid
                aligned_face = img
            aligned_face = cv2.resize(aligned_face, self.target_size)
        
        # Normalize brightness
        normalized_face = self.normalize_brightness(aligned_face)
        
        return normalized_face, {"success": True, "landmarks": landmarks}
    
    def batch_process(self, image_paths, output_dir):
        """
        Process a batch of images and save them to output directory.
        
        Args:
            image_paths (list): List of image file paths
            output_dir (str): Directory to save processed images
            
        Returns:
            list: List of dictionaries with processing results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, img_path in enumerate(image_paths):
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    results.append({
                        "path": img_path,
                        "success": False,
                        "message": "Failed to read image"
                    })
                    continue
                
                # Process image
                processed_img, info = self.preprocess_face(img)
                
                # Generate output path
                basename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, basename)
                
                # Save processed image
                cv2.imwrite(output_path, processed_img)
                
                # Store result
                results.append({
                    "path": img_path,
                    "output_path": output_path,
                    "success": info["success"]
                })
                
            except Exception as e:
                results.append({
                    "path": img_path,
                    "success": False,
                    "message": str(e)
                })
        
        return results


# Example usage
if __name__ == "__main__":
    # Path to dlib's pre-trained facial landmark detector
    # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_path = "models/shape_predictor_68_face_landmarks.dat"
    
    # Initialize face preprocessor
    preprocessor = FacePreprocessor(target_size=(224, 224), 
                                  landmark_predictor_path=landmark_path)
    
    # Process a single image
    image = cv2.imread("data/raw/sample.jpg")
    if image is not None:
        processed_face, info = preprocessor.preprocess_face(image)
        
        # Display results
        cv2.imshow("Original", image)
        cv2.imshow("Processed", processed_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # # Process a batch of images
    # image_paths = ["data/raw/sample1.jpg", "data/raw/sample2.jpg"]
    # results = preprocessor.batch_process(image_paths, "data/processed/aligned")
    # for result in results:
    #     print(result)