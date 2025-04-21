# src/face_preprocessing/normalization.py

import cv2
import numpy as np

class ImageNormalizer:
    """
    Class for normalizing face images with various techniques.
    """
    
    def __init__(self):
        """
        Initialize the image normalizer.
        """
        pass
    
    def normalize_size(self, image, target_size=(224, 224)):
        """
        Resize an image to target size.
        
        Args:
            image (numpy.ndarray): Input image
            target_size (tuple): Target size (width, height)
            
        Returns:
            numpy.ndarray: Resized image
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_color(self, image):
        """
        Normalize image colors using histogram equalization in LAB color space.
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Color normalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge([cl, a, b])
        
        # Convert back to BGR
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    def normalize_illumination(self, image):
        """
        Normalize image illumination using Retinex algorithm.
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Illumination normalized image
        """
        # Simple Single-Scale Retinex (SSR)
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Apply log transform
        log_img = np.log1p(img_float)
        
        # Apply Gaussian blur
        blur_img = cv2.GaussianBlur(img_float, (0, 0), sigmaX=10)
        blur_img = np.log1p(blur_img)
        
        # Subtract the log of the blurred image from the log of the original
        retinex_img = log_img - blur_img
        
        # Scale the result to [0, 1]
        retinex_img = (retinex_img - np.min(retinex_img)) / (np.max(retinex_img) - np.min(retinex_img))
        
        # Convert back to 8-bit
        retinex_img = (retinex_img * 255).astype(np.uint8)
        
        return retinex_img
    
    def gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction to an image.
        
        Args:
            image (numpy.ndarray): Input BGR image
            gamma (float): Gamma value
            
        Returns:
            numpy.ndarray: Gamma corrected image
        """
        # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    
    def normalize_contrast(self, image):
        """
        Normalize image contrast using adaptive histogram equalization.
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Contrast normalized image
        """
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Split channels
        y, cr, cb = cv2.split(ycrcb)
        
        # Apply CLAHE to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)
        
        # Merge channels
        ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
        
        # Convert back to BGR
        return cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)
    
    def normalize_white_balance(self, image):
        """
        Apply white balance correction using the Gray World algorithm.
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: White balanced image
        """
        # Split channels
        b, g, r = cv2.split(image)
        
        # Calculate averages
        r_avg = np.mean(r)
        g_avg = np.mean(g)
        b_avg = np.mean(b)
        
        # Find the average of averages
        avg = (r_avg + g_avg + b_avg) / 3
        
        # Calculate scaling factors
        r_scale = avg / r_avg if r_avg > 0 else 1
        g_scale = avg / g_avg if g_avg > 0 else 1
        b_scale = avg / b_avg if b_avg > 0 else 1
        
        # Apply scaling to each channel
        r = cv2.convertScaleAbs(r, alpha=r_scale)
        g = cv2.convertScaleAbs(g, alpha=g_scale)
        b = cv2.convertScaleAbs(b, alpha=b_scale)
        
        # Merge channels
        balanced = cv2.merge([b, g, r])
        
        return balanced
    
    def apply_all_normalizations(self, image, target_size=(224, 224)):
        """
        Apply all normalization techniques to an image.
        
        Args:
            image (numpy.ndarray): Input BGR image
            target_size (tuple): Target size (width, height)
            
        Returns:
            numpy.ndarray: Fully normalized image
        """
        # Resize
        normalized = self.normalize_size(image, target_size)
        
        # White balance
        normalized = self.normalize_white_balance(normalized)
        
        # Contrast normalization
        normalized = self.normalize_contrast(normalized)
        
        # Color normalization
        normalized = self.normalize_color(normalized)
        
        return normalized


# Example usage
if __name__ == "__main__":
    # Initialize normalizer
    normalizer = ImageNormalizer()
    
    # Load an image
    image = cv2.imread("data/raw/sample.jpg")
    if image is not None:
        # Apply all normalizations
        normalized = normalizer.apply_all_normalizations(image)
        
        # Display results
        cv2.imshow("Original", image)
        cv2.imshow("Normalized", normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        cv2.imwrite("data/processed/normalized_sample.jpg", normalized)