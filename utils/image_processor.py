import cv2
import numpy as np
import os
from typing import Tuple, Optional

class ImageProcessor:
    # Constants for image processing
    DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (512, 512)
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    
    @staticmethod
    def preprocess_image(
        image_data: np.ndarray,
        output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a fundus image.
        
        Args:
            image_data: Raw image data as numpy array
            output_size: Desired output image size
            
        Returns:
            Tuple of (normalized_image, enhanced_image)
        """
        # Resize image
        img_resized = cv2.resize(image_data, output_size, interpolation=cv2.INTER_AREA)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=ImageProcessor.CLAHE_CLIP_LIMIT,
            tileGridSize=ImageProcessor.CLAHE_GRID_SIZE
        )
        l_clahe = clahe.apply(l_channel)
        
        # Merge channels back
        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
        img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Normalize pixel intensities
        img_normalized = img_enhanced.astype(np.float32) / 255.0
        
        return img_normalized, img_enhanced 