from fundus_lesions_toolkit.models import segment
from fundus_lesions_toolkit.constants import DEFAULT_COLORS, LESIONS
from fundus_lesions_toolkit.utils.visualization import plot_image_and_mask
import numpy as np
import cv2
import io
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SegmentationProcessor:
    # Constants
    DEVICE: str = 'cpu'  # Explicitly set to CPU
    ALPHA: float = 0.8
    TITLE: str = 'Fundus Image Segmentation'
    
    @staticmethod
    def segment_image(image: np.ndarray) -> bytes:
        """
        Segment a fundus image and create visualization.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Bytes of the segmented image visualization
        """
        try:
            logger.debug(f"Input image shape: {image.shape}")
            
            # Ensure image is in BGR format (OpenCV default)
            if len(image.shape) == 2:  # If grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug("Converted image to RGB")
            
            # Perform segmentation with explicit device setting
            prediction = segment(
                image_rgb, 
                device=torch.device(SegmentationProcessor.DEVICE),
                compile=False  # Disable compilation for CPU
            )
            logger.debug(f"Segmentation prediction shape: {prediction.shape}")
            
            # Create visualization using the toolkit's visualization function
            plt.ioff()  # Turn off interactive mode
            fig = plt.figure(figsize=(10, 5))
            plot_image_and_mask(
                image_rgb,  # Use RGB image for plotting
                prediction, 
                alpha=SegmentationProcessor.ALPHA,
                title=SegmentationProcessor.TITLE,
                colors=DEFAULT_COLORS,
                labels=LESIONS
            )
            
            # Convert plot to image bytes without displaying
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)  # Close the figure explicitly
            buf.seek(0)
            
            # Get the bytes and verify it's actually bytes
            result_bytes = buf.getvalue()
            if not isinstance(result_bytes, bytes):
                logger.error(f"Expected bytes but got {type(result_bytes)}")
                result_bytes = bytes(result_bytes)
            
            logger.debug(f"Returning bytes of length: {len(result_bytes)}")
            return result_bytes
            
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}", exc_info=True)
            # Return a placeholder image in case of error
            error_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, error_img_encoded = cv2.imencode('.png', error_img)
            return error_img_encoded.tobytes()
        finally:
            # Make sure all figures are closed
            plt.close('all')
            
    @staticmethod
    def get_raw_segmentation(image: np.ndarray) -> np.ndarray:
        """
        Get raw segmentation mask without visualization.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Segmentation mask as numpy array
        """
        try:
            # Convert BGR to RGB for consistent processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return segment(
                image_rgb, 
                device=torch.device(SegmentationProcessor.DEVICE),
                compile=False  # Disable compilation for CPU
            )
        except Exception as e:
            print(f"Raw segmentation error: {str(e)}")  # Add logging
            raise ValueError(f"Raw segmentation failed: {str(e)}") 