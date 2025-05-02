import numpy as np
import cv2
from utils.image_processor import ImageProcessor
from utils.segmentation_processor import SegmentationProcessor
from utils.dr_classifier import DRClassifier
from typing import Dict, Any
import io
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageHandler:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.segmentation_processor = SegmentationProcessor()
        self.dr_classifier = DRClassifier()
    
    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Process the uploaded image data, segment it, and classify DR severity.
        
        Args:
            image_data: Raw bytes of the uploaded image
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.debug("Starting image processing")
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image data")
            
            # Process the image
            normalized_img, enhanced_img = self.image_processor.preprocess_image(img)
            logger.debug("Image preprocessing complete")
            
            # Convert enhanced image to bytes for classification
            _, enhanced_img_encoded = cv2.imencode('.png', enhanced_img)
            enhanced_img_bytes = enhanced_img_encoded.tobytes()
            
            # Get segmentation
            try:
                segmented_image_bytes = self.segmentation_processor.segment_image(enhanced_img)
                logger.debug(f"Segmentation complete, bytes length: {len(segmented_image_bytes)}")
            except Exception as e:
                logger.error(f"Segmentation error: {str(e)}")
                segmented_image_bytes = None
            
            # Classify DR severity
            try:
                dr_classification = self.dr_classifier.classify_severity(enhanced_img_bytes)
                logger.debug("DR classification complete")
            except Exception as e:
                logger.error(f"Classification error: {str(e)}")
                dr_classification = {"success": False, "error": str(e)}
            
            # Base64 encode the images
            try:
                processed_image_b64 = base64.b64encode(enhanced_img_bytes).decode('utf-8')
                segmented_image_b64 = base64.b64encode(segmented_image_bytes).decode('utf-8') if segmented_image_bytes else None
                logger.debug("Base64 encoding complete")
            except Exception as e:
                logger.error(f"Base64 encoding error: {str(e)}")
                raise ValueError(f"Failed to encode images: {str(e)}")
            
            # Prepare response
            response = {
                "success": True,
                "message": "Image processed successfully",
                "processed_image": processed_image_b64,
                "segmented_image": segmented_image_b64,
                "dr_classification": None,
                "dr_classification_error": None
            }
            
            # Add classification if successful
            if dr_classification.get("success", False):
                response["dr_classification"] = dr_classification["classification"]
            else:
                response["dr_classification_error"] = dr_classification.get("error", "Unknown error")
            
            logger.debug("Response preparation complete")
            return response
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error processing image: {str(e)}",
                "processed_image": None,
                "segmented_image": None,
                "dr_classification": None,
                "dr_classification_error": None
            } 