import ollama
from typing import Dict, Any
import tempfile
import os
import logging
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DRClassifier:
    # Constants
    MODEL_NAME: str = "rohithbojja/llava-med-v1.6"  # Using medical-specific LLaVA model
    PROMPT_TEMPLATE: str = """Analyze this retinal image and classify the severity of diabetic retinopathy (DR) according to this scale:

0: No DR — No signs of diabetic retinopathy
1: Mild — Microaneurysms only
2: Moderate — More than just microaneurysms but less than severe nonproliferative DR
3: Severe — More than 20 intraretinal hemorrhages in each quadrant, venous beading in 2+ quadrants, IRMA in 1+ quadrant
4: Proliferative DR — Neovascularization and/or vitreous/preretinal hemorrhage

Provide ONLY:
1. The numeric grade (0-4)
2. A brief, specific justification based on what you see in THIS image
Do not give general descriptions or disclaimers."""

    @staticmethod
    def classify_severity(image_bytes: bytes) -> Dict[str, Any]:
        """
        Classify the severity of diabetic retinopathy in a retinal image.
        
        Args:
            image_bytes: Raw bytes of the image
            
        Returns:
            Dictionary containing classification results and explanation
        """
        temp_path = None
        try:
            logger.debug(f"Input image_bytes type: {type(image_bytes)}")
            
            if not isinstance(image_bytes, bytes):
                logger.error(f"Invalid input type: {type(image_bytes)}")
                raise ValueError(f"Expected bytes, got {type(image_bytes)}")
            
            logger.debug(f"Image bytes length: {len(image_bytes)}")
            
            # Verify image bytes are valid by decoding them
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image data")
            
            # Re-encode the image to ensure valid PNG format
            _, img_encoded = cv2.imencode('.png', img)
            valid_image_bytes = img_encoded.tobytes()

            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(valid_image_bytes)
                temp_file.flush()  # Ensure all data is written
                temp_path = temp_file.name
                
            logger.debug(f"Wrote image to temporary file: {temp_path}")

            # Verify the file exists and has content
            if not os.path.exists(temp_path):
                raise ValueError("Temporary file was not created")
            
            file_size = os.path.getsize(temp_path)
            logger.debug(f"Temporary file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Temporary file is empty")

            # Generate classification using Ollama with image path
            logger.debug("Calling Ollama API...")
            response = ollama.generate(
                model=DRClassifier.MODEL_NAME,
                prompt=DRClassifier.PROMPT_TEMPLATE,
                images=[temp_path]
            )
            logger.debug("Received response from Ollama API")
            
            if not response or "response" not in response:
                raise ValueError("No response from Ollama API")
            
            if "I'm unable to" in response["response"] or "I cannot" in response["response"]:
                return {
                    "success": False,
                    "error": "Model failed to process the image properly"
                }
            
            return {
                "success": True,
                "classification": response["response"]
            }
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
            
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug("Cleaned up temporary file")
                except Exception as e:
                    logger.error(f"Failed to delete temp file: {str(e)}") 