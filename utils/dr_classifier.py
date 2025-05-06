import ollama
from typing import Dict, Any, Tuple
import tempfile
import os
import logging
import cv2
import numpy as np
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DRClassifier:
    # Constants
    MODEL_NAME: str = "rohithbojja/llava-med-v1.6"  # Using medical-specific LLaVA model
    PROMPT_TEMPLATE: str = """Analyze this retinal image by CAREFULLY COUNTING the colored markers:

MARKER LEGEND:
- Yellow (CTW): Cotton Wool Spots
- Green (EX): Exudates
- Light Blue (HE): Hemorrhages
- Dark Blue (MA): Microaneurysms

COUNTING INSTRUCTIONS:
1. Count each colored marker individually
2. Note the location (quadrant) of each marker
3. Small, isolated dots count as individual markers
4. Clusters should be counted as separate markers if distinct

Grade DR severity using these EXACT marker counts:

0 (No DR):
- NO markers of any color present
- Completely clean image

1 (Mild NPDR):
- ONLY 1-2 dark blue (MA) markers
- NO other colored markers present
- If ANY other colors exist, grade higher

2 (Moderate NPDR):
- 3-5 markers total of any type OR
- Small cluster of green (EX) markers OR
- Any combination of 2-3 different colored markers
- But less than severe criteria

3 (Severe NPDR):
- More than 5 markers total OR
- Large cluster of any marker type OR
- 3 or more different colored markers present

4 (Proliferative DR):
- Only grade as 4 if there are obvious signs of neovascularization
- These would appear as irregular vessel patterns
- Do NOT grade as 4 based on markers alone

OUTPUT FORMAT:
Grade: [0-4]
Justification: [List EXACT COUNT of each colored marker by type and location]"""

    @staticmethod
    def _parse_response(response_text: str) -> Tuple[int, str]:
        """
        Parse the model's response to extract grade and justification.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Tuple of (grade, justification)
            
        Raises:
            ValueError: If unable to parse grade or grade is invalid
        """
        # Extract grade using regex
        grade_match = re.search(r"Grade:\s*(\d+)", response_text)
        if not grade_match:
            raise ValueError("Could not find grade in model response")
            
        grade = int(grade_match.group(1))
        if grade < 0 or grade > 4:
            raise ValueError(f"Invalid grade {grade}, must be between 0 and 4")
            
        # Extract justification
        justification_match = re.search(r"Justification:\s*(.+)", response_text)
        justification = justification_match.group(1) if justification_match else "No justification provided"
        
        return grade, justification

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
            logger.debug(f"Raw model response: {response['response']}")
            
            if not response or "response" not in response:
                raise ValueError("No response from Ollama API")
            
            if "I'm unable to" in response["response"] or "I cannot" in response["response"]:
                return {
                    "success": False,
                    "error": "Model failed to process the image properly"
                }
            
            # Parse and validate the response
            try:
                grade, justification = DRClassifier._parse_response(response["response"])
                logger.debug(f"Parsed response - Grade: {grade}, Justification: {justification}")
                
                return {
                    "success": True,
                    "classification": f"Grade: {grade}\nJustification: {justification}",
                    "grade": grade,
                    "justification": justification,
                    "raw_response": response["response"]
                }
            except ValueError as ve:
                return {
                    "success": False,
                    "error": f"Failed to parse model response: {str(ve)}",
                    "raw_response": response.get("response", "No response")
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