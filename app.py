from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from handlers.image_handler import ImageHandler
import base64
import logging
from typing import Optional
from fastapi.openapi.models import MediaType
from fastapi.openapi.utils import get_openapi

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
PROCESS_ENDPOINT_PATH: str = "/process"
ALLOWED_MIME_TYPES: set = {"image/jpeg", "image/png", "application/octet-stream"}

# FastAPI app instance
app = FastAPI(
    title="RetinAI API",
    description="API for processing and analyzing retinal images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_handler = ImageHandler()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RetinAI API",
        version="1.0.0",
        description="API for processing and analyzing retinal images",
        routes=app.routes,
    )
    
    # Modify the schema for the /process endpoint
    for path in openapi_schema["paths"].values():
        for method in path.values():
            if "requestBody" in method:
                method["requestBody"]["content"] = {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "format": "binary"
                                }
                            }
                        }
                    }
                }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.post(
    PROCESS_ENDPOINT_PATH,
    summary="Process a retinal image",
    description="Upload a retinal image (PNG or JPEG) for processing, segmentation, and DR classification",
    response_description="Processed and segmented images with DR classification"
)
async def process_image(
    file: UploadFile = File(..., description="The retinal image file (PNG or JPEG)")
) -> dict:
    """
    Process and segment an uploaded fundus image.
    
    Args:
        file: Uploaded image file (PNG or JPEG)
        
    Returns:
        JSON response with processed and segmented images
    """
    try:
        # Log file details
        logger.debug(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Check if file is provided
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Basic validations
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Process and segment the image
        try:
            result = await image_handler.process_image(content)
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Return the response
        return {
            "message": result["message"],
            "processed_image": result["processed_image"],
            "segmented_image": result["segmented_image"],
            "dr_classification": result["dr_classification"],
            "dr_classification_error": result["dr_classification_error"]
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        await file.close() 