# RetinAI

A FastAPI-based service for retinal image analysis, segmentation, and diabetic retinopathy classification.

## Features

- Image preprocessing and enhancement
- Retinal image segmentation
- Diabetic Retinopathy (DR) classification using llava-med-v1.6 (a medical-specific LLaVA model)
- REST API endpoints for image processing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and pull the medical LLaVA model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull rohithbojja/llava-med-v1.6
```

3. Start the Ollama server:
```bash
ollama serve
```

4. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Access the Swagger UI documentation at `http://localhost:8000/docs`

### API Endpoints

#### POST /process
Processes a retinal image and returns:
- Enhanced image
- Segmented image with lesion detection
- DR severity classification (0-4 scale)

## Project Structure 