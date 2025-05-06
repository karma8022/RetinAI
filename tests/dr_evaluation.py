import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from pathlib import Path
import logging
import cv2
import re
import asyncio
import argparse
import aiohttp
from typing import Dict, Tuple, List
from datetime import datetime

# Set up logging for console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DREvaluation:
    def __init__(self, image_dir: str, ground_truth_file: str, api_url: str = "http://localhost:8000"):
        """
        Initialize the DR evaluation suite.
        
        Args:
            image_dir: Directory containing the test images
            ground_truth_file: Path to the CSV file with ground truth data
            api_url: URL of the FastAPI server
        """
        self.image_dir = Path(image_dir)
        self.api_url = api_url.rstrip('/')
        # Read CSV and ensure column names match
        self.ground_truth = pd.read_csv(ground_truth_file)
        if 'id_code' not in self.ground_truth.columns:
            # Try to rename the first column to id_code if it contains image names
            self.ground_truth = self.ground_truth.rename(columns={self.ground_truth.columns[0]: 'id_code'})
        if 'diagnosis' not in self.ground_truth.columns:
            # If diagnosis column is missing, try to find a column with numeric values
            numeric_cols = self.ground_truth.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.ground_truth = self.ground_truth.rename(columns={numeric_cols[0]: 'diagnosis'})
        
        logger.info(f"Loaded {len(self.ground_truth)} ground truth records")
        logger.info(f"Columns found: {self.ground_truth.columns.tolist()}")
        logger.info(f"Using API endpoint: {self.api_url}")
        
        self.results = []
        
    def extract_grade(self, classification_output: str) -> int:
        """Extract numerical grade from classifier output."""
        match = re.search(r'Grade:\s*(\d)', classification_output)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract grade from: {classification_output}")
    
    async def evaluate_single_image(self, session: aiohttp.ClientSession, image_path: Path) -> Dict:
        """
        Evaluate a single image and return results.
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to bytes
            _, img_encoded = cv2.imencode('.png', img)
            img_bytes = img_encoded.tobytes()
            
            # Prepare the file for upload
            data = aiohttp.FormData()
            data.add_field('file',
                          img_bytes,
                          filename=image_path.name,
                          content_type='image/png')
            
            # Make API request
            async with session.post(f"{self.api_url}/process", data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"API request failed with status {response.status}: {error_text}")
                
                result = await response.json()
                
                if not result.get("dr_classification"):
                    raise ValueError(f"No classification in response: {result}")
                
                # Extract grade
                predicted_grade = self.extract_grade(result["dr_classification"])
                
                # Get ground truth
                image_name = image_path.name
                ground_truth = self.ground_truth[
                    self.ground_truth['id_code'].str.contains(image_name, na=False)
                ]['diagnosis'].iloc[0]
                
                return {
                    'image_name': image_name,
                    'predicted': predicted_grade,
                    'ground_truth': ground_truth,
                    'full_output': result["dr_classification"]
                }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
            
    async def run_evaluation(self) -> Dict[str, float]:
        """
        Run evaluation on first 50 images and compute metrics.
        """
        logger.info(f"Starting evaluation in directory: {self.image_dir}")
        logger.info(f"Looking for images with extensions: .png, .jpg, .jpeg")
        
        # Get all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(list(self.image_dir.glob(f'*{ext}')))
        
        # Take only first 50 images
        image_files = image_files[:10]
        total_images = len(image_files)
        logger.info(f"Processing first {total_images} images")
        processed = 0
        
        # Create aiohttp session for reuse
        async with aiohttp.ClientSession() as session:
            for image_path in image_files:
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                    
                result = await self.evaluate_single_image(session, image_path)
                if result:
                    self.results.append(result)
                    processed += 1
                    print(
                        f"[{processed}/{total_images}] Processed {result['image_name']}: "
                        f"Predicted={result['predicted']}, "
                        f"Actual={result['ground_truth']}"
                    )
                
                # Add a small delay between requests to avoid overwhelming the server
                await asyncio.sleep(0.1)
        
        print("\nProcessing complete. Computing metrics...")
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        """
        if not self.results:
            return {}
            
        y_true = [r['ground_truth'] for r in self.results]
        y_pred = [r['predicted'] for r in self.results]
        
        # Convert to one-hot for ROC AUC
        y_true_bin = pd.get_dummies(y_true)
        y_pred_bin = pd.get_dummies(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc_ovr': roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr'),
            'roc_auc_ovo': roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovo')
        }
        
        # Print detailed metrics with clear formatting
        print("\n" + "="*50)
        print("           EVALUATION RESULTS")
        print("="*50)
        print(f"\nTotal Images Processed: {len(self.results)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC (One-vs-Rest): {metrics['roc_auc_ovr']:.4f}")
        print(f"ROC AUC (One-vs-One): {metrics['roc_auc_ovo']:.4f}")
        
        # Print confusion matrix with labels
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative")
        print("-"*50)
        for i, row in enumerate(cm):
            print(f"Grade {i}: {row}")
        
        # Print classification report with better formatting
        print("\nDetailed Classification Report:")
        print("-"*50)
        report = classification_report(y_true, y_pred, 
                                    target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
        print(report)
        
        # Save results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('evaluation_results_50.csv', index=False)
        print("\nDetailed results saved to 'evaluation_results_50.csv'")
        
        return metrics

async def main():
    parser = argparse.ArgumentParser(description='Evaluate DR Classification Model')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to CSV file with ground truth data')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000', help='FastAPI server URL')
    
    args = parser.parse_args()
    
    # Initialize evaluation
    evaluator = DREvaluation(args.image_dir, args.ground_truth, args.api_url)
    
    # Run evaluation
    try:
        metrics = await evaluator.run_evaluation()
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 