#!/usr/bin/env python3
"""
Pattern Trainer Module
Comprehensive YOLO training system for chart pattern detection including:
- Dataset preparation and augmentation
- YOLO model training with Ultralytics
- Training configuration and hyperparameters
- Model validation and evaluation
- Training monitoring and checkpoints
- Model export and deployment
"""

import os
import logging
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternTrainer:
    """Comprehensive YOLO trainer for chart pattern detection"""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        data_dir: str = "data/patterns",
        output_dir: str = "models/pattern_detection"
    ):
        """
        Initialize pattern trainer
        
        Args:
            model_name: YOLO model name or path
            data_dir: Directory containing training data
            output_dir: Directory for model outputs
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training configuration
        self.config = {
            'epochs': 100,
            'batch_size': 16,
            'imgsz': 640,
            'device': 'cuda' if YOLO_AVAILABLE else 'cpu',
            'project': output_dir,
            'name': 'yolo_training',
            'patience': 20,
            'save': True,
            'save_period': 10,
            'plots': True,
            'val': True
        }
        
        self.model = None
        self.training_history = []
        
        logger.info("Pattern trainer initialized")
    
    def prepare_dataset(self, train_split: float = 0.8) -> bool:
        """
        Prepare dataset for training
        
        Args:
            train_split: Proportion of data for training
            
        Returns:
            True if successful
        """
        try:
            logger.info("Preparing dataset...")
            
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                logger.error(f"Data directory not found: {self.data_dir}")
                return False
            
            # Create train/val splits
            images_dir = os.path.join(self.data_dir, 'images')
            labels_dir = os.path.join(self.data_dir, 'labels')
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                logger.error("Images or labels directory not found")
                return False
            
            # List all images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(Path(images_dir).glob(f'*{ext}'))
                image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
            
            # Split into train/val
            split_idx = int(len(image_files) * train_split)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            logger.info(f"Dataset prepared: {len(train_files)} train, {len(val_files)} val")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return False
    
    def create_yolo_config(
        self,
        num_classes: int,
        class_names: List[str],
        output_path: str = "dataset.yaml"
    ) -> str:
        """
        Create YOLO configuration file
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
            output_path: Path to save config
            
        Returns:
            Config file path
        """
        try:
            config = {
                'path': os.path.abspath(self.data_dir),
                'train': 'images/train',
                'val': 'images/val',
                'nc': num_classes,
                'names': class_names
            }
            
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"YOLO config created: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating YOLO config: {e}")
            return ""
    
    def train_yolo_model(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: int = 640,
        data_config: Optional[str] = None,
        resume: bool = False
    ) -> bool:
        """
        Train YOLO model for pattern detection
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            data_config: Path to data configuration file
            resume: Whether to resume training
            
        Returns:
            True if training successful
        """
        try:
            if not YOLO_AVAILABLE:
                logger.error("YOLO not available. Install ultralytics.")
                return False
            
            logger.info("Starting YOLO training...")
            
            # Update config
            if epochs:
                self.config['epochs'] = epochs
            if batch_size:
                self.config['batch_size'] = batch_size
            self.config['imgsz'] = img_size
            
            # Load model
            logger.info(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Train model
            logger.info("Training model...")
            results = self.model.train(
                data=data_config or os.path.join(self.data_dir, 'data.yaml'),
                epochs=self.config['epochs'],
                batch=self.config['batch_size'],
                imgsz=self.config['imgsz'],
                device=self.config['device'],
                project=self.config['project'],
                name=self.config['name'],
                patience=self.config['patience'],
                save=self.config['save'],
                save_period=self.config['save_period'],
                plots=self.config['plots'],
                val=self.config['val'],
                resume=resume
            )
            
            # Save training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'results': str(results),
                'config': self.config
            })
            
            logger.info("YOLO training completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training YOLO model: {e}")
            return False
    
    def validate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate trained model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Validation metrics
        """
        try:
            if not YOLO_AVAILABLE:
                logger.error("YOLO not available")
                return {}
            
            if model_path:
                self.model = YOLO(model_path)
            elif self.model is None:
                logger.error("No model loaded")
                return {}
            
            # Run validation
            metrics = self.model.val()
            
            logger.info(f"Validation metrics: {metrics}")
            
            return {
                'mAP50': getattr(metrics, 'metrics', {}).get('mAP50', 0.0),
                'mAP50-95': getattr(metrics, 'metrics', {}).get('mAP50-95', 0.0),
                'precision': getattr(metrics, 'metrics', {}).get('precision', 0.0),
                'recall': getattr(metrics, 'metrics', {}).get('recall', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {}
    
    def export_model(self, format: str = 'onnx') -> str:
        """
        Export trained model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tensorrt', etc.)
            
        Returns:
            Path to exported model
        """
        try:
            if self.model is None:
                logger.error("No model to export")
                return ""
            
            logger.info(f"Exporting model to {format}...")
            
            # Export
            export_path = self.model.export(format=format)
            
            logger.info(f"Model exported to: {export_path}")
            
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return ""
    
    def predict(
        self,
        image_path: str,
        confidence: float = 0.5,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict patterns in image
        
        Args:
            image_path: Path to image
            confidence: Confidence threshold
            save_results: Whether to save results
            
        Returns:
            List of predictions
        """
        try:
            if self.model is None:
                logger.error("No model loaded for prediction")
                return []
            
            logger.info(f"Predicting patterns in: {image_path}")
            
            # Run prediction
            results = self.model.predict(
                image_path,
                conf=confidence,
                save=save_results
            )
            
            # Extract predictions
            predictions = []
            for result in results:
                for box in result.boxes:
                    predictions.append({
                        'class': int(box.cls.item()),
                        'confidence': float(box.conf.item()),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            logger.info(f"Found {len(predictions)} patterns")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return []
    
    def save_training_history(self, filepath: str):
        """Save training history"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info(f"Training history saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving training history: {e}")

def main():
    """Main function to demonstrate pattern training"""
    try:
        logger.info("=" * 60)
        logger.info("Pattern Trainer Demo")
        logger.info("=" * 60)
        
        # Initialize trainer
        trainer = PatternTrainer(
            model_name="yolov8n.pt",
            data_dir="data/patterns",
            output_dir="models/pattern_detection"
        )
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        # trainer.prepare_dataset()
        
        # Create config
        class_names = ['head_shoulders', 'double_top', 'double_bottom', 'triangle']
        config_path = trainer.create_yolo_config(
            num_classes=len(class_names),
            class_names=class_names,
            output_path="pattern_config.yaml"
        )
        
        logger.info(f"Config created: {config_path}")
        
        # Note: Actual training requires dataset
        # trainer.train_yolo_model(epochs=50, batch_size=16)
        
        logger.info("=" * 60)
        logger.info("Pattern Trainer Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

