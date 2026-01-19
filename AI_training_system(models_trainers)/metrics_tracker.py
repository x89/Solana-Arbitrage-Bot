#!/usr/bin/env python3
"""
Training Metrics Module
Comprehensive training metrics tracking and visualization including:
- Real-time metric logging
- Training/validation split tracking
- Loss tracking and early stopping
- Performance metric aggregation
- Comparison across experiments
- Export to various formats
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Comprehensive metrics tracking for AI training"""
    
    def __init__(self, log_file: str = "training_metrics.json", enable_plotting: bool = True):
        """
        Initialize metrics tracker
        
        Args:
            log_file: Path to metrics log file
            enable_plotting: Enable automatic plotting
        """
        self.log_file = log_file
        self.enable_plotting = enable_plotting
        self.metrics = []
        self.current_epoch = 0
        self.current_metrics = {}
        self.experiment_name = None
        self.model_name = None
    
    def log(
        self,
        metrics: Dict[str, Any],
        epoch: int = None,
        phase: str = 'training'
    ):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
            phase: Phase of training (training/validation/test)
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch if epoch is not None else self.current_epoch,
                'phase': phase,
                **metrics
            }
            
            if self.experiment_name:
                log_entry['experiment'] = self.experiment_name
            if self.model_name:
                log_entry['model'] = self.model_name
            
            self.metrics.append(log_entry)
            self.current_epoch = epoch if epoch is not None else self.current_epoch
            
            logger.debug(f"Logged metrics for {phase} phase")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float] = None,
        test_metrics: Dict[str, float] = None
    ):
        """
        Log metrics for an entire epoch
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics
        """
        try:
            self.current_epoch = epoch
            
            # Log training metrics
            self.log(train_metrics, epoch=epoch, phase='training')
            
            # Log validation metrics if provided
            if val_metrics:
                self.log(val_metrics, epoch=epoch, phase='validation')
            
            # Log test metrics if provided
            if test_metrics:
                self.log(test_metrics, epoch=epoch, phase='test')
            
        except Exception as e:
            logger.error(f"Error logging epoch metrics: {e}")
    
    def save(self, filepath: Optional[str] = None):
        """
        Save metrics to file
        
        Args:
            filepath: Optional custom filepath
        """
        try:
            filepath = filepath or self.log_file
            
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def load(self, filepath: Optional[str] = None):
        """
        Load metrics from file
        
        Args:
            filepath: Optional custom filepath
        """
        try:
            filepath = filepath or self.log_file
            
            if not os.path.exists(filepath):
                logger.warning(f"Metrics file not found: {filepath}")
                return
            
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)
            
            logger.info(f"Loaded {len(self.metrics)} metric entries")
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics as DataFrame"""
        try:
            if not self.metrics:
                return pd.DataFrame()
            
            return pd.DataFrame(self.metrics)
            
        except Exception as e:
            logger.error(f"Error creating metrics DataFrame: {e}")
            return pd.DataFrame()
    
    def get_best_epoch(self, metric_name: str = 'val_loss', mode: str = 'min') -> int:
        """
        Get epoch with best metric value
        
        Args:
            metric_name: Name of metric to optimize
            mode: 'min' or 'max'
            
        Returns:
            Best epoch number
        """
        try:
            val_metrics = [m for m in self.metrics if m.get('phase') == 'validation']
            
            if not val_metrics:
                return 0
            
            values = [m.get(metric_name, float('inf') if mode == 'min' else float('-inf')) 
                     for m in val_metrics]
            
            if mode == 'min':
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            return val_metrics[best_idx].get('epoch', 0)
            
        except Exception as e:
            logger.error(f"Error getting best epoch: {e}")
            return 0
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> bool:
        """
        Plot training curves
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            True if successful
        """
        try:
            df = self.get_metrics_df()
            
            if df.empty:
                logger.warning("No metrics to plot")
                return False
            
            # Identify metric columns
            metric_cols = [col for col in df.columns if col not in 
                             ['timestamp', 'epoch', 'phase', 'experiment', 'model']]
            
            if not metric_cols:
                return False
            
            # Filter by phase
            train_df = df[df['phase'] == 'training']
            val_df = df[df['phase'] == 'validation']
            
            # Create subplots
            n_metrics = min(len(metric_cols), 6)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            
            if n_metrics == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metric_cols[:n_metrics]):
                ax = axes[i]
                
                # Plot training
                if not train_df.empty:
                    ax.plot(train_df['epoch'], train_df[metric], 
                          label='Training', alpha=0.7)
                
                # Plot validation
                if not val_df.empty:
                    ax.plot(val_df['epoch'], val_df[metric], 
                          label='Validation', alpha=0.7, linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_metrics, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training curves saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")
            return False
    
    def compare_experiments(self, save_path: Optional[str] = None) -> bool:
        """
        Compare multiple experiments
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            True if successful
        """
        try:
            df = self.get_metrics_df()
            
            if df.empty or 'experiment' not in df.columns:
                logger.warning("No experiments to compare")
                return False
            
            # Get unique experiments
            experiments = df['experiment'].unique()
            
            # Plot metrics comparison
            metric_cols = [col for col in df.columns if col not in 
                         ['timestamp', 'epoch', 'phase', 'experiment', 'model']]
            
            fig, axes = plt.subplots(len(metric_cols), 1, figsize=(12, 4 * len(metric_cols)))
            
            if len(metric_cols) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metric_cols):
                ax = axes[i]
                
                for exp in experiments:
                    exp_data = df[(df['experiment'] == exp) & (df['phase'] == 'validation')]
                    if not exp_data.empty:
                        ax.plot(exp_data['epoch'], exp_data[metric], 
                              label=exp, alpha=0.7)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Experiment comparison saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error comparing experiments: {e}")
            return False
    
    def export_to_csv(self, filepath: str):
        """Export metrics to CSV"""
        try:
            df = self.get_metrics_df()
            df.to_csv(filepath, index=False)
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of metrics"""
        try:
            df = self.get_metrics_df()
            
            if df.empty:
                return {}
            
            summary = {
                'total_epochs': len(df[df['phase'] == 'training']),
                'experiments': df['experiment'].unique().tolist() if 'experiment' in df.columns else [],
                'models': df['model'].unique().tolist() if 'model' in df.columns else [],
                'phases': df['phase'].unique().tolist()
            }
            
            # Get metric statistics
            metric_cols = [col for col in df.columns if col not in 
                         ['timestamp', 'epoch', 'phase', 'experiment', 'model']]
            
            summary['metrics'] = {}
            for metric in metric_cols:
                metric_values = df[metric].dropna()
                if len(metric_values) > 0:
                    summary['metrics'][metric] = {
                        'mean': float(metric_values.mean()),
                        'std': float(metric_values.std()),
                        'min': float(metric_values.min()),
                        'max': float(metric_values.max())
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {}
    
    def reset(self):
        """Reset tracker"""
        self.metrics = []
        self.current_epoch = 0
        self.current_metrics = {}
        logger.info("Metrics tracker reset")
    
    def set_experiment(self, name: str):
        """Set current experiment name"""
        self.experiment_name = name
        logger.info(f"Set experiment: {name}")
    
    def set_model(self, name: str):
        """Set current model name"""
        self.model_name = name
        logger.info(f"Set model: {name}")

