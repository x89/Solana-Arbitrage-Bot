#!/usr/bin/env python3
"""
Model Evaluation Utilities
Comprehensive model evaluation system including:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Regression metrics (MSE, MAE, RMSE, R²)
- Time series metrics (MAPE, SMAPE, directional accuracy)
- Model comparison and benchmarking
- Confusion matrix analysis
- ROC curve generation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = []
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Add AUC if probabilities available
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    else:  # Multi-class
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                except:
                    pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {}
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2_score': r2_score(y_true, y_pred),
                'mape': self._calculate_mape(y_true, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            return {}
    
    def evaluate_time_series(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate time series model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2_score': r2_score(y_true, y_pred),
                'mape': self._calculate_mape(y_true, y_pred),
                'smape': self._calculate_smape(y_true, y_pred),
                'directional_accuracy': self._calculate_directional_accuracy(y_true, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating time series model: {e}")
            return {}
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        try:
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        except:
            return 0.0
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        try:
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mask = denominator != 0
            return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        except:
            return 0.0
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (for time series)"""
        try:
            if len(y_true) < 2:
                return 0.0
            
            # Calculate direction
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            
            accuracy = np.mean(true_direction == pred_direction)
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating directional accuracy: {e}")
            return 0.0
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        save_path: Optional[str] = None
    ) -> bool:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            save_path: Optional path to save plot
            
        Returns:
            True if successful
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            return False
    
    def plot_regression_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> bool:
        """Plot regression predictions vs actual"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            ax1 = axes[0]
            ax1.scatter(y_true, y_pred, alpha=0.6)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Predictions vs Actual')
            ax1.grid(True, alpha=0.3)
            
            # Residual plot
            ax2 = axes[1]
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Regression plot saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting regression: {e}")
            return False
    
    def compare_models(
        self,
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            evaluation_results: Dictionary of model_name -> metrics
            
        Returns:
            DataFrame with comparison results
        """
        try:
            comparison_data = []
            
            for model_name, metrics in evaluation_results.items():
                row = {'Model': model_name}
                row.update(metrics)
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            
            logger.info(f"Compared {len(df)} models")
            return df
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> bool:
        """
        Plot model comparison
        
        Args:
            comparison_df: DataFrame with model metrics
            save_path: Optional path to save plot
            
        Returns:
            True if successful
        """
        try:
            if comparison_df.empty:
                return False
            
            metric_cols = [col for col in comparison_df.columns if col != 'Model']
            
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
                
                comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax)
                ax.set_title(f'{metric} Comparison')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
            
            # Hide empty subplots
            for i in range(n_metrics, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting model comparison: {e}")
            return False

# Legacy function for backward compatibility
def evaluate_model(y_true, y_pred):
    """Evaluate model performance (legacy function)"""
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics
    except Exception as e:
        logger.error(f"Error in legacy evaluate_model: {e}")
        return {}

