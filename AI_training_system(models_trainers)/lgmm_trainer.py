#!/usr/bin/env python3
"""
LGMM Trainer Module
Latent Gaussian Mixture Model for Market Regime Detection
Based on Reynolds (2009) paper on Gaussian mixtures

This module implements LGMM for training on financial data, specifically designed for:
- Market regime identification (stable, volatile, mixed periods)
- Volatility clustering
- Trend state detection
- SPY stock data analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LGMMTrainer:
    """
    LGMM Trainer for Market Regime Detection
    
    Latent Gaussian Mixture Models identify hidden market regimes by modeling 
    data as mixtures of Gaussian distributions using EM algorithm.
    
    Formula: P(x|k) = 1 / √(2πσ_k²) e^(-(x-μ_k)²/(2σ_k²))
    
    Where:
    - μ_k: Mean of cluster k (cluster center)
    - σ_k²: Variance of cluster k (spread)
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        random_state: int = 42,
        max_iter: int = 100,
        normalize: bool = True
    ):
        """
        Initialize LGMM trainer
        
        Args:
            n_components: Number of Gaussian components (regimes)
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for EM algorithm
            normalize: Whether to normalize data using z-score
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.normalize = normalize
        
        self.gmm = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.probabilities = None
        self.training_metrics = {}
        self.regime_statistics = {}
        
        logger.info(f"LGMM trainer initialized with {n_components} components")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        volume_col: str = 'Volume'
    ) -> np.ndarray:
        """
        Prepare market features for LGMM training
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            Feature array for LGMM
        """
        try:
            features = []
            
            # Calculate returns
            if price_col in df.columns:
                returns = df[price_col].pct_change().dropna().values
                features.append(returns)
            
            # Calculate volume changes
            if volume_col in df.columns:
                volume_changes = df[volume_col].pct_change().dropna().values
                features.append(volume_changes)
            
            # Combine features
            if len(features) == 0:
                logger.error("No features to prepare")
                return np.array([])
            
            # Pad shorter arrays
            min_len = min(len(f) for f in features)
            features_padded = [f[:min_len] for f in features]
            
            # Stack features
            feature_array = np.column_stack(features_padded)
            
            logger.info(f"Prepared {len(feature_array)} samples with {feature_array.shape[1]} features")
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([])
    
    def load_spy_data(
        self,
        start_date: str = '2024-01-01',
        end_date: str = '2025-06-01',
        symbol: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Load SPY stock data for demonstration
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            symbol: Stock symbol (default: SPY)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading {symbol} data from {start_date} to {end_date}")
            data = yf.download(symbol, start=start_date, end=end_date)
            return data
            
        except Exception as e:
            logger.error(f"Error loading {symbol} data: {e}")
            return pd.DataFrame()
    
    def train(
        self,
        data: np.ndarray,
        n_init: int = 10
    ) -> Dict[str, Any]:
        """
        Train LGMM model on data
        
        Args:
            data: Input data array (samples x features)
            n_init: Number of initialization attempts
            
        Returns:
            Dictionary with training results
        """
        try:
            # Normalize data if required
            if self.normalize:
                data_scaled = self.scaler.fit_transform(data)
            else:
                data_scaled = data
                self.scaler = None
            
            # Fit Gaussian Mixture Model with EM algorithm
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=n_init
            )
            
            # Fit the model (EM algorithm)
            self.gmm.fit(data_scaled)
            
            # Get predictions and probabilities
            self.regime_labels = self.gmm.predict(data_scaled)
            self.probabilities = self.gmm.predict_proba(data_scaled)
            
            # Calculate metrics
            bic_score = self.gmm.bic(data_scaled)
            log_likelihood = self.gmm.score(data_scaled)
            
            # Analyze regimes
            self.regime_statistics = self._analyze_regimes(data_scaled)
            
            training_results = {
                'bic_score': float(bic_score),
                'log_likelihood': float(log_likelihood),
                'converged': self.gmm.converged_,
                'n_iter': self.gmm.n_iter_,
                'regime_stats': self.regime_statistics,
                'means': self.gmm.means_.tolist(),
                'covariances': [cov.tolist() for cov in self.gmm.covariances_]
            }
            
            self.training_metrics = {
                'bic': float(bic_score),
                'aic': float(self.gmm.aic(data_scaled)),
                'log_likelihood': float(log_likelihood),
                'converged': self.gmm.converged_,
                'n_iter': self.gmm.n_iter_
            }
            
            logger.info(f"LGMM training completed. BIC: {bic_score:.2f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training LGMM: {e}")
            return {}
    
    def optimize_components(
        self,
        data: np.ndarray,
        max_components: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize number of components using BIC criterion
        
        Args:
            data: Input data array
            max_components: Maximum number of components to test
            
        Returns:
            Optimal configuration and BIC scores
        """
        try:
            best_bic = float('inf')
            best_n_components = self.n_components
            bic_scores = []
            
            for n in range(2, max_components + 1):
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    random_state=self.random_state,
                    max_iter=self.max_iter
                )
                
                data_scaled = self.scaler.transform(data) if self.scaler else data
                gmm.fit(data_scaled)
                bic = gmm.bic(data_scaled)
                bic_scores.append({'n_components': n, 'bic': float(bic)})
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n
            
            logger.info(f"Optimal components: {best_n_components} (BIC: {best_bic:.2f})")
            
            return {
                'best_n_components': best_n_components,
                'best_bic': float(best_bic),
                'all_scores': bic_scores
            }
            
        except Exception as e:
            logger.error(f"Error optimizing components: {e}")
            return {}
    
    def _analyze_regimes(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze characteristics of each regime"""
        try:
            regime_stats = []
            
            for i in range(self.n_components):
                regime_mask = self.regime_labels == i
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    stats = {
                        'regime': i,
                        'count': int(np.sum(regime_mask)),
                        'proportion': float(np.sum(regime_mask) / len(self.regime_labels)),
                        'mean': regime_data.mean(axis=0).tolist(),
                        'std': regime_data.std(axis=0).tolist(),
                        'mean_volatility': float(regime_data.std().mean()),
                        'regime_type': self._identify_regime_type(i)
                    }
                else:
                    stats = {
                        'regime': i,
                        'count': 0,
                        'proportion': 0.0,
                        'mean': [],
                        'std': [],
                        'mean_volatility': 0.0,
                        'regime_type': 'Unknown'
                    }
                
                regime_stats.append(stats)
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"Error analyzing regimes: {e}")
            return []
    
    def _identify_regime_type(self, regime: int) -> str:
        """Identify type of regime based on volatility"""
        try:
            if hasattr(self, 'gmm') and self.gmm is not None:
                # Calculate regime volatility from covariance matrix
                cov = self.gmm.covariances_[regime]
                if isinstance(cov, np.ndarray):
                    volatility = np.trace(cov)
                    
                    if volatility < 0.5:
                        return 'Stable'
                    elif volatility < 1.5:
                        return 'Normal'
                    else:
                        return 'Volatile'
            
            return 'Unknown'
            
        except Exception as e:
            logger.error(f"Error identifying regime type: {e}")
            return 'Unknown'
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regime for new data
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (regime_labels, probabilities)
        """
        try:
            if self.gmm is None:
                logger.error("Model not trained yet. Call train() first.")
                return None, None
            
            # Normalize if scaler was used
            if self.scaler is not None:
                data_scaled = self.scaler.transform(data)
            else:
                data_scaled = data
            
            # Predict regimes
            labels = self.gmm.predict(data_scaled)
            probabilities = self.gmm.predict_proba(data_scaled)
            
            return labels, probabilities
                
        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            return None, None
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return self.training_metrics.copy()
    
    def get_regime_statistics(self) -> List[Dict[str, Any]]:
        """Get regime statistics"""
        return self.regime_statistics.copy()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'gmm': self.gmm,
                'scaler': self.scaler if self.normalize else None,
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'training_metrics': self.training_metrics,
                'regime_statistics': self.regime_statistics
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"LGMM model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            
            self.gmm = model_data['gmm']
            self.scaler = model_data.get('scaler')
            self.n_components = model_data['n_components']
            self.covariance_type = model_data['covariance_type']
            self.training_metrics = model_data.get('training_metrics', {})
            self.regime_statistics = model_data.get('regime_statistics', {})
            
            logger.info(f"LGMM model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def visualize_regimes(
        self,
        data: np.ndarray,
        save_path: Optional[str] = None
    ) -> bool:
        """Visualize identified regimes"""
        try:
            if data.shape[1] < 2:
                logger.warning("Need at least 2 features for visualization")
                return False
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Normalize if needed
            if self.scaler is not None:
                data_scaled = self.scaler.transform(data)
            else:
                data_scaled = data
            
            # Get current predictions
            labels, _ = self.predict(data)
            
            if labels is None:
                logger.error("Could not generate predictions")
                return False
            
            # Plot 1: Clustered data
            ax1 = axes[0]
            scatter = ax1.scatter(
                data_scaled[:, 0],
                data_scaled[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            
            # Plot means
            ax1.scatter(
                self.gmm.means_[:, 0],
                self.gmm.means_[:, 1],
                marker='x',
                s=200,
                c='red',
                linewidths=3,
                label='Regime Centers'
            )
            
            ax1.set_xlabel('Normalized Feature 1 (Returns)')
            ax1.set_ylabel('Normalized Feature 2 (Volume)')
            ax1.set_title('LGMM Market Regime Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Regime proportions
            ax2 = axes[1]
            regime_counts = np.bincount(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(regime_counts)))
            bars = ax2.bar(
                range(len(regime_counts)),
                regime_counts,
                color=colors,
                alpha=0.7
            )
            
            ax2.set_xlabel('Regime')
            ax2.set_ylabel('Count')
            ax2.set_title('Regime Distribution')
            ax2.set_xticks(range(len(regime_counts)))
            ax2.set_xticklabels([f'R{i}' for i in range(len(regime_counts))])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing regimes: {e}")
            return False

def demonstrate_lgmm_training():
    """Demonstration of LGMM training on SPY data"""
    try:
        print("=" * 70)
        print("LGMM Training Demonstration - Market Regime Detection")
        print("=" * 70)
        
        # Initialize trainer
        trainer = LGMMTrainer(n_components=3)
        
        # Load SPY data
        print("\nStep 1: Loading SPY data...")
        spy_data = trainer.load_spy_data()
        
        if spy_data.empty:
            print("Error: Could not load SPY data")
            return
        
        print(f"✓ Loaded {len(spy_data)} days of data")
        
        # Prepare features
        print("\nStep 2: Preparing market features...")
        features = trainer.prepare_features(spy_data)
        
        if len(features) == 0:
            print("Error: Could not prepare features")
            return
        
        print(f"✓ Prepared {len(features)} samples")
        
        # Optimize components
        print("\nStep 3: Optimizing number of components...")
        optimization_results = trainer.optimize_components(features, max_components=5)
        
        if 'best_n_components' in optimization_results:
            best_n = optimization_results['best_n_components']
            print(f"✓ Optimal components: {best_n}")
            trainer.n_components = best_n
        
        # Train model
        print("\nStep 4: Training LGMM model...")
        fit_results = trainer.train(features)
        
        if fit_results:
            print(f"✓ BIC Score: {fit_results.get('bic_score', 0):.2f}")
            print(f"✓ Log Likelihood: {fit_results.get('log_likelihood', 0):.4f}")
            print(f"✓ Converged: {fit_results.get('converged', False)}")
            print(f"✓ Iterations: {fit_results.get('n_iter', 0)}")
        
        # Analyze regimes
        print("\nStep 5: Analyzing regimes...")
        for stats in fit_results.get('regime_stats', []):
            print(f"\nRegime {stats['regime']} ({stats['regime_type']}):")
            print(f"  - Count: {stats['count']}")
            print(f"  - Proportion: {stats['proportion']:.1%}")
            print(f"  - Mean Volatility: {stats['mean_volatility']:.4f}")
        
        # Visualize
        print("\nStep 6: Creating visualization...")
        success = trainer.visualize_regimes(features, save_path='spy_lgmm_regimes.png')
        
        if success:
            print("✓ Plot saved to spy_lgmm_regimes.png")
        
        print("\n" + "=" * 70)
        print("LGMM Training Demonstration Completed!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in LGMM demonstration: {e}")

if __name__ == "__main__":
    demonstrate_lgmm_training()

