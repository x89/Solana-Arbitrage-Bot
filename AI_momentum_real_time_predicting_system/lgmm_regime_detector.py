#!/usr/bin/env python3
"""
LGMM Regime Detection Module
Comprehensive Latent Gaussian Mixture Model implementation for market regime detection including:
- Market regime identification (stable, volatile, mixed)
- Gaussian Mixture Model fitting with EM algorithm
- BIC optimization for component selection
- Regime probability calculation
- Visualization and analysis tools
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
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LGMMRegimeDetector:
    """
    Latent Gaussian Mixture Model for Market Regime Detection
    
    LGMM identifies hidden market regimes by modeling data as mixtures of 
    Gaussian distributions. Excellent for market regime detection in SPY data,
    volatility clustering, and trend state identification.
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        random_state: int = 42,
        max_iter: int = 100
    ):
        """
        Initialize LGMM Regime Detector
        
        Args:
            n_components: Number of Gaussian components (regimes)
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for EM algorithm
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.gmm = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.probabilities = None
        self.bic_scores = []
        self.regime_names = ['Stable', 'Volatile', 'Mixed']
        
        logger.info(f"LGMM initialized with {n_components} components")
    
    def fit(self, data: np.ndarray, normalize: bool = True) -> Dict[str, Any]:
        """
        Fit LGMM model to data
        
        Args:
            data: Input data array (samples x features)
            normalize: Whether to normalize data using z-score
            
        Returns:
            Dictionary with fitting results
        """
        try:
            # Normalize data
            if normalize:
                data_scaled = self.scaler.fit_transform(data)
            else:
                data_scaled = data
                self.scaler = None
            
            # Fit Gaussian Mixture Model
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            
            self.gmm.fit(data_scaled)
            
            # Get predictions and probabilities
            self.regime_labels = self.gmm.predict(data_scaled)
            self.probabilities = self.gmm.predict_proba(data_scaled)
            
            # Calculate metrics
            bic_score = self.gmm.bic(data_scaled)
            log_likelihood = self.gmm.score(data_scaled)
            
            # Analyze regimes
            regime_stats = self._analyze_regimes(data_scaled)
            
            results = {
                'bic_score': float(bic_score),
                'log_likelihood': float(log_likelihood),
                'converged': self.gmm.converged_,
                'n_iter': self.gmm.n_iter_,
                'regime_stats': regime_stats,
                'means': self.gmm.means_.tolist(),
                'covariances': [cov.tolist() for cov in self.gmm.covariances_]
            }
            
            logger.info(f"LGMM fitted successfully. BIC: {bic_score:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fitting LGMM: {e}")
            return {}
    
    def predict_regime(
        self,
        data: np.ndarray,
        return_probabilities: bool = True
    ) -> np.ndarray:
        """
        Predict market regime for new data
        
        Args:
            data: Input data array
            return_probabilities: Return probabilities for each regime
            
        Returns:
            Regime predictions or probabilities
        """
        try:
            if self.gmm is None:
                logger.error("Model not fitted yet. Call fit() first.")
                return None
            
            # Normalize if scaler was used
            if self.scaler is not None:
                data_scaled = self.scaler.transform(data)
            else:
                data_scaled = data
            
            # Predict regimes
            if return_probabilities:
                return self.gmm.predict_proba(data_scaled)
            else:
                return self.gmm.predict(data_scaled)
                
        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            return None
    
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
                
                gmm.fit(self.scaler.transform(data) if self.scaler else data)
                bic = gmm.bic(self.scaler.transform(data) if self.scaler else data)
                bic_scores.append({'n_components': n, 'bic': bic})
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n
            
            self.bic_scores = bic_scores
            
            logger.info(f"Optimal components: {best_n_components} (BIC: {best_bic:.2f})")
            
            return {
                'best_n_components': best_n_components,
                'best_bic': best_bic,
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
                        'mean_volatility': float(regime_data.std().mean())
                    }
                else:
                    stats = {
                        'regime': i,
                        'count': 0,
                        'proportion': 0.0,
                        'mean': [],
                        'std': [],
                        'mean_volatility': 0.0
                    }
                
                regime_stats.append(stats)
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"Error analyzing regimes: {e}")
            return []
    
    def get_regime_probability(
        self,
        data_point: np.ndarray,
        regime: int
    ) -> float:
        """
        Get probability that a data point belongs to a specific regime
        
        Args:
            data_point: Single data point
            regime: Regime index
            
        Returns:
            Probability [0, 1]
        """
        try:
            probabilities = self.predict_regime(data_point.reshape(1, -1), True)
            if probabilities is not None and regime < len(probabilities[0]):
                return float(probabilities[0][regime])
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting regime probability: {e}")
            return 0.0
    
    def identify_regime_type(self, regime: int) -> str:
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
    
    def plot_regimes(
        self,
        data: np.ndarray,
        save_path: Optional[str] = None,
        title: str = 'LGMM Market Regime Detection'
    ) -> bool:
        """
        Visualize identified regimes
        
        Args:
            data: Original data array
            save_path: Path to save plot
            title: Plot title
            
        Returns:
            True if successful
        """
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
            
            # Plot 1: Clustered data
            ax1 = axes[0]
            scatter = ax1.scatter(
                data_scaled[:, 0],
                data_scaled[:, 1],
                c=self.regime_labels,
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
            
            ax1.set_xlabel('Normalized Feature 1')
            ax1.set_ylabel('Normalized Feature 2')
            ax1.set_title('LGMM Clustered Data')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Regime proportions
            ax2 = axes[1]
            regime_counts = np.bincount(self.regime_labels)
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
            
            plt.suptitle(title, fontsize=14, y=1.02)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting regimes: {e}")
            return False
    
    def load_spy_data(
        self,
        start_date: str = '2024-01-01',
        end_date: str = '2025-06-01'
    ) -> pd.DataFrame:
        """
        Load SPY stock data for demonstration
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading SPY data from {start_date} to {end_date}")
            spy = yf.download('SPY', start=start_date, end=end_date)
            return spy
            
        except Exception as e:
            logger.error(f"Error loading SPY data: {e}")
            return pd.DataFrame()
    
    def prepare_market_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare market features for regime detection
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Feature array for LGMM
        """
        try:
            features = []
            
            # Calculate returns
            returns = df['Close'].pct_change().dropna()
            
            # Calculate volume changes
            volume_changes = df['Volume'].pct_change().dropna()
            
            # Combine features
            feature_df = pd.DataFrame({
                'returns': returns.values,
                'volume_changes': volume_changes.values
            }).dropna()
            
            # Convert to numpy array
            feature_array = feature_df.values
            
            logger.info(f"Prepared {len(feature_array)} samples with 2 features")
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preparing market features: {e}")
            return np.array([])

def demonstrate_lgmm_spy():
    """
    Demonstration of LGMM on SPY data
    
    This demonstrates the complete workflow from data loading to regime detection
    """
    try:
        print("=" * 70)
        print("LGMM Market Regime Detection - SPY Data Demo")
        print("=" * 70)
        
        # Step 1: Load SPY data
        print("\nStep 1: Loading SPY data...")
        detector = LGMMRegimeDetector(n_components=3)
        spy_data = detector.load_spy_data()
        
        if spy_data.empty:
            print("Error: Could not load SPY data")
            return
        
        print(f"✓ Loaded {len(spy_data)} days of data")
        
        # Step 2: Prepare features
        print("\nStep 2: Preparing market features...")
        features = detector.prepare_market_features(spy_data)
        
        if len(features) == 0:
            print("Error: Could not prepare features")
            return
        
        print(f"✓ Prepared {len(features)} samples")
        
        # Step 3: Optimize components
        print("\nStep 3: Optimizing number of components...")
        optimization_results = detector.optimize_components(features, max_components=5)
        
        if 'best_n_components' in optimization_results:
            best_n = optimization_results['best_n_components']
            print(f"✓ Optimal components: {best_n}")
            detector.n_components = best_n
        
        # Step 4: Fit LGMM
        print("\nStep 4: Fitting LGMM model...")
        fit_results = detector.fit(features, normalize=True)
        
        if fit_results:
            print(f"✓ BIC Score: {fit_results.get('bic_score', 0):.2f}")
            print(f"✓ Log Likelihood: {fit_results.get('log_likelihood', 0):.4f}")
            print(f"✓ Converged: {fit_results.get('converged', False)}")
            print(f"✓ Iterations: {fit_results.get('n_iter', 0)}")
        
        # Step 5: Analyze regimes
        print("\nStep 5: Analyzing regimes...")
        for i, stats in enumerate(fit_results.get('regime_stats', [])):
            regime_type = detector.identify_regime_type(i)
            print(f"\nRegime {i} ({regime_type}):")
            print(f"  - Count: {stats['count']}")
            print(f"  - Proportion: {stats['proportion']:.1%}")
            print(f"  - Mean Volatility: {stats['mean_volatility']:.4f}")
        
        # Step 6: Visualize
        print("\nStep 6: Creating visualization...")
        success = detector.plot_regimes(
            features,
            save_path='spy_lgmm_regimes.png',
            title='SPY Market Regime Detection using LGMM'
        )
        
        if success:
            print("✓ Plot saved to spy_lgmm_regimes.png")
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in LGMM demonstration: {e}")

if __name__ == "__main__":
    demonstrate_lgmm_spy()

