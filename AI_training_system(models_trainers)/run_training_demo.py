#!/usr/bin/env python3
"""
Training System Demo
Demonstrates how to use the AI training system
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("AI Training System Demo")
    print("=" * 70)
    
    # Import training system components
    try:
        from training_config import TRAINING_CONFIG
        from advanced_ai_trainer import XGBoostTrainer, LSTMTrainer, TrainingConfig
        from model_evaluator import ModelEvaluator
        from lgmm_trainer import LGMMTrainer
        print("\n[SUCCESS] All core modules imported successfully")
    except Exception as e:
        print(f"\n[ERROR] Failed to import modules: {e}")
        return
    
    print("\n" + "=" * 70)
    print("Demo 1: Configuration")
    print("=" * 70)
    
    print(f"\nConfiguration loaded:")
    print(f"  - Default epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  - Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  - Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"\nLGMM Configuration:")
    lgmm_config = TRAINING_CONFIG['lgmm']
    print(f"  - Components: {lgmm_config['n_components']}")
    print(f"  - Covariance type: {lgmm_config['covariance_type']}")
    print(f"  - Features: {lgmm_config['features']}")
    
    print("\n" + "=" * 70)
    print("Demo 2: Create Sample Data")
    print("=" * 70)
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with momentum indicators
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    # Price data with trend
    price_data = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    # Volume data
    volume_data = np.random.randint(100000, 1000000, n_samples)
    
    # Technical indicators
    data = pd.DataFrame({
        'date': dates,
        'price': price_data,
        'volume': volume_data,
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.randn(n_samples) * 2,
        'sma_20': price_data + np.random.randn(n_samples) * 0.1,
        'sma_50': price_data + np.random.randn(n_samples) * 0.2,
        'bb_position': np.random.uniform(0, 1, n_samples),
        'momentum': np.random.uniform(-1, 1, n_samples),
        'target': np.random.randint(0, 2, n_samples)  # Binary classification target
    })
    
    print(f"\nGenerated sample data:")
    print(f"  - Records: {len(data)}")
    print(f"  - Columns: {list(data.columns)}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print("\n" + "=" * 70)
    print("Demo 3: Model Evaluation")
    print("=" * 70)
    
    try:
        evaluator = ModelEvaluator()
        print("\nModelEvaluator initialized successfully")
        
        # Demonstrate evaluation with dummy data
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.8, 0.6, 0.9, 0.3, 0.2, 0.85])
        
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
        print(f"\nClassification Metrics:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
    except Exception as e:
        print(f"[ERROR] Model evaluation failed: {e}")
    
    print("\n" + "=" * 70)
    print("Demo 4: LGMM Trainer")
    print("=" * 70)
    
    try:
        trainer = LGMMTrainer(n_components=3)
        print("\nLGMM Trainer initialized")
        print(f"  - Components: {trainer.n_components}")
        print(f"  - Covariance type: {trainer.covariance_type}")
        
        # Prepare features
        returns = data['price'].pct_change().dropna().values
        volume_changes = data['volume'].pct_change().dropna().values
        
        # Pad to same length
        min_len = min(len(returns), len(volume_changes))
        features = np.column_stack([
            returns[:min_len],
            volume_changes[:min_len]
        ])
        
        print(f"\nFeatures prepared: {features.shape}")
        print(f"  - Feature 1: Returns (mean={features[:, 0].mean():.4f}, std={features[:, 0].std():.4f})")
        print(f"  - Feature 2: Volume changes (mean={features[:, 1].mean():.4f}, std={features[:, 1].std():.4f})")
        
        # Train model (commented out to avoid long run time in demo)
        # print("\nTraining LGMM model...")
        # results = trainer.train(features)
        # print(f"\nTraining completed:")
        # print(f"  - BIC Score: {results.get('bic_score', 0):.2f}")
        # print(f"  - Converged: {results.get('converged', False)}")
        
        print("\n[INFO] LGMM training skipped in demo (uncomment to run)")
        
    except Exception as e:
        print(f"[ERROR] LGMM demo failed: {e}")
    
    print("\n" + "=" * 70)
    print("Demo 5: Integration")
    print("=" * 70)
    
    try:
        from integrated_trainer import IntegratedModelTrainer
        
        config = {
            'use_mlflow': False,
            'use_wandb': False,
            'use_tensorboard': False
        }
        
        integrated_trainer = IntegratedModelTrainer(config)
        print("\nIntegratedModelTrainer initialized")
        print(f"  - Modern models available: {integrated_trainer.modern_trainer is not None}")
        print(f"  - Registered models: {list(integrated_trainer.model_registry.keys())}")
        
    except Exception as e:
        print(f"[ERROR] Integration demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    print("\nTo use the training system:")
    print("  1. Prepare your training data (DataFrame with features)")
    print("  2. Choose a trainer (XGBoost, LSTM, LGMM, etc.)")
    print("  3. Configure training parameters")
    print("  4. Train and evaluate the model")
    print("\nExample:")
    print("  from advanced_ai_trainer import XGBoostTrainer, TrainingConfig")
    print("  trainer = XGBoostTrainer(config)")
    print("  metrics = trainer.train(X, y)")
    
    print("\nFor more details, see:")
    print("  - README.md")
    print("  - FIXES_SUMMARY.md")
    
if __name__ == "__main__":
    main()

