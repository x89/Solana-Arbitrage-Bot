# How to Run the AI Training System

## Quick Start

### 1. Run the Demo
```bash
cd "AI_training_system(models_trainers)"
python run_training_demo.py
```

This will demonstrate:
- Configuration loading
- Sample data generation
- Model evaluation
- LGMM trainer
- System integration

### 2. Test All Imports
```bash
python test_imports.py
```

This verifies all modules import without errors.

---

## Running Individual Components

### A. Basic Training (XGBoost)

```python
from advanced_ai_trainer import XGBoostTrainer, TrainingConfig
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'price': np.random.randn(n_samples).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, n_samples),
    'rsi': np.random.uniform(0, 100, n_samples),
    'macd': np.random.randn(n_samples),
    'target': np.random.randint(0, 2, n_samples)
})

# Configure training
config = TrainingConfig(
    model_type='xgboost',
    features=['price', 'volume', 'rsi', 'macd'],
    target_column='target'
)

# Train model
trainer = XGBoostTrainer(config)
X, y = trainer.prepare_data(data)
metrics = trainer.train(X, y)

print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Score: {metrics.f1_score:.4f}")
```

### B. LSTM Training

```python
from advanced_ai_trainer import LSTMTrainer, TrainingConfig

# Configure for LSTM
config = TrainingConfig(
    model_type='lstm',
    features=['price', 'volume', 'rsi', 'macd'],
    target_column='target'
)

# Train LSTM
trainer = LSTMTrainer(config, sequence_length=60)
train_loader, test_loader = trainer.prepare_data(data)

if train_loader and test_loader:
    metrics = trainer.train(train_loader, test_loader)
    print(f"Accuracy: {metrics.accuracy:.4f}")
```

### C. LGMM Regime Detection

```python
from lgmm_trainer import LGMMTrainer
import yfinance as yf

# Initialize trainer
trainer = LGMMTrainer(n_components=3)

# Load data (e.g., SPY)
spy_data = trainer.load_spy_data(
    start_date='2024-01-01',
    end_date='2025-01-01',
    symbol='SPY'
)

# Prepare features
features = trainer.prepare_features(spy_data)

# Train model
results = trainer.train(features)

print(f"BIC Score: {results['bic_score']:.2f}")
print(f"Regime statistics:")
for regime in results['regime_stats']:
    print(f"  Regime {regime['regime']}: {regime['regime_type']} ({regime['proportion']:.1%})")
```

### D. Model Evaluation

```python
from model_evaluator import ModelEvaluator
import numpy as np

evaluator = ModelEvaluator()

# Classification metrics
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1])
y_proba = np.array([0.1, 0.9, 0.8, 0.4, 0.9])

metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC AUC: {metrics['roc_auc']:.4f}")

# Regression metrics
y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8])

metrics_reg = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
print(f"RMSE: {metrics_reg['rmse']:.4f}")
print(f"R²: {metrics_reg['r2_score']:.4f}")
```

### E. Ensemble Training

```python
from advanced_ai_trainer import EnsembleTrainer, TrainingConfig

config = TrainingConfig(
    model_type='ensemble',
    features=['price', 'volume', 'rsi', 'macd'],
    target_column='target'
)

trainer = EnsembleTrainer(config)
results = trainer.train_ensemble(data)

print("Ensemble training results:")
for model_name, metrics in results.items():
    print(f"  {model_name}: F1={metrics.f1_score:.4f}")
```

---

## Using the Integrated Trainer

```python
from integrated_trainer import IntegratedModelTrainer

# Initialize with configuration
config = {
    'use_mlflow': False,    # Set to True if mlflow installed
    'use_wandb': False,     # Set to True if wandb installed
    'use_tensorboard': False
}

trainer = IntegratedModelTrainer(config)

# List available models
print("Available models:")
print(trainer.model_registry.keys())

# Create a model
# model = trainer.create_model('lstm', input_dim=10)
```

---

## Configuration

Edit `training_config.py` to customize:

```python
TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    
    'lgmm': {
        'n_components': 3,
        'covariance_type': 'full',
        'max_iter': 100
    },
    
    # ... other configurations
}
```

---

## Complete Workflow Example

```python
# 1. Import modules
from advanced_ai_trainer import TrainingConfig, XGBoostTrainer
from model_evaluator import ModelEvaluator
import pandas as pd
import numpy as np

# 2. Load your data
df = pd.read_csv('your_data.csv')

# 3. Configure training
config = TrainingConfig(
    model_type='xgboost',
    features=['feature1', 'feature2', 'feature3'],
    target_column='target'
)

# 4. Train model
trainer = XGBoostTrainer(config)
X, y = trainer.prepare_data(df)
metrics = trainer.train(X, y)

print(f"Training completed!")
print(f"  Accuracy: {metrics.accuracy:.4f}")

# 5. Evaluate on test data
# (Add your evaluation logic here)

# 6. Save model
trainer.save_model('model.pkl')

# 7. Make predictions
# predictions = trainer.model.predict(X_test)
```

---

## File Structure

```
AI_training_system(models_trainers)/
├── training_config.py          # Configuration
├── advanced_ai_trainer.py     # XGBoost, LSTM, Transformer trainers
├── modern_ai_models.py        # Modern AI models (Transformer, TFT, etc.)
├── integrated_trainer.py      # Integrated training system
├── lgmm_trainer.py            # LGMM regime detection
├── llm_rl_hybrid.py          # LLM+RL hybrid system
├── model_evaluator.py         # Model evaluation utilities
├── model_registry.py          # Model registry
├── metrics_tracker.py         # Metrics tracking
├── run_training_demo.py       # Demo script
├── test_imports.py            # Import test
└── HOW_TO_RUN.md             # This file
```

---

## Common Tasks

### Train a Model
```python
python -c "
from advanced_ai_trainer import XGBoostTrainer, TrainingConfig
import numpy as np
import pandas as pd

# Your training code here
print('Training complete')
"
```

### Evaluate Models
```python
from model_evaluator import ModelEvaluator
evaluator = ModelEvaluator()
# Your evaluation code here
```

### Use LGMM for Regime Detection
```python
from lgmm_trainer import LGMMTrainer
trainer = LGMMTrainer(n_components=3)
# Your regime detection code here
```

---

## Tips

1. **Start with XGBoost** - Fastest and easiest to debug
2. **Use LSTM for time series** - Better for sequential patterns
3. **Use LGMM for regime detection** - Identifies market regimes
4. **Monitor metrics** - Use ModelEvaluator to track performance
5. **Save models** - Use `trainer.save_model()` to persist trained models

---

## Troubleshooting

**Import errors?**
```bash
python test_imports.py
```

**Missing dependencies?**
```bash
pip install optuna mlflow wandb tensorboard torch-geometric einops
```

**Slow training?**
- Use smaller batch size
- Reduce epochs
- Use GPU if available

**Out of memory?**
- Reduce batch size
- Use smaller sequence length
- Process data in chunks

---

## Next Steps

1. Run the demo: `python run_training_demo.py`
2. Test imports: `python test_imports.py`
3. Read README.md for detailed documentation
4. Check FIXES_SUMMARY.md for recent changes

