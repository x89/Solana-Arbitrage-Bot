# AI Training System

## Overview

Advanced AI model training system supporting multiple architectures including Chronos, TimesFM, LGMM, and LLM+RL hybrid models with comprehensive evaluation and metrics tracking.

## Features

- ✅ Multiple AI architectures (Chronos, TimesFM, LGMM)
- ✅ LLM+RL hybrid training
- ✅ Automated hyperparameter optimization with Optuna
- ✅ Model evaluation and comparison
- ✅ Metrics tracking and visualization
- ✅ Model registry for versioning
- ✅ Distributed training support
- ✅ GPU acceleration

## Files

| File | Description |
|------|-------------|
| `modern_ai_models.py` | Modern AI model implementations |
| `integrated_trainer.py` | Unified training interface |
| `lgmm_trainer.py` | LGMM regime detection training |
| `llm_rl_hybrid.py` | LLM+RL hybrid training |
| `model_registry.py` | Model versioning and registry |
| `model_evaluator.py` | Model evaluation and metrics |
| `metrics_tracker.py` | Training metrics tracking |
| `advanced_ai_trainer.py` | Advanced training features |
| `training_config.py` | Configuration settings |

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **Chronos** | Time Series | Pre-trained foundation model |
| **TimesFM-1B** | Time Series | Large-scale forecasting model |
| **LGMM** | Regime Detection | Gaussian Mixture Model for regimes |
| **LLM+RL** | Hybrid | Language model + reinforcement learning |
| **Custom ML** | Various | LSTM, Transformer, etc. |

## Quick Start

### 1. Basic Training

```python
from integrated_trainer import IntegratedTrainer

trainer = IntegratedTrainer()

# Train a model
model = trainer.train(
    model_type='chronos_t5',
    data=train_data,
    epochs=10,
    batch_size=32
)
```

### 2. With Hyperparameter Optimization

```python
from integrated_trainer import train_with_optuna

# Optimize hyperparameters
best_model = train_with_optuna(
    model_type='chronos_t5',
    data=train_data,
    n_trials=50  # Optuna trials
)

print(f"Best score: {best_model['score']}")
print(f"Best params: {best_model['params']}")
```

### 3. Evaluation

```python
from model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate model
metrics = evaluator.evaluate(
    model=model,
    test_data=test_data
)

print(f"MAE: {metrics['mae']}")
print(f"RMSE: {metrics['rmse']}")
print(f"R²: {metrics['r2']}")
```

## Usage

### Chronos Model Training

```python
from modern_ai_models import ChronosModel

# Initialize model
model = ChronosModel(
    model_size='small',
    context_length=128,
    horizon_length=64
)

# Train
trainer = IntegratedTrainer()
trained_model = trainer.train(
    model=model,
    data=train_data,
    epochs=10,
    learning_rate=3e-4
)
```

### TimesFM Training

```python
from modern_ai_models import TimesFMModel

# Initialize TimesFM
model = TimesFMModel(
    model_size='1B',  # 1 Billion parameters
    context_length=128
)

# Train
trainer = IntegratedTrainer()
trained_model = trainer.train(
    model=model,
    data=train_data,
    epochs=5,
    batch_size=16  # Smaller batch for large model
)
```

### LGMM Regime Detection

```python
from lgmm_trainer import LGMTrainer

trainer = LGMTrainer()

# Train regime detection model
model = trainer.train(
    data=train_data,
    n_regimes=3,  # Bull, Bear, Neutral
    iterations=100
)

# Predict regimes
regimes = model.predict_regime(data)
```

### LLM+RL Hybrid

```python
from llm_rl_hybrid import LLMRLTrainer

trainer = LLMRLTrainer(
    llm_model='gpt-3.5-turbo',
    task='trading'
)

# Train with RL
model = trainer.train(
    data=train_data,
    episodes=100,
    learning_rate=1e-4
)
```

## Configuration

Edit `training_config.py`:

```python
class TrainingConfig:
    # Model Settings
    DEFAULT_MODEL = 'chronos_t5'
    CONTEXT_LENGTH = 128
    HORIZON_LENGTH = 64
    
    # Training Settings
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    
    # GPU Settings
    CUDA_ENABLED = True
    MIXED_PRECISION = True
    
    # Optuna Settings
    OPTUNA_TRIALS = 50
    OPTUNA_TIMEOUT = 3600  # seconds
```

## Model Registry

```python
from model_registry import ModelRegistry

registry = ModelRegistry()

# Save model
registry.save_model(
    model=trained_model,
    name='chronos_t5_v1',
    metadata={'score': 0.85, 'version': '1.0'}
)

# Load model
model = registry.load_model('chronos_t5_v1')

# List all models
models = registry.list_models()
```

## Evaluation Metrics

### Regression Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### Time Series Metrics
- Directional Accuracy
- Forecast Horizon Error
- Trend Preservation

## Requirements

```
torch>=1.12.0
transformers>=4.30.0
lightgbm>=3.3.0
xgboost>=1.6.0
optuna>=3.0.0
scikit-learn>=1.0.0
```

## Best Practices

1. **Use GPU** - Enable CUDA for faster training
2. **Normalize data** - Ensure data is properly scaled
3. **Validation split** - Use 20% for validation
4. **Early stopping** - Prevent overfitting
5. **Hyperparameter optimization** - Find best parameters
6. **Model versioning** - Track different model versions
7. **Evaluation metrics** - Monitor multiple metrics

## Troubleshooting

**Issue**: Training too slow
- Use GPU acceleration
- Reduce batch size
- Use mixed precision training
- Enable data loading workers

**Issue**: Overfitting
- Add regularization
- Use dropout
- Early stopping
- Data augmentation

**Issue**: Out of memory
- Reduce batch size
- Use gradient checkpointing
- Use smaller model
- Clear GPU cache

