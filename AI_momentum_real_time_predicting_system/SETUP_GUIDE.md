# Setup Guide for AI Momentum Real-time Predicting System

## Step-by-Step Installation

### Step 1: Check Current Environment

First, verify what you have:

```bash
python --version  # Should be 3.8 or higher
pip --version
```

### Step 2: Install Base Python Packages

```bash
pip install -r requirements.txt
```

### Step 3: Special Installation for TA-Lib

**TA-Lib is required for technical indicators calculation.**

#### Windows (Recommended Method):
1. Download pre-built wheel from: https://github.com/cgohlke/talib-build/releases
2. Find the file matching your Python version:
   - Python 3.9: `TA_Lib-0.4.25-cp39-cp39-win_amd64.whl`
   - Python 3.10: `TA_Lib-0.4.25-cp310-cp310-win_amd64.whl`
   - Python 3.11: `TA_Lib-0.4.25-cp311-cp311-win_amd64.whl`
3. Install the wheel file:
   ```bash
   pip install path\to\TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
   ```

**Alternative for Windows - Using Conda:**
If you have Anaconda/Miniconda installed:
```bash
conda install -c conda-forge ta-lib
```

**Alternative - Using pip (may not work on Windows):**
```bash
pip install TA-Lib
```

#### Linux:
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

#### Mac:
```bash
brew install ta-lib
pip install TA-Lib
```

### Step 4: Verify Installation

```bash
python check_installation.py
```

This should show all packages as installed. If any are missing, install them.

### Step 5: Test Module Imports

```bash
python test_modules.py
```

This verifies all Python modules can be imported.

### Step 6: Run Full Demo

```bash
python run_main.py
```

This runs a complete demonstration of the system.

## Troubleshooting

### Issue: TA-Lib installation fails

**Windows:**
- Download the pre-built wheel file from the link above
- Make sure you download the version matching your Python version (cp39, cp310, etc.)
- Make sure you download the architecture (win32 or win_amd64)

**Linux:**
- Make sure you have build tools installed: `sudo apt-get install build-essential`
- Make sure you have the ta-lib library installed: `sudo apt-get install libta-lib-dev`

**Mac:**
- Install Homebrew first: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- Then run: `brew install ta-lib`

### Issue: Import errors for specific modules

If you get import errors:

1. Check that the package is installed:
   ```bash
   pip list | grep package_name
   ```

2. If missing, install it:
   ```bash
   pip install package_name
   ```

3. If still having issues, try reinstalling:
   ```bash
   pip uninstall package_name
   pip install package_name
   ```

### Issue: CUDA/GPU errors

The system will automatically use CPU if GPU is not available. To use GPU:

1. Install PyTorch with CUDA:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Issue: Memory errors during training

If you encounter out-of-memory errors:

1. Reduce batch size in training configuration
2. Reduce the number of features
3. Use CPU instead of GPU
4. Increase system RAM or use a machine with more memory

## Using the System

### Example 1: Calculate Momentum Indicators

```python
from momentum_calculator import MomentumCalculator
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate indicators
calculator = MomentumCalculator(df)
indicators = calculator.calculate_all_indicators(df)

print(f"Calculated {len(indicators)} indicators")
```

### Example 2: Train a Model

```python
from momentum_trainer import MomentumTrainer
from config import CONFIG

trainer = MomentumTrainer()

# Prepare data
train_loader, test_loader, scaler = trainer.prepare_data(df, target_column='momentum')

# Train LSTM
model, metrics = trainer.train_lstm_model(
    train_loader, 
    test_loader, 
    input_dim=len(df.columns) - 1  # number of features
)

print(f"Training complete: {metrics}")
```

### Example 3: Generate Prediction

```python
from momentum_predictor import MomentumManager

manager = MomentumManager()

# Predict momentum
prediction = manager.predict_momentum(df, symbol='SOLUSDT', timeframe='1h')

if prediction:
    print(f"Prediction: {prediction.predicted_momentum:.4f}")
    print(f"Confidence: {prediction.confidence:.2f}")
```

## Configuration

Edit `config.py` to customize the system:

- **MODEL_CONFIGS**: Adjust model hyperparameters
- **ENSEMBLE_CONFIG**: Set ensemble weights
- **FEATURE_CONFIG**: Enable/disable features
- **PERFORMANCE_THRESHOLDS**: Set accuracy requirements

## Support

If you encounter issues:

1. Check the error message carefully
2. Verify all dependencies are installed
3. Check the logs for detailed error information
4. Make sure you have sufficient data (at least 100 samples)
5. Check that your data has the required columns (open, high, low, close, volume)

## Next Steps

After successful installation:

1. **Customize Configuration**: Edit `config.py` for your needs
2. **Prepare Your Data**: Format your market data properly
3. **Train Models**: Use `momentum_trainer.py` to train models
4. **Generate Predictions**: Use `momentum_predictor.py` for predictions
5. **Validate Results**: Use `prediction_validator.py` to check accuracy

