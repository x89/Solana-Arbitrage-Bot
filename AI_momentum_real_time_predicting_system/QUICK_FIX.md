# Quick Fix for Installation Issues

Based on your error output, here are the commands to fix the missing packages:

## Step 1: Install XGBoost and LightGBM

```bash
pip install xgboost lightgbm
```

If this fails, try:
```bash
# For XGBoost
pip install --upgrade pip
pip install xgboost

# For LightGBM
pip install lightgbm
```

## Step 2: Install TA-Lib (Windows)

You have three options:

### Option A: Use Pre-built Wheel (Easiest for Windows)

1. **Find your Python version:**
   ```bash
   python --version
   ```

2. **Download the matching wheel file from:**
   https://github.com/cgohlke/talib-build/releases/latest

   Look for: `TA_Lib-0.4.25-cpXX-cpXX-win_amd64.whl`
   - Replace `XX` with your Python version (39, 310, 311, etc.)

3. **Install it:**
   ```bash
   pip install C:\path\to\TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
   ```

### Option B: Use Conda (If you have Anaconda)

```bash
conda install -c conda-forge ta-lib
```

### Option C: Skip TA-Lib for Now

If you can't install TA-Lib, the system will still work but without technical indicators. You can:
- Use the system for momentum calculations that don't require TA-Lib
- Focus on using LSTM/Transformer models that don't need technical indicators
- Install TA-Lib later when you have the wheel file

## Step 3: Verify Installation

```bash
python check_installation.py
```

## Step 4: Test Modules

```bash
python test_modules.py
```

## Alternative: Minimal Installation

If you want to run the system without all features, you can skip the problematic packages:

```bash
# Install only essential packages
pip install torch numpy pandas scikit-learn joblib yfinance matplotlib

# The system will work but without:
# - XGBoost predictions
# - LightGBM predictions  
# - Technical indicators (if TA-Lib is missing)
```

## Running Without Missing Packages

The system is designed to work partially even without some packages:

- **Without XGBoost/LightGBM**: Will still work with LSTM, Transformer, and Random Forest
- **Without TA-Lib**: Will still work but technical indicator calculations will fail
- **Without TensorFlow**: PyTorch is used by default anyway

## Next Steps

1. Install XGBoost and LightGBM (Step 1)
2. Install TA-Lib (Step 2)
3. Verify installation (Step 3)
4. Run test modules (Step 4)
5. Run full demo: `python run_main.py`

## Troubleshooting

### "Failed building wheel for pickle5"
**This is OK!** pickle5 is only needed for Python 2.7. I've removed it from requirements.txt. You don't need it.

### "XGBoost installation failed"
Try installing Visual C++ Redistributables:
- Download from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
- Then retry: `pip install xgboost`

### "LightGBM installation failed"
Similar to XGBoost, make sure you have Visual C++ build tools, then:
```bash
pip install lightgbm --no-cache-dir
```

### "TA-Lib installation failed"  
This is common on Windows. Use Option A (pre-built wheel) from Step 2 above.

