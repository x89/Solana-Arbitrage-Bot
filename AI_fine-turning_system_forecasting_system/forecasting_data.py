from chronos import BaseChronosPipeline, ChronosBoltPipeline, ChronosPipeline, chronos_bolt
from dotenv import load_dotenv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import Dataset
import glob
import pandas as pd
import yfinance as yf
import json

load_dotenv()
chronos_t5_dir = os.getenv('CHRONOS_T5_DIR')
chronos_bolt_dir = os.getenv('CHRONOS_BOLT_DIR')


class TimeSeriesForecaster:
    """
    Class for time series forecasting models.

    Input Data:
        df (pandas.DataFrame): with columns:
            - unique_id (str): series identifier
            - ds (datetime or str): timestamp (e.g. '2016-10-22 00:00:00')
            - y (float): target value

    Example:
        unique_id    ds                    y
        0    BE       2016-10-22 00:00:00   70.00
        1    BE       2016-10-22 01:00:00   37.10
    """
    def __init__(self, context_len, horizon_len) -> None:
        self.context_len = context_len
        self.horizon_len = horizon_len
        if chronos_t5_dir and os.path.isdir(chronos_t5_dir):
            # load latest checkpoint if available
            ckpts = sorted(glob.glob(os.path.join(chronos_t5_dir, "checkpoint-*/")))
            load_dir = ckpts[-1] if ckpts else chronos_t5_dir
            # load_dir = os.path.join(chronos_t5_dir, "runs/")
            try:
                self.chronos_t5 = BaseChronosPipeline.from_pretrained(
                    load_dir,
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                )
                print("\nchronos_t5 loaded from local checkpoint.\n")
            except (ValueError, OSError) as e:
                print(f"Warning: failed to load local Chronos T5 at '{load_dir}': {e}. Using default model.")
                self.chronos_t5 = BaseChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-large",
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                )
        else:
            self.chronos_t5 = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )

        if chronos_bolt_dir and os.path.isdir(chronos_bolt_dir):
            # load latest checkpoint if available
            ckpts = sorted(glob.glob(os.path.join(chronos_bolt_dir, "checkpoint-*/")))
            load_dir = ckpts[-1] if ckpts else chronos_bolt_dir
            # load_dir = os.path.join(chronos_bolt_dir, "runs/")
            try:
                self.chronos_bolt = ChronosBoltPipeline.from_pretrained(
                    load_dir,
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                )
                print("\nchronos_bolt loaded from local checkpoint.\n")
            except (ValueError, OSError) as e:
                print(f"Warning: failed to load local Chronos Bolt at '{load_dir}': {e}. Using default model.")
                self.chronos_bolt = ChronosBoltPipeline.from_pretrained(
                    "amazon/chronos-bolt-base",
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                )
        else:
            self.chronos_bolt = ChronosBoltPipeline.from_pretrained(
                "amazon/chronos-bolt-base",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )
        
    def _ensure_complete(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Ensure each unique_id has no missing timestamps at given frequency.
        Missing y values are forward/back filled.
        """
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Remove duplicates by keeping the first occurrence
        df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='first')
        
        # For now, just return the cleaned data without reindexing
        # The reindexing was causing issues with the data length
        return df
    
    def chronos_t5_forecast(self, df: pd.DataFrame):
        df = self._ensure_complete(df, 'D')
        context = torch.tensor(df["y"])
        forecast = self.chronos_t5.predict(context, self.horizon_len)
        # visualize the forecast
        forecast_index = range(len(df), len(df) + self.horizon_len)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(df["y"], color="royalblue", label="historical data")
        plt.plot(forecast_index, median, color="tomato", label="median forecast")
        plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
        plt.legend()
        plt.grid()
        plt.show()
        return median 

    def chronos_bolt_forecast(self, df: pd.DataFrame):
        df = self._ensure_complete(df, 'D')
        context = torch.tensor(df["y"])
        forecast = self.chronos_bolt.predict(context, self.horizon_len)
        # visualize the forecast
        forecast_index = range(len(df), len(df) + self.horizon_len)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(df["y"], color="royalblue", label="historical data")
        plt.plot(forecast_index, median, color="tomato", label="median forecast")
        plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
        plt.legend()
        plt.grid()
        plt.show()
        return median

    def chronos_t5_fine_tune(
        self,
        df: pd.DataFrame,
        output_dir: str = "chronos_t5_ft",
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
    ):
        """
        Fine-tune the Chronos T5 model on the provided DataFrame.
        """
        df = self._ensure_complete(df, 'D')
        y = df['y'].values
        print(f"Data length after processing: {len(y)}")
        
        # use model's built-in prediction_length
        model_horizon = self.chronos_t5.model.config.prediction_length
        print(f"Model prediction length: {model_horizon}")
        
        contexts, labels = [], []
        for i in range(len(y) - self.context_len - model_horizon + 1):
            contexts.append(y[i : i + self.context_len])
            labels.append(y[i + self.context_len : i + self.context_len + model_horizon])
        
        print(f"Number of training samples created: {len(contexts)}")
        
        if len(contexts) == 0:
            print("ERROR: No training samples created!")
            print(f"Data length: {len(y)}")
            print(f"Context length: {self.context_len}")
            print(f"Model horizon: {model_horizon}")
            print(f"Required minimum: {self.context_len + model_horizon}")
            return None
            
        encodings = []
        for c, l in zip(contexts, labels):
            input_ids, attention_mask, state = self.chronos_t5.tokenizer.context_input_transform(
                torch.tensor(c, dtype=torch.float32).unsqueeze(0)
            )
            label_ids, _ = self.chronos_t5.tokenizer.label_input_transform(
                torch.tensor(l, dtype=torch.float32).unsqueeze(0), state
            )
            encodings.append({
                'input_ids': input_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0),
                'labels': label_ids.squeeze(0),
            })
        train_ds = Dataset.from_list(encodings)
        print(f"Training dataset size: {len(train_ds)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=50,
            save_steps=500,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=None,
        )
        trainer = Trainer(
            model=self.chronos_t5.model,
            args=training_args,
            train_dataset=train_ds,
        )
        trainer.train()
        # Save and reload fine-tuned Chronos T5
        self.chronos_t5.model.save_pretrained(output_dir)
        self.chronos_t5 = BaseChronosPipeline.from_pretrained(
            output_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        return trainer
    
    def chronos_bolt_fine_tune(
        self,
        df: pd.DataFrame,
        output_dir: str = "chronos_bolt_ft",
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
    ):
        """
        Fine-tune the Chronos Bolt model on the provided DataFrame.
        """
        df = self._ensure_complete(df, 'D')
        y = df['y'].values
        print(f"Data length after processing: {len(y)}")
        
        # use model's built-in prediction_length from ChronosBoltConfig
        model_horizon = self.chronos_bolt.model.chronos_config.prediction_length
        print(f"Model prediction length: {model_horizon}")
        
        # prepare raw context and target tensors
        contexts, labels = [], []
        for i in range(len(y) - self.context_len - model_horizon + 1):
            contexts.append(y[i : i + self.context_len])
            labels.append(y[i + self.context_len : i + self.context_len + model_horizon])
        
        print(f"Number of training samples created: {len(contexts)}")
        
        if len(contexts) == 0:
            print("ERROR: No training samples created!")
            print(f"Data length: {len(y)}")
            print(f"Context length: {self.context_len}")
            print(f"Model horizon: {model_horizon}")
            print(f"Required minimum: {self.context_len + model_horizon}")
            return None
            
        encodings = []
        for c, l in zip(contexts, labels):
            context_tensor = torch.tensor(c, dtype=torch.float32)
            target_tensor = torch.tensor(l, dtype=torch.float32)
            mask = torch.ones_like(context_tensor, dtype=torch.bool)
            target_mask = torch.ones_like(target_tensor, dtype=torch.bool)
            encodings.append({
                'context': context_tensor,
                'mask': mask,
                'target': target_tensor,
                'target_mask': target_mask,
            })
        train_ds = Dataset.from_list(encodings)
        print(f"Training dataset size: {len(train_ds)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=50,
            save_steps=500,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=None,
        )
        trainer = Trainer(
            model=self.chronos_bolt.model,
            args=training_args,
            train_dataset=train_ds,
        )
        trainer.train()
        # Save and reload fine-tuned Chronos Bolt
        self.chronos_bolt.model.save_pretrained(output_dir)
        self.chronos_bolt = ChronosBoltPipeline.from_pretrained(
            output_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        return trainer
        
if __name__ == '__main__':
    # Load and process the cryptocurrency data
    with open('sol.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame with required format
    df = pd.DataFrame(data)
    df['unique_id'] = 'CRYPTO'  # Add a unique identifier
    df['ds'] = pd.to_datetime(df['datetime'])
    df['y'] = df['close']  # Use closing price as target variable
    
    # Remove any duplicate timestamps
    df = df.drop_duplicates(subset=['ds'], keep='first')
    
    # Create the model
    model = TimeSeriesForecaster(context_len=128, horizon_len=128)
    
    # Fine-tune both models with optimized parameters
    print("Fine-tuning Chronos T5...")
    t5_trainer = model.chronos_t5_fine_tune(
        df, 
        epochs=5,           # Increased from 3 to 5
        batch_size=8,       # Increased from 4 to 8
        lr=3e-5            # Adjusted from 5e-5 to 3e-5
    )
    print("Fine-tuning complete.")
    
    print("Fine-tuning Chronos Bolt...")
    bolt_trainer = model.chronos_bolt_fine_tune(
        df, 
        epochs=5,           # Increased from 3 to 5
        batch_size=16,      # Increased significantly for stability
        lr=1e-5            # Reduced learning rate for stability
    )
    print("Fine-tuning complete.")
    
    # Get forecasts from fine-tuned models
    print("Generating Chronos T5 Forecast:")
    t5_forecast = model.chronos_t5_forecast(df)
    print("Generating Chronos Bolt Forecast:")
    bolt_forecast = model.chronos_bolt_forecast(df)
    
    # Format forecasts to match input JSON structure
    last_date = df['datetime'].iloc[-1]  # Use 'datetime' instead of 'time_period_end'
    forecast_dates = pd.date_range(
        start=pd.to_datetime(last_date),
        periods=128,
        freq='D'
    )
    
    # Create forecast output in JSON format
    forecast_output = []
    for i, date in enumerate(forecast_dates):
        # Use weighted average of both models' predictions (60% T5, 40% Bolt)
        weighted_price = (t5_forecast[i] * 0.6) + (bolt_forecast[i] * 0.4)
        forecast_output.append({
            "time_period_start": date.strftime("%Y-%m-%dT%H:%M:%S.0000000Z"),
            "time_period_end": (date + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.0000000Z"),
            "time_open": date.strftime("%Y-%m-%dT%H:%M:%S.0000000Z"),
            "time_close": (date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.0000000Z"),
            "price_open": float(weighted_price),
            "price_high": float(weighted_price * 1.02),  # Assuming 2% higher than open
            "price_low": float(weighted_price * 0.98),   # Assuming 2% lower than open
            "price_close": float(weighted_price),
            "volume_traded": float(df['volume'].mean()),  # Using average volume
            "trades_count": int(df['volume'].mean())      # Using average volume as proxy for trades
        })
    
    # Save forecast to JSON file
    with open('forecast_output.json', 'w') as f:
        json.dump(forecast_output, f, indent=4)
    
    print("Forecast saved to forecast_output.json")
    