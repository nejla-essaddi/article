# lstm_forecaster_improved.py

# -*- coding: utf-8 -*-
"""
Long-Term (30-Year) Prediction Script - LSTM Model

This script implements and runs an LSTM model for long-term time-series forecasting,
including Monte Carlo simulations for worst/best-case scenarios and
visualizations of future feature assumptions and 3D impact.

This script is adapted from a similar implementation using the Informer model,
retaining the advanced features like cross-validation and exogenous feature strategies.

## Citation for Core Model:
This script uses the LSTM model, a foundational architecture for sequence modeling.

1.  **LSTM**:
    - Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
      Neural computation, 9(8), 1735-1780.

## Key Features:
1. Enhanced exogenous feature forecasting with multiple strategies.
2. Cross-validation for robust residual calculation.
3. Early stopping to prevent overfitting.
"""

# --- 1. Imports and Configuration ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf # Keep for tqdm_callback if needed
import warnings
import os
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings('ignore')

# --- Global Configurations with Best Parameters for LSTM ---
BEST_CONFIG_LSTM = {
    'model_type': 'lstm',
    'features': ['ALLSKY_KT', 'T2M', 'RH2M'],
    'target': 'GHI',
    'seq_len': 12, # Use 12 months of data to predict the next one
    'params': {
        'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2,
        'learning_rate': 0.0001, 'batch_size': 32, 'epochs': 100,
        'patience': 10, 'min_delta': 0.0001
    }
}
MONTE_CARLO_SIMULATIONS = 10000

# --- Styles, Seeds, and Device ---
plt.style.use('seaborn-v0_8-whitegrid')
tf.random.set_seed(42)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Configuration loaded | PyTorch Device: {device}")

# --- 2. Model Classes (LSTM) ---

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # We only need the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        # Pass the last output through the linear layer
        out = self.linear(last_time_step_out)
        return out

class DatasetForLSTM(Dataset):
    def __init__(self, data, seq_len, feature_indices, target_idx):
        self.data = data
        self.seq_len = seq_len
        self.feature_indices = feature_indices
        self.target_idx = target_idx
        self.length = len(data) - seq_len

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError("Index out of range")
        
        # Input sequence (features)
        seq_x = self.data[i : i + self.seq_len, self.feature_indices]
        
        # Target value (at the end of the sequence)
        seq_y = self.data[i + self.seq_len, self.target_idx]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor([seq_y])
    
    def __len__(self): 
        return self.length

# --- 3. Enhanced Exogenous Feature Forecasting ---
def forecast_exogenous_features(df_full, feature_cols, n_steps=360, strategy='enhanced_monthly_avg'):
    """
    🌍 AMÉLIORATION 1: Représentation Améliorée des Caractéristiques Exogènes Futures
    """
    print(f"🌍 Forecasting exogenous features using strategy: {strategy}")
    
    available_feature_cols = [col for col in feature_cols if col in df_full.columns]
    if not available_feature_cols:
        print("❌ No valid feature columns found in the dataset")
        return None
    
    monthly_avg_features = df_full[available_feature_cols].groupby(df_full.index.month).mean()
    monthly_std_features = df_full[available_feature_cols].groupby(df_full.index.month).std()
    
    future_dates = pd.date_range(start=df_full.index[-1], periods=n_steps + 1, freq='M')[1:]
    future_features_df = pd.DataFrame(index=future_dates, columns=available_feature_cols)
    
    trends = {}
    for col in available_feature_cols:
        years = df_full.index.year.values
        values = df_full[col].values
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) > 1:
            slope, intercept, _, _, _ = stats.linregress(years[valid_mask], values[valid_mask])
            trends[col] = slope
        else:
            trends[col] = 0

    if strategy == 'monthly_avg':
        for date in future_dates:
            future_features_df.loc[date] = monthly_avg_features.loc[date.month]
    
    elif strategy == 'linear_trend_monthly_avg':
        for date in future_dates:
            base_values = monthly_avg_features.loc[date.month]
            years_ahead = date.year - df_full.index[-1].year
            for col in available_feature_cols:
                trend_adjustment = trends[col] * years_ahead
                future_features_df.loc[date, col] = base_values[col] + trend_adjustment
    
    elif strategy == 'stochastic_monthly_avg':
        for date in future_dates:
            base_values = monthly_avg_features.loc[date.month]
            std_values = monthly_std_features.loc[date.month]
            for col in available_feature_cols:
                std_val = std_values[col] if not np.isnan(std_values[col]) else 0
                noise = np.random.normal(0, std_val * 0.5)
                future_features_df.loc[date, col] = base_values[col] + noise
    
    elif strategy == 'enhanced_monthly_avg':
        for date in future_dates:
            base_values = monthly_avg_features.loc[date.month]
            std_values = monthly_std_features.loc[date.month]
            years_ahead = date.year - df_full.index[-1].year
            for col in available_feature_cols:
                trend_adjustment = trends[col] * years_ahead
                std_val = std_values[col] if not np.isnan(std_values[col]) else 0
                noise = np.random.normal(0, std_val * 0.3)
                future_features_df.loc[date, col] = base_values[col] + trend_adjustment + noise
    
    return future_features_df


# --- 4. Prediction Orchestration Class ---
class LongTermForecaster:
    def __init__(self, file_path, model_config):
        self.file_path = file_path
        self.config = model_config
        self.model_type = self.config['model_type']
        self.scaler = StandardScaler()
        self.model = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.feature_cols = list(self.config['features'])
        self.full_cols = self.feature_cols + [self.config['target']]

    def _load_and_prepare_data(self):
        try:
            df = pd.read_excel(self.file_path, parse_dates=['YEAR']).rename(columns={'YEAR': 'datetime'}).set_index('datetime').sort_index()
        except Exception as e:
            print(f"Error reading the Excel file: {e}")
            return None
        df_monthly = df.resample('M').mean(numeric_only=True)
        if not all(col in df_monthly.columns for col in self.config['features'] + [self.config['target']]):
            print(f"Error: Missing columns.")
            return None
        print(f"[{self.model_type.upper()}] ✅ Data loaded. Range: {df_monthly.index.min()} to {df_monthly.index.max()}")
        return df_monthly[self.full_cols].dropna()

    def _early_stopping_check(self, val_loss):
        if val_loss < self.best_val_loss - self.config['params'].get('min_delta', 0.0001):
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['params'].get('patience', 10):
                return True
        return False

    def train(self, df_train):
        print(f"[{self.model_type.upper()}] 🤖 Starting training with early stopping...")
        
        min_required_length = self.config['seq_len'] + 10
        if len(df_train) < min_required_length * 2: # Ensure enough data for train and validation
            print(f"[{self.model_type.upper()}] ❌ Insufficient data for training and validation split: {len(df_train)} points")
            return

        train_size = int(len(df_train) * 0.8)
        df_train_split, df_val_split = df_train.iloc[:train_size], df_train.iloc[train_size:]

        data_scaled = self.scaler.fit_transform(df_train_split)
        val_data_scaled = self.scaler.transform(df_val_split)
        
        params = self.config['params']
        feature_indices = [self.full_cols.index(c) for c in self.feature_cols]
        target_idx = self.full_cols.index(self.config['target'])
        
        train_dataset = DatasetForLSTM(data_scaled, self.config['seq_len'], feature_indices, target_idx)
        val_dataset = DatasetForLSTM(val_data_scaled, self.config['seq_len'], feature_indices, target_idx)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"[{self.model_type.upper()}] ❌ Dataset is empty after processing. Check data length and seq_len.")
            return
            
        train_loader = DataLoader(train_dataset, batch_size=min(params['batch_size'], len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(params['batch_size'], len(val_dataset)), shuffle=False)
        
        self.model = LSTMModel(
            input_size=len(self.feature_cols),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=1,
            dropout=params['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        
        for epoch in tqdm(range(params['epochs']), desc=f"Training {self.model_type.upper()}"):
            self.model.train()
            for x_true, y_true in train_loader:
                optimizer.zero_grad()
                output = self.model(x_true.to(device))
                loss = criterion(output, y_true.to(device))
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_true, y_true in val_loader:
                    output = self.model(x_true.to(device))
                    loss = criterion(output, y_true.to(device))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if self._early_stopping_check(val_loss):
                print(f"[{self.model_type.upper()}] ⏰ Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"[{self.model_type.upper()}] ✅ Training finished.")


    def predict_future(self, df_full, n_steps=360, exog_strategy='enhanced_monthly_avg'):
        print(f"[{self.model_type.upper()}] 🔮 Generating predictions for {n_steps} months...")
        if self.model is None:
            print(f"[{self.model_type.upper()}] ❌ Model not trained yet")
            return None
            
        future_features_df = forecast_exogenous_features(df_full, self.feature_cols, n_steps, exog_strategy)
        if future_features_df is None:
            return None
        
        data_scaled = self.scaler.transform(df_full)
        current_sequence_scaled = list(data_scaled[-self.config['seq_len']:])
        future_preds_scaled = []
        
        future_dates = pd.date_range(start=df_full.index[-1], periods=n_steps + 1, freq='M')[1:]
        feature_indices = [self.full_cols.index(c) for c in self.feature_cols]
        target_idx = self.full_cols.index(self.config['target'])

        for i in tqdm(range(n_steps), desc=f"Predicting {self.model_type.upper()}"):
            try:
                input_seq = np.array([row[feature_indices] for row in current_sequence_scaled])
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(input_tensor)
                next_pred_scaled = prediction[0, 0].item()
                future_preds_scaled.append(next_pred_scaled)

                current_date = future_dates[i]
                next_step_features = future_features_df.loc[current_date].to_dict()
                next_step_full = next_step_features
                next_step_full[self.config['target']] = next_pred_scaled # Placeholder for scaling

                next_step_df = pd.DataFrame([next_step_full], columns=self.full_cols)
                next_step_scaled = self.scaler.transform(next_step_df)[0]
                
                # Correctly set the predicted value in the scaled array
                next_step_scaled[target_idx] = next_pred_scaled

                current_sequence_scaled.pop(0)
                current_sequence_scaled.append(next_step_scaled)
                
            except Exception as e:
                print(f"[{self.model_type.upper()}] ❌ Error at step {i}: {str(e)}")
                break

        if len(future_preds_scaled) == 0: return None
        
        dummy_array = np.zeros((len(future_preds_scaled), len(self.full_cols)))
        dummy_array[:, target_idx] = future_preds_scaled
        future_preds_denorm = self.scaler.inverse_transform(dummy_array)[:, target_idx]
        
        pred_dates = future_dates[:len(future_preds_denorm)]
        return pd.Series(future_preds_denorm, index=pred_dates)

    def get_residuals_cross_validation(self, df, n_splits=5):
        """
        🎲 AMÉLIORATION 2: Validation Croisée pour le Calcul Robuste des Résidus
        """
        print(f"[{self.model_type.upper()}] 🧮 Calculating residuals using cross-validation...")
        
        min_required_length = self.config['seq_len'] + 20 # Extra margin
        if len(df) < min_required_length * n_splits:
            print(f"[{self.model_type.upper()}] ⚠️ Not enough data for {n_splits}-fold CV, using simple split")
            return self.get_residuals(df)
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_residuals = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            print(f"[{self.model_type.upper()}] 📊 Processing fold {fold + 1}/{n_splits}")
            df_train_fold, df_test_fold = df.iloc[train_idx], df.iloc[test_idx]
            
            if len(df_train_fold) < min_required_length or len(df_test_fold) < 1:
                print(f"[{self.model_type.upper()}] ⚠️ Skipping fold {fold + 1} due to insufficient data")
                continue
            
            fold_forecaster = LongTermForecaster(self.file_path, self.config)
            fold_forecaster.train(df_train_fold.copy())
            
            if fold_forecaster.model is None:
                print(f"[{self.model_type.upper()}] ⚠️ Skipping fold {fold + 1} due to training failure")
                continue
            
            try:
                preds = fold_forecaster.predict_future(df_train_fold.copy(), 
                                                     n_steps=len(df_test_fold), 
                                                     exog_strategy='enhanced_monthly_avg')
                if preds is None: continue
                fold_residuals = df_test_fold[self.config['target']].values[:len(preds)] - preds.values
                all_residuals.extend(fold_residuals)
            except Exception as e:
                print(f"[{self.model_type.upper()}] ⚠️ Error in fold {fold + 1}: {str(e)}")
        
        if len(all_residuals) < 10:
            print(f"[{self.model_type.upper()}] ⚠️ Not enough residuals from CV, falling back to simple split")
            return self.get_residuals(df)
        
        all_residuals = np.array(all_residuals)
        print(f"[{self.model_type.upper()}] ✅ Residuals calculated using cross-validation on {len(all_residuals)} points.")
        return all_residuals

    def run_monte_carlo(self, base_predictions, residuals):
        print(f"[{self.model_type.upper()}] 🇲🇨 Running Monte Carlo for worst/best-case scenarios...")
        n_sims = MONTE_CARLO_SIMULATIONS
        sim_matrix = base_predictions.values[:, np.newaxis] + np.random.choice(residuals, (len(base_predictions), n_sims), replace=True)
        sim_matrix[sim_matrix < 0] = 0

        return pd.DataFrame({
            'mean': np.mean(sim_matrix, axis=1),
            'p1': np.percentile(sim_matrix, 1, axis=1),
            'p5': np.percentile(sim_matrix, 5, axis=1),
            'p95': np.percentile(sim_matrix, 95, axis=1),
            'p99': np.percentile(sim_matrix, 99, axis=1)
        }, index=base_predictions.index)

    def get_residuals(self, df):
        """ Fallback method for calculating residuals on a simple train/test split. """
        print(f"[{self.model_type.upper()}] 🧮 Calculating historical residuals (simple split)...")
        train_size = int(len(df) * 0.8)
        df_train_res, df_test_res = df.iloc[:train_size], df.iloc[train_size:]
        residual_forecaster = LongTermForecaster(self.file_path, self.config)
        residual_forecaster.train(df_train_res.copy())
        preds = residual_forecaster.predict_future(df_train_res.copy(), n_steps=len(df_test_res))
        if preds is None:
            print(f"[{self.model_type.upper()}] ❌ Failed to generate predictions for residuals.")
            return np.array([0]) # Return a non-empty array with zero error
        residuals = df_test_res[self.config['target']].values[:len(preds)] - preds.values
        print(f"[{self.model_type.upper()}] ✅ Residuals calculated on {len(residuals)} points.")
        return residuals


# --- 5. Utility and Visualization Functions ---
def plot_monte_carlo_results(historical_data, mc_results, model_name):
    print(f"\n📊 Generating Monte Carlo graphs for {model_name}...")
    output_dir = "results_lstm_improved"
    os.makedirs(output_dir, exist_ok=True)
    target = historical_data.columns[-1]

    plt.figure(figsize=(15, 7))
    plt.plot(historical_data.index, historical_data[target], label='Historical Data', color='gray', alpha=0.6)
    plt.plot(mc_results.index, mc_results['mean'], label=f'Mean Prediction ({model_name})', color='blue', linestyle='--')
    plt.fill_between(mc_results.index, mc_results['p5'], mc_results['p95'], color='blue', alpha=0.3, label='90% Confidence Interval')
    plt.fill_between(mc_results.index, mc_results['p1'], mc_results['p5'], color='red', alpha=0.15, label='Worst Case (1st percentile)')
    plt.fill_between(mc_results.index, mc_results['p95'], mc_results['p99'], color='green', alpha=0.15, label='Best Case (99th percentile)')

    plt.title(f'Monte Carlo Simulation for {model_name} over 30 Years (Extreme Scenarios)')
    plt.xlabel('Year')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(output_dir, f"monte_carlo_forecast_{model_name.lower()}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_future_feature_assumptions(future_features, model_name):
    print(f"\n🌍 Generating future feature assumptions plot for {model_name}...")
    output_dir = "results_lstm_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    if future_features is None or future_features.empty:
        print("❌ No future features data to plot")
        return
    
    n_features = len(future_features.columns)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    for i, column in enumerate(future_features.columns):
        ax = axes_flat[i]
        ax.plot(future_features.index, future_features[column], color='orange', alpha=0.7)
        ax.set_title(f'Future {column} Assumptions')
        ax.set_xlabel('Date')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
    
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'future_feature_assumptions_{model_name.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_impact_analysis(mc_results, model_name):
    print(f"\n🎯 Generating 3D impact analysis for {model_name}...")
    output_dir = "results_lstm_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    years = mc_results.index.year
    months = mc_results.index.month
    unique_years = sorted(list(set(years)))
    unique_months = list(range(1, 13))
    
    X, Y = np.meshgrid(unique_months, unique_years)
    Z = np.full(X.shape, np.nan) # Use NaN for missing values
    
    for i, year in enumerate(unique_years):
        for j, month in enumerate(unique_months):
            mask = (years == year) & (months == month)
            if mask.any():
                Z[i, j] = mc_results.loc[mask, 'mean'].iloc[0]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_zlabel('GHI (kWh/m²)')
    ax.set_title(f'3D Impact Analysis - {model_name} Predictions')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='GHI (kWh/m²)')
    
    plt.savefig(os.path.join(output_dir, f'3d_impact_analysis_{model_name.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(historical_data, mc_results, model_name, exog_strategy):
    print(f"\n📄 Generating detailed report for {model_name}...")
    output_dir = "results_lstm_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    target = historical_data.columns[-1]
    historical_mean = historical_data[target].mean()
    historical_std = historical_data[target].std()
    
    future_mean = mc_results['mean'].mean()
    future_std = mc_results['mean'].std()
    
    years = mc_results.index.year
    values = mc_results['mean'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    
    monthly_means = mc_results.groupby(mc_results.index.month)['mean'].mean()
    seasonal_variability = monthly_means.std()
    
    report = f"""
# Detailed Report: Long-Term GHI Forecast ({model_name})

## Model Configuration
- **Model**: {model_name}
- **Exogenous Feature Strategy**: {exog_strategy}
- **Forecast Period**: 30 years ({mc_results.index.min().strftime('%Y-%m')} to {mc_results.index.max().strftime('%Y-%m')})
- **Monte Carlo Simulations**: {MONTE_CARLO_SIMULATIONS:,}

## Historical Statistics
- **Historical Mean**: {historical_mean:.2f} kWh/m²
- **Historical Std Dev**: {historical_std:.2f} kWh/m²
- **Historical Period**: {historical_data.index.min().strftime('%Y-%m')} to {historical_data.index.max().strftime('%Y-%m')}

## Forecast Results
- **Future Mean**: {future_mean:.2f} kWh/m²
- **Future Std Dev**: {future_std:.2f} kWh/m²
- **Relative Change**: {((future_mean - historical_mean) / historical_mean * 100):+.1f}%

## Trend Analysis
- **Trend Slope**: {slope:.4f} kWh/m²/year
- **Correlation (R-value)**: {r_value:.3f}
- **P-value**: {p_value:.6f}
- **Standard Error**: {std_err:.4f}

## Confidence Intervals
- **90% Confidence Interval**: [{mc_results['p5'].mean():.2f}, {mc_results['p95'].mean():.2f}] kWh/m²
- **Worst Case (1st percentile)**: {mc_results['p1'].mean():.2f} kWh/m²
- **Best Case (99th percentile)**: {mc_results['p99'].mean():.2f} kWh/m²

## Seasonal Variability
- **Seasonal Std Dev**: {seasonal_variability:.2f} kWh/m²
- **Peak Month**: {monthly_means.idxmax()} ({monthly_means.max():.2f} kWh/m²)
- **Lowest Month**: {monthly_means.idxmin()} ({monthly_means.min():.2f} kWh/m²)

## Predicted Monthly Averages
"""
    
    for month in range(1, 13):
        month_name = pd.Timestamp(2024, month, 1).strftime('%B')
        month_mean = monthly_means.loc[month]
        report += f"- **{month_name}**: {month_mean:.2f} kWh/m²\n"
    
    report_path = os.path.join(output_dir, f'detailed_report_{model_name.lower()}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Report saved to {report_path}")

# --- 6. Main Execution ---
def main():
    print("🚀 Starting long-term GHI forecast with LSTM model")
    print("=" * 80)
    
    file_path = "dataset500.xlsx"
    config = BEST_CONFIG_LSTM
    
    if not os.path.exists(file_path):
        print(f"❌ Error: The file {file_path} does not exist.")
        return

    forecaster = LongTermForecaster(file_path, config)
    
    print("\n1️⃣ Loading and preparing data...")
    df_full = forecaster._load_and_prepare_data()
    if df_full is None: return

    print("\n2️⃣ Training the model with early stopping...")
    forecaster.train(df_full.copy())
    
    print("\n3️⃣ Calculating residuals with cross-validation...")
    residuals = forecaster.get_residuals_cross_validation(df_full.copy())
    
    print("\n4️⃣ Generating future predictions...")
    exog_strategy = 'enhanced_monthly_avg'
    future_predictions = forecaster.predict_future(df_full.copy(), n_steps=360, exog_strategy=exog_strategy)
    if future_predictions is None:
        print("❌ Prediction failed. Aborting.")
        return
        
    future_features = forecast_exogenous_features(df_full, forecaster.feature_cols, n_steps=360, strategy=exog_strategy)
    
    print("\n5️⃣ Running Monte Carlo simulation...")
    mc_results = forecaster.run_monte_carlo(future_predictions, residuals)
    
    print("\n6️⃣ Generating visualizations...")
    plot_monte_carlo_results(df_full, mc_results, "LSTM")
    plot_future_feature_assumptions(future_features, "LSTM")
    plot_3d_impact_analysis(mc_results, "LSTM")
    
    print("\n7️⃣ Generating detailed report...")
    generate_detailed_report(df_full, mc_results, "LSTM", exog_strategy)
    
    print("\n8️⃣ Saving results...")
    output_dir = "results_lstm_improved"
    mc_results.to_csv(os.path.join(output_dir, 'monte_carlo_predictions_lstm.csv'))
    if future_features is not None:
        future_features.to_csv(os.path.join(output_dir, 'future_features_assumptions_lstm.csv'))
    
    print(f"\n✅ Analysis complete! All results saved in the '{output_dir}/' directory.")
    
    print("\n" + "=" * 80)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 80)
    target = df_full.columns[-1]
    historical_mean = df_full[target].mean()
    future_mean = mc_results['mean'].mean()
    change_percent = ((future_mean - historical_mean) / historical_mean * 100)
    
    print(f"🔹 Historical GHI Mean: {historical_mean:.2f} kWh/m²")
    print(f"🔹 Predicted 30-Year Mean: {future_mean:.2f} kWh/m²")
    print(f"🔹 Relative Change: {change_percent:+.1f}%")
    print(f"🔹 90% Confidence Interval: [{mc_results['p5'].mean():.2f}, {mc_results['p95'].mean():.2f}] kWh/m²")
    print(f"🔹 Pessimistic Scenario: {mc_results['p1'].mean():.2f} kWh/m²")
    print(f"🔹 Optimistic Scenario: {mc_results['p99'].mean():.2f} kWh/m²")

if __name__ == "__main__":
    try:
        main()
        print("\n🎉 Execution finished successfully!")
    except Exception as e:
        print(f"\n❌ A critical error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()