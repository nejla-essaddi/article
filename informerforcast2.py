# informer_forecaster_improved.py

# -*- coding: utf-8 -*-
"""
Long-Term (30-Year) Prediction Script - Optimized Informer Model (Improved Version)

This script implements and runs the Optimized Informer model for long-term time-series forecasting,
including Monte Carlo simulations for worst/best-case scenarios and
visualizations of future feature assumptions and 3D impact.

## Citations
This script uses the following model:

1.  **Informer**: A model designed for long sequence time-series forecasting,
    which won the AAAI 2021 Best Paper award.
    - Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021).
      Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.
      In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 11106-11115).
      Retrieved from https://arxiv.org/abs/2012.07436

## Improvements:
1. Enhanced exogenous feature forecasting with multiple strategies
2. Cross-validation for robust residual calculation
3. Early stopping to prevent overfitting
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf # Keep for tqdm_callback if needed, though TF model is removed.
import warnings
import os
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# --- Global Configurations with Best Parameters ---
BEST_CONFIG_INFORMER = {
    'model_type': 'informer',
    'features': ['ALLSKY_KT', 'T2M', 'RH2M'],
    'target': 'GHI',
    'seq_len': 12, 'label_len': 6, 'pred_len': 1,
    'params': {
        'embed_size': 256, 'n_heads': 4, 'e_layers': 3, 'd_layers': 1,
        'd_ff': 512, 'dropout': 0.10366, 'activation': 'gelu',
        'learning_rate': 0.0003231, 'batch_size': 32, 'epochs': 100,
        'patience': 10, 'min_delta': 0.0001  # Added for early stopping
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

# --- 2. Model Classes (Informer) ---

# --- Informer Model Components ---
# Citation: Zhou et al. (2021). "Informer: Beyond Efficient Transformer..."
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, seq_len=12):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        pos = torch.arange(0, x.size(1)).long().to(device).clamp(0, self.position_embedding.num_embeddings - 1)
        x = self.value_embedding(x) + self.position_embedding(pos)
        return self.dropout(x)

class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FullAttention, self).__init__()
        self.d_keys = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_keys ** -0.5
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        scores = torch.einsum("blhe,bshe->bhls", (queries, keys))
        attn = self.dropout(torch.softmax(scores * self.scale, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", (attn, values)).contiguous().view(B, L, -1)
        return self.out_projection(V)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super(EncoderLayer, self).__init__()
        self.attention = FullAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super(DecoderLayer, self).__init__()
        self.self_attention = FullAttention(d_model, n_heads, dropout=dropout)
        self.cross_attention = FullAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm3(x+y)

class InformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, **kwargs):
        super(InformerModel, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, kwargs['embed_size'], kwargs['dropout'], seq_len)
        self.dec_embedding = DataEmbedding(dec_in, kwargs['embed_size'], kwargs['dropout'], seq_len)
        self.encoder = nn.ModuleList([EncoderLayer(kwargs['embed_size'], kwargs['n_heads'], kwargs['d_ff'], kwargs['dropout'], kwargs['activation']) for _ in range(kwargs['e_layers'])])
        self.decoder = nn.ModuleList([DecoderLayer(kwargs['embed_size'], kwargs['n_heads'], kwargs['d_ff'], kwargs['dropout'], kwargs['activation']) for _ in range(kwargs['d_layers'])])
        self.projection = nn.Linear(kwargs['embed_size'], c_out, bias=True)
    def forward(self, x_enc, x_dec):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.dec_embedding(x_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)
        return self.projection(dec_out)

class DatasetForInformer(Dataset):
    def __init__(self, data_x, data_y, seq_len, label_len, pred_len):
        self.data_x, self.data_y, self.seq_len, self.label_len, self.pred_len = data_x, data_y, seq_len, label_len, pred_len
        # Vérifier que nous avons assez de données
        self.length = max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
        
    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError("Index out of range")
            
        s_end = i + self.seq_len
        r_begin = s_end - self.label_len
        seq_x = self.data_x[i:s_end]
        seq_y_target = self.data_y[r_begin : r_begin + self.label_len + self.pred_len]
        decoder_input_token = self.data_y[r_begin : r_begin + self.label_len]
        decoder_placeholder = np.zeros((self.pred_len, self.data_y.shape[1]))
        decoder_input = np.concatenate([decoder_input_token, decoder_placeholder], axis=0)
        return torch.FloatTensor(seq_x), torch.FloatTensor(decoder_input), torch.FloatTensor(seq_y_target)
    
    def __len__(self): 
        return self.length


# --- 3. Enhanced Exogenous Feature Forecasting ---
def forecast_exogenous_features(df_full, feature_cols, n_steps=360, strategy='enhanced_monthly_avg'):

    print(f"🌍 Forecasting exogenous features using strategy: {strategy}")
    
    # Calculer les moyennes mensuelles historiques
    monthly_avg_features = df_full[feature_cols].groupby(df_full.index.month).mean()
    monthly_std_features = df_full[feature_cols].groupby(df_full.index.month).std()
    
    # Définir les dates futures
    future_dates = pd.date_range(start=df_full.index[-1], periods=n_steps + 1, freq='M')[1:]
    
    # Initialiser le DataFrame des caractéristiques futures
    future_features_df = pd.DataFrame(index=future_dates, columns=feature_cols)
    
    if strategy == 'monthly_avg':
        # Méthode originale: simple répétition des moyennes mensuelles
        for date in future_dates:
            future_features_df.loc[date] = monthly_avg_features.loc[date.month]
    
    elif strategy == 'linear_trend_monthly_avg':
        # Calculer les tendances linéaires pour chaque caractéristique
        trends = {}
        for col in feature_cols:
            years = df_full.index.year.values
            values = df_full[col].values
            slope, intercept, _, _, _ = stats.linregress(years, values)
            trends[col] = slope
        
        for date in future_dates:
            base_values = monthly_avg_features.loc[date.month]
            years_ahead = date.year - df_full.index[-1].year
            for col in feature_cols:
                trend_adjustment = trends[col] * years_ahead
                future_features_df.loc[date, col] = base_values[col] + trend_adjustment
    
    elif strategy == 'stochastic_monthly_avg':
        # Ajouter de la variabilité stochastique basée sur les écarts-types historiques
        for date in future_dates:
            base_values = monthly_avg_features.loc[date.month]
            std_values = monthly_std_features.loc[date.month]
            for col in feature_cols:
                # Ajouter du bruit gaussien basé sur l'écart-type historique
                noise = np.random.normal(0, std_values[col] * 0.5)  # 50% de la variabilité historique
                future_features_df.loc[date, col] = base_values[col] + noise
    
    elif strategy == 'enhanced_monthly_avg':
        # Combiner tendance linéaire et variabilité stochastique
        # Calculer les tendances linéaires
        trends = {}
        for col in feature_cols:
            years = df_full.index.year.values
            values = df_full[col].values
            slope, intercept, _, _, _ = stats.linregress(years, values)
            trends[col] = slope
        
        for date in future_dates:
            base_values = monthly_avg_features.loc[date.month]
            std_values = monthly_std_features.loc[date.month]
            years_ahead = date.year - df_full.index[-1].year
            
            for col in feature_cols:
                # Appliquer la tendance linéaire
                trend_adjustment = trends[col] * years_ahead
                # Ajouter de la variabilité stochastique
                noise = np.random.normal(0, std_values[col] * 0.3)  # 30% de la variabilité historique
                future_features_df.loc[date, col] = base_values[col] + trend_adjustment + noise
    
    return future_features_df

# --- Prediction Orchestration Class ---
class LongTermForecaster:
    def __init__(self, file_path, model_config):
        self.file_path = file_path
        self.config = model_config
        self.model_type = self.config['model_type']
        self.scaler = StandardScaler()
        self.model = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Clearly define column lists to avoid dimension errors
        self.feature_cols = list(self.config['features'])
        self.feature_cols_model = self.feature_cols + ['time_feat'] # Informer specific
        self.full_cols = self.feature_cols_model + [self.config['target']]

    def _load_and_prepare_data(self):
        try:
            df = pd.read_excel(self.file_path, parse_dates=['YEAR']).rename(columns={'YEAR': 'datetime'}).set_index('datetime').sort_index()
        except Exception as e:
            print(f"Error reading the Excel file: {e}")
            return None
        df_monthly = df.resample('M').mean(numeric_only=True)
        if self.model_type == 'informer':
            df_monthly['time_feat'] = (df_monthly.index.month - 1) / 11.0
        if not all(col in df_monthly.columns for col in self.config['features'] + [self.config['target']]):
            print(f"Error: Missing columns.")
            return None
        print(f"[{self.model_type.upper()}] ✅ Data loaded. Range: {df_monthly.index.min()} to {df_monthly.index.max()}")
        return df_monthly[self.full_cols].dropna()

    def _early_stopping_check(self, val_loss):
        """
        📈 AMÉLIORATION 3: Détection Précoce pour Éviter le Surapprentissage
        
        Cette méthode implémente l'early stopping pour éviter le surapprentissage.
        Elle surveille la perte de validation et arrête l'entraînement si aucune amélioration
        n'est observée pendant un certain nombre d'époques (patience).
        """
        if val_loss < self.best_val_loss - self.config['params'].get('min_delta', 0.0001):
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False  # Continue training
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['params'].get('patience', 10):
                return True  # Stop training
        return False

    def train(self, df_train):
        print(f"[{self.model_type.upper()}] 🤖 Starting training with early stopping...")
        
        # Vérifier que nous avons assez de données
        min_required_length = self.config['seq_len'] + self.config['pred_len'] + 10  # +10 pour une marge
        if len(df_train) < min_required_length:
            print(f"[{self.model_type.upper()}] ❌ Insufficient data: {len(df_train)} < {min_required_length}")
            return
        
        # Split data for training and validation
        train_size = int(len(df_train) * 0.8)
        # S'assurer que le set de validation a au moins la taille minimale requise
        val_size = len(df_train) - train_size
        if val_size < min_required_length:
            # Ajuster le découpage si nécessaire
            train_size = len(df_train) - min_required_length
            if train_size < min_required_length:
                print(f"[{self.model_type.upper()}] ❌ Dataset too small for training and validation")
                return
        
        df_train_split = df_train.iloc[:train_size]
        df_val_split = df_train.iloc[train_size:]
        
        # Scale data
        data_scaled = self.scaler.fit_transform(df_train_split)
        val_data_scaled = self.scaler.transform(df_val_split)
        
        params = self.config['params']
        feature_indices = [self.full_cols.index(c) for c in self.feature_cols_model]
        target_idx = self.full_cols.index(self.config['target'])
        
        # Prepare training data
        data_x, data_y = data_scaled[:, feature_indices], data_scaled[:, [target_idx]]
        train_dataset = DatasetForInformer(data_x, data_y, self.config['seq_len'], self.config['label_len'], self.config['pred_len'])
        
        # Vérifier que le dataset d'entraînement n'est pas vide
        if len(train_dataset) == 0:
            print(f"[{self.model_type.upper()}] ❌ Training dataset is empty after processing")
            return
        
        train_loader = DataLoader(train_dataset, batch_size=min(params['batch_size'], len(train_dataset)), shuffle=True)
        
        # Prepare validation data
        val_x, val_y = val_data_scaled[:, feature_indices], val_data_scaled[:, [target_idx]]
        val_dataset = DatasetForInformer(val_x, val_y, self.config['seq_len'], self.config['label_len'], self.config['pred_len'])
        
        # Vérifier que le dataset de validation n'est pas vide
        if len(val_dataset) == 0:
            print(f"[{self.model_type.upper()}] ❌ Validation dataset is empty after processing")
            return
        
        val_loader = DataLoader(val_dataset, batch_size=min(params['batch_size'], len(val_dataset)), shuffle=False)
        
        # Initialize model
        self.model = InformerModel(enc_in=len(self.feature_cols_model), dec_in=1, c_out=1, seq_len=self.config['seq_len'], **params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        for epoch in tqdm(range(params['epochs']), desc=f"Training {self.model_type.upper()}"):
            # Training phase
            self.model.train()
            train_loss = 0
            for x_enc, x_dec, y_true in train_loader:
                optimizer.zero_grad()
                output = self.model(x_enc.to(device), x_dec.to(device))
                loss = criterion(output[:, -self.config['pred_len']:], y_true.to(device)[:, -self.config['pred_len']:])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_enc, x_dec, y_true in val_loader:
                    output = self.model(x_enc.to(device), x_dec.to(device))
                    loss = criterion(output[:, -self.config['pred_len']:], y_true.to(device)[:, -self.config['pred_len']:])
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping check
            if self._early_stopping_check(val_loss):
                print(f"[{self.model_type.upper()}] ⏰ Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"[{self.model_type.upper()}] ✅ Training finished.")



    def predict_future(self, df_full, n_steps=360, exog_strategy='enhanced_monthly_avg'):
        """
        🔮 Génération de prédictions futures avec stratégies améliorées pour les caractéristiques exogènes
        """
        print(f"[{self.model_type.upper()}] 🔮 Generating predictions for {n_steps} months...")
        
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            print(f"[{self.model_type.upper()}] ❌ Model not trained yet")
            return None
        
        # Forecast future exogenous features using the enhanced strategy
        future_features_df = forecast_exogenous_features(df_full, self.feature_cols_model, n_steps, exog_strategy)
        
        if future_features_df is None:
            print(f"[{self.model_type.upper()}] ❌ Failed to generate future features")
            return None
        
        # Scale the full historical data
        data_scaled = self.scaler.transform(df_full)
        current_sequence = list(data_scaled[-self.config['seq_len']:])
        future_preds_scaled = []
        
        future_dates = pd.date_range(start=df_full.index[-1], periods=n_steps + 1, freq='M')[1:]
        feature_indices = [self.full_cols.index(c) for c in self.feature_cols_model]
        target_idx = self.full_cols.index(self.config['target'])

        for i in tqdm(range(n_steps), desc=f"Predicting {self.model_type.upper()}"):
            try:
                input_seq_scaled = np.array(current_sequence)
                x_enc = torch.FloatTensor(input_seq_scaled[:, feature_indices]).unsqueeze(0).to(device)
                dec_token = input_seq_scaled[-self.config['label_len']:, [target_idx]]
                dec_input = torch.FloatTensor(np.concatenate([dec_token, np.zeros((self.config['pred_len'], 1))])).unsqueeze(0).to(device)

                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(x_enc, dec_input)
                next_pred_scaled = prediction[0, -1, 0].item()
                future_preds_scaled.append(next_pred_scaled)

                # Use the forecasted exogenous features for the next step
                current_date = future_dates[i]
                next_step_values = {}
                
                # Get the forecasted exogenous features for this date
                for col_name in self.feature_cols_model:
                    if col_name in future_features_df.columns:
                        next_step_values[col_name] = future_features_df.loc[current_date, col_name]
                    else:
                        # Si la colonne n'existe pas, utiliser une valeur par défaut
                        next_step_values[col_name] = 0.0
                
                next_step_values[self.config['target']] = next_pred_scaled

                # Create a single-row DataFrame with columns in the correct order
                next_step_df = pd.DataFrame([next_step_values], columns=self.full_cols)
                next_step_scaled = self.scaler.transform(next_step_df)

                current_sequence.pop(0)
                current_sequence.append(next_step_scaled[0])
                
            except Exception as e:
                print(f"[{self.model_type.upper()}] ❌ Error at step {i}: {str(e)}")
                break

        if len(future_preds_scaled) == 0:
            print(f"[{self.model_type.upper()}] ❌ No predictions generated")
            return None
        
        # Denormalize predictions
        dummy_array = np.zeros((len(future_preds_scaled), len(self.full_cols)))
        dummy_array[:, target_idx] = future_preds_scaled
        future_preds_denorm = self.scaler.inverse_transform(dummy_array)[:, target_idx]
        
        # Créer les dates correspondantes
        pred_dates = future_dates[:len(future_preds_scaled)]
        return pd.Series(future_preds_denorm, index=pred_dates)
        """
        🌍 AMÉLIORATION 1: Représentation Améliorée des Caractéristiques Exogènes Futures
        """
        print(f"🌍 Forecasting exogenous features using strategy: {strategy}")
        
        # Filtrer les colonnes qui existent réellement dans df_full
        available_feature_cols = [col for col in feature_cols if col in df_full.columns]
        
        if not available_feature_cols:
            print("❌ No valid feature columns found in the dataset")
            return None
        
        # Calculer les moyennes mensuelles historiques
        monthly_avg_features = df_full[available_feature_cols].groupby(df_full.index.month).mean()
        monthly_std_features = df_full[available_feature_cols].groupby(df_full.index.month).std()
        
        # Définir les dates futures
        future_dates = pd.date_range(start=df_full.index[-1], periods=n_steps + 1, freq='M')[1:]
        
        # Initialiser le DataFrame des caractéristiques futures
        future_features_df = pd.DataFrame(index=future_dates, columns=available_feature_cols)
        
        if strategy == 'monthly_avg':
            # Méthode originale: simple répétition des moyennes mensuelles
            for date in future_dates:
                future_features_df.loc[date] = monthly_avg_features.loc[date.month]
        
        elif strategy == 'linear_trend_monthly_avg':
            # Calculer les tendances linéaires pour chaque caractéristique
            trends = {}
            for col in available_feature_cols:
                years = df_full.index.year.values
                values = df_full[col].values
                # Supprimer les valeurs NaN
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) > 1:
                    slope, intercept, _, _, _ = stats.linregress(years[valid_mask], values[valid_mask])
                    trends[col] = slope
                else:
                    trends[col] = 0
            
            for date in future_dates:
                base_values = monthly_avg_features.loc[date.month]
                years_ahead = date.year - df_full.index[-1].year
                for col in available_feature_cols:
                    trend_adjustment = trends[col] * years_ahead
                    future_features_df.loc[date, col] = base_values[col] + trend_adjustment
        
        elif strategy == 'stochastic_monthly_avg':
            # Ajouter de la variabilité stochastique basée sur les écarts-types historiques
            for date in future_dates:
                base_values = monthly_avg_features.loc[date.month]
                std_values = monthly_std_features.loc[date.month]
                for col in available_feature_cols:
                    # Éviter les NaN dans les écarts-types
                    std_val = std_values[col] if not np.isnan(std_values[col]) else 0
                    noise = np.random.normal(0, std_val * 0.5)
                    future_features_df.loc[date, col] = base_values[col] + noise
        
        elif strategy == 'enhanced_monthly_avg':
            # Combiner tendance linéaire et variabilité stochastique
            # Calculer les tendances linéaires
            trends = {}
            for col in available_feature_cols:
                years = df_full.index.year.values
                values = df_full[col].values
                # Supprimer les valeurs NaN
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) > 1:
                    slope, intercept, _, _, _ = stats.linregress(years[valid_mask], values[valid_mask])
                    trends[col] = slope
                else:
                    trends[col] = 0
            
            for date in future_dates:
                base_values = monthly_avg_features.loc[date.month]
                std_values = monthly_std_features.loc[date.month]
                years_ahead = date.year - df_full.index[-1].year
                
                for col in available_feature_cols:
                    # Appliquer la tendance linéaire
                    trend_adjustment = trends[col] * years_ahead
                    # Ajouter de la variabilité stochastique
                    std_val = std_values[col] if not np.isnan(std_values[col]) else 0
                    noise = np.random.normal(0, std_val * 0.3)
                    future_features_df.loc[date, col] = base_values[col] + trend_adjustment + noise
        
        return future_features_df

    def get_residuals_cross_validation(self, df, n_splits=5):
        """
        🎲 AMÉLIORATION 2: Validation Croisée pour le Calcul Robuste des Résidus
        """
        print(f"[{self.model_type.upper()}] 🧮 Calculating residuals using cross-validation...")
        
        # Vérifier que nous avons assez de données pour la validation croisée
        min_required_length = self.config['seq_len'] + self.config['pred_len'] + 10
        if len(df) < min_required_length * n_splits:
            print(f"[{self.model_type.upper()}] ⚠️ Not enough data for {n_splits}-fold CV, using simple split")
            return self.get_residuals(df)
        
        # Utiliser TimeSeriesSplit pour respecter l'ordre temporel
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_residuals = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            print(f"[{self.model_type.upper()}] 📊 Processing fold {fold + 1}/{n_splits}")
            
            # Diviser les données
            df_train_fold = df.iloc[train_idx]
            df_test_fold = df.iloc[test_idx]
            
            # Vérifier que nous avons assez de données pour ce fold
            if len(df_train_fold) < min_required_length or len(df_test_fold) < 1:
                print(f"[{self.model_type.upper()}] ⚠️ Skipping fold {fold + 1} due to insufficient data")
                continue
            
            # Créer un nouveau forecaster pour ce fold
            fold_forecaster = LongTermForecaster(self.file_path, self.config)
            
            # Entraîner le modèle pour ce fold
            fold_forecaster.train(df_train_fold.copy())
            
            # Vérifier que le modèle a été entraîné correctement
            if fold_forecaster.model is None:
                print(f"[{self.model_type.upper()}] ⚠️ Skipping fold {fold + 1} due to training failure")
                continue
            
            # Prédire sur l'ensemble de test
            try:
                preds = fold_forecaster.predict_future(df_train_fold.copy(), 
                                                     n_steps=len(df_test_fold), 
                                                     exog_strategy='enhanced_monthly_avg')
                
                # Calculer les résidus pour ce fold
                fold_residuals = df_test_fold[self.config['target']].values - preds.values
                all_residuals.extend(fold_residuals)
            except Exception as e:
                print(f"[{self.model_type.upper()}] ⚠️ Error in fold {fold + 1}: {str(e)}")
                continue
        
        # Si nous n'avons pas assez de résidus, utiliser la méthode simple
        if len(all_residuals) < 10:
            print(f"[{self.model_type.upper()}] ⚠️ Not enough residuals from CV, using simple split")
            return self.get_residuals(df)
        
        all_residuals = np.array(all_residuals)
        print(f"[{self.model_type.upper()}] ✅ Residuals calculated using cross-validation on {len(all_residuals)} points.")
        return all_residuals

    def run_monte_carlo(self, base_predictions, residuals):
        print(f"[{self.model_type.upper()}] 🇲🇨 Running Monte Carlo for worst/best-case scenarios...")
        n_sims = MONTE_CARLO_SIMULATIONS
        # Add residuals to simulate uncertainty on the target prediction
        sim_matrix = base_predictions.values[:, np.newaxis] + np.random.choice(residuals, (len(base_predictions), n_sims), replace=True)
        sim_matrix[sim_matrix < 0] = 0 # Ensure values are not negative

        # Calculate percentiles for best (99th) and worst (1st) cases
        # and keep the mean and 90% confidence interval
        return pd.DataFrame({
            'mean': np.mean(sim_matrix, axis=1),
            'p1': np.percentile(sim_matrix, 1, axis=1),  # Worst case (1st percentile)
            'p5': np.percentile(sim_matrix, 5, axis=1),  # Lower bound 90% CI
            'p95': np.percentile(sim_matrix, 95, axis=1), # Upper bound 90% CI
            'p99': np.percentile(sim_matrix, 99, axis=1) # Best case (99th percentile)
        }, index=base_predictions.index)

    def get_residuals(self, df):
        """
        Méthode originale maintenue pour la compatibilité
        """
        print(f"[{self.model_type.upper()}] 🧮 Calculating historical residuals...")
        train_size = int(len(df) * 0.8)
        df_train_res, df_test_res = df.iloc[:train_size], df.iloc[train_size:]
        residual_forecaster = LongTermForecaster(self.file_path, self.config)
        residual_forecaster.train(df_train_res.copy())
        preds = residual_forecaster.predict_future(df_train_res.copy(), n_steps=len(df_test_res))
        residuals = df_test_res[self.config['target']].values - preds.values
        print(f"[{self.model_type.upper()}] ✅ Residuals calculated on {len(residuals)} points.")
        return residuals

# --- Utility and Visualization Functions ---
class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, desc):
        self.pbar = tqdm(total=total_epochs, desc=desc)
    def on_epoch_end(self, epoch, logs=None): self.pbar.update(1)
    def on_train_end(self, logs=None): self.pbar.close()

def tqdm_callback(epochs, desc): return TQDMProgressBar(total_epochs=epochs, desc=desc)

def plot_monte_carlo_results(historical_data, mc_results, model_name):
    print(f"\n📊 Generating Monte Carlo graphs for {model_name}...")
    output_dir = "results_informer_only"
    os.makedirs(output_dir, exist_ok=True)
    target = historical_data.columns[-1]

    plt.figure(figsize=(15, 7))
    plt.plot(historical_data.index, historical_data[target], label='Historical Data', color='gray', alpha=0.6)
    plt.plot(mc_results.index, mc_results['mean'], label=f'Mean Prediction ({model_name})', color='blue', linestyle='--')

    # 90% Confidence Interval
    plt.fill_between(mc_results.index, mc_results['p5'], mc_results['p95'], color='blue', alpha=0.3, label='90% Confidence Interval')

    # Worst Case (1st percentile) and Best Case (99th percentile)
    plt.fill_between(mc_results.index, mc_results['p1'], mc_results['p5'], color='red', alpha=0.15, label='Worst Case (1st percentile)')
    plt.fill_between(mc_results.index, mc_results['p95'], mc_results['p99'], color='green', alpha=0.15, label='Best Case (99th percentile)')

    plt.title(f'Monte Carlo Simulation for {model_name} over 30 Years (Extreme Scenarios)')
    plt.xlabel('Year')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(output_dir, f"figure_2_monte_carlo_{model_name}_extreme_scenarios.png"))
    plt.savefig(f'{output_dir}/figure_2_monte_carlo{model_name.lower()}.png', dpi=300, bbox_inches='tight') 
    plt.show()
def plot_future_feature_assumptions(future_features, model_name):
    """
    📊 Visualisation des hypothèses futures pour les caractéristiques exogènes
    """
    print(f"\n🌍 Generating future feature assumptions plot for {model_name}...")
    output_dir = "results_informer_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    # Vérifier que nous avons des données à visualiser
    if future_features is None or future_features.empty:
        print("❌ No future features data to plot")
        return
    
    # Déterminer le nombre de sous-graphiques nécessaires
    n_features = len(future_features.columns)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # S'assurer que axes est toujours un array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Aplatir pour faciliter l'itération
    axes_flat = axes.flatten()
    
    for i, column in enumerate(future_features.columns):
        if i < len(axes_flat):
            ax = axes_flat[i]
            ax.plot(future_features.index, future_features[column], color='orange', alpha=0.7)
            ax.set_title(f'Future {column} Assumptions')
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.grid(True, alpha=0.3)
    
    # Masquer les axes inutilisés
    for i in range(len(future_features.columns), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/future_feature_assumptions_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
def plot_future_feature_assumptions(future_features, model_name):
    """
    📊 Visualisation des hypothèses futures pour les caractéristiques exogènes
    """
    print(f"\n🌍 Generating future feature assumptions plot for {model_name}...")
    output_dir = "results_informer_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    feature_names = ['ALLSKY_KT', 'T2M', 'RH2M', 'time_feat']
    feature_labels = ['Solar Clearness Index', 'Temperature (°C)', 'Humidity (%)', 'Time Feature']
    
    for i, (feature, label) in enumerate(zip(feature_names, feature_labels)):
        if feature in future_features.columns:
            axes[i].plot(future_features.index, future_features[feature], color='orange', alpha=0.7)
            axes[i].set_title(f'Future {label} Assumptions')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(label)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/future_feature_assumptions_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_impact_analysis(historical_data, mc_results, model_name):
    """
    🎯 Analyse d'impact 3D montrant l'évolution temporelle des prédictions
    """
    print(f"\n🎯 Generating 3D impact analysis for {model_name}...")
    output_dir = "results_informer_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer les données 3D
    years = mc_results.index.year
    months = mc_results.index.month
    
    # Créer une grille pour l'affichage 3D
    unique_years = sorted(set(years))
    unique_months = list(range(1, 13))
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Préparer les données pour le plotting 3D
    X, Y = np.meshgrid(unique_months, unique_years)
    Z = np.zeros_like(X, dtype=float)
    
    # Remplir Z avec les valeurs moyennes
    for i, year in enumerate(unique_years):
        for j, month in enumerate(unique_months):
            mask = (years == year) & (months == month)
            if mask.any():
                Z[i, j] = mc_results.loc[mask, 'mean'].iloc[0]
    
    # Créer la surface 3D
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    # Ajouter les lignes de contour
    ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_zlabel('GHI (kWh/m²)')
    ax.set_title(f'3D Impact Analysis - {model_name} Predictions')
    
    # Ajouter une barre de couleur
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig(f'{output_dir}/3d_impact_analysis_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(historical_data, mc_results, model_name, exog_strategy):
    """
    📄 Generates a detailed report of the results.
    """
    print(f"\n📄 Generating detailed report for {model_name}...")
    output_dir = "results_informer_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    target = historical_data.columns[-1]
    
    # Calculate detailed statistics
    historical_mean = historical_data[target].mean()
    historical_std = historical_data[target].std()
    
    future_mean = mc_results['mean'].mean()
    future_std = mc_results['mean'].std()
    
    # 30-year trend
    years = mc_results.index.year
    values = mc_results['mean'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    
    # Calculate seasonal variability
    monthly_means = mc_results.groupby(mc_results.index.month)['mean'].mean()
    seasonal_variability = monthly_means.std()
    
    # Generate the report
    report = f"""
# Detailed Report: Long-Term GHI Forecast ({model_name})

## Model Configuration
- **Model**: {model_name}
- **Exogenous Features Strategy**: {exog_strategy}
- **Forecast Period**: 30 years ({mc_results.index.min().strftime('%Y-%m')} to {mc_results.index.max().strftime('%Y-%m')})
- **Monte Carlo Simulations**: {MONTE_CARLO_SIMULATIONS:,}

## Historical Statistics
- **Historical Mean**: {historical_mean:.2f} kWh/m²
- **Historical Standard Deviation**: {historical_std:.2f} kWh/m²
- **Historical Period**: {historical_data.index.min().strftime('%Y-%m')} to {historical_data.index.max().strftime('%Y-%m')}

## Forecast Results
- **Future Mean**: {future_mean:.2f} kWh/m²
- **Future Standard Deviation**: {future_std:.2f} kWh/m²
- **Relative Change**: {((future_mean - historical_mean) / historical_mean * 100):+.1f}%

## Trend Analysis
- **Trend Slope**: {slope:.4f} kWh/m²/year
- **Correlation Coefficient**: {r_value:.3f}
- **P-value**: {p_value:.6f}
- **Standard Error**: {std_err:.4f}

## Confidence Intervals
- **90% Confidence Interval**: [{mc_results['p5'].mean():.2f}, {mc_results['p95'].mean():.2f}] kWh/m²
- **Pessimistic Scenario (1st percentile)**: {mc_results['p1'].mean():.2f} kWh/m²
- **Optimistic Scenario (99th percentile)**: {mc_results['p99'].mean():.2f} kWh/m²

## Seasonal Variability
- **Seasonal Standard Deviation**: {seasonal_variability:.2f} kWh/m²
- **Highest Month**: {monthly_means.idxmax()} ({monthly_means.max():.2f} kWh/m²)
- **Lowest Month**: {monthly_means.idxmin()} ({monthly_means.min():.2f} kWh/m²)

## Predicted Monthly Averages
"""
    
    for month in range(1, 13):
        month_name = pd.Timestamp(2024, month, 1).strftime('%B')
        month_mean = monthly_means.get(month, float('nan')) # Use .get for safety
        report += f"- **{month_name}**: {month_mean:.2f} kWh/m²\n"
    
    # Save the report
    report_path = f'{output_dir}/detailed_report_{model_name.lower()}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to {report_path}")

def main():
    """
    🚀 MAIN FUNCTION - Complete Orchestration with Enhancements
    """
    print("🚀 Starting long-term GHI forecast with the improved Informer model")
    print("=" * 80)
    
    # Configuration
    file_path = "dataset500.xlsx"
    config = BEST_CONFIG_INFORMER
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: The file {file_path} does not exist.")
        print("Please ensure the data file is in the current directory.")
        # Create a dummy file to allow the script to run for demonstration
        print("Creating a dummy 'dataset500.xlsx' for demonstration purposes.")
        pd.DataFrame().to_excel(file_path)


    # Initialize the forecaster
    forecaster = LongTermForecaster(file_path, config)
    
    # Load and prepare data
    print("\n1️⃣ Loading and preparing data...")
    df_full = forecaster._load_and_prepare_data()
    if df_full is None:
        print("❌ Error loading data.")
        return
    
    # Train the model with early stopping
    print("\n2️⃣ Training the model with early stopping...")
    forecaster.train(df_full.copy())
    
    # Calculate residuals with cross-validation
    print("\n3️⃣ Calculating residuals with cross-validation...")
    residuals = forecaster.get_residuals_cross_validation(df_full.copy())
    
    # Generate future predictions with an improved strategy
    print("\n4️⃣ Generating future predictions...")
    exog_strategy = 'enhanced_monthly_avg'
    future_predictions = forecaster.predict_future(df_full.copy(), n_steps=360, exog_strategy=exog_strategy)
    
    # Prepare future features for visualization
    future_features = forecast_exogenous_features(df_full, forecaster.feature_cols, n_steps=360, strategy=exog_strategy)
    
    # Monte Carlo Simulation
    print("\n5️⃣ Running Monte Carlo simulation for uncertainty analysis...")
    mc_results = forecaster.run_monte_carlo(future_predictions, residuals)
    
    # Visualizations
    print("\n6️⃣ Generating visualizations...")
    plot_monte_carlo_results(df_full, mc_results, "Informer")
    plot_future_feature_assumptions(future_features, "Informer")
    plot_3d_impact_analysis(df_full, mc_results, "Informer")
    
    # Generate the detailed report
    print("\n7️⃣ Generating the detailed report...")
    generate_detailed_report(df_full, mc_results, "Informer", exog_strategy)
    
    # Save the results
    print("\n8️⃣ Saving results...")
    output_dir = "results_informer_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    mc_results.to_csv(f'{output_dir}/monte_carlo_predictions_informer.csv')
    future_features.to_csv(f'{output_dir}/future_features_assumptions_informer.csv')
    
    print("\n✅ Analysis completed successfully!")
    print(f"📁 All results have been saved in the folder: {output_dir}/")
    print("\nGenerated files:")
    print("- monte_carlo_forecast_informer.png")
    print("- future_feature_assumptions_informer.png")
    print("- 3d_impact_analysis_informer.png")
    print("- detailed_report_informer.md")
    print("- monte_carlo_predictions_informer.csv")
    print("- future_features_assumptions_informer.csv")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 80)
    target = df_full.columns[-1]
    historical_mean = df_full[target].mean()
    future_mean = mc_results['mean'].mean()
    change_percent = ((future_mean - historical_mean) / historical_mean * 100)
    
    print(f"🔹 Historical GHI Mean: {historical_mean:.2f} kWh/m²")
    print(f"🔹 Predicted Mean (30 years): {future_mean:.2f} kWh/m²")
    print(f"🔹 Relative Change: {change_percent:+.1f}%")
    print(f"🔹 90% Confidence Interval: [{mc_results['p5'].mean():.2f}, {mc_results['p95'].mean():.2f}] kWh/m²")
    print(f"🔹 Pessimistic Scenario: {mc_results['p1'].mean():.2f} kWh/m²")
    print(f"🔹 Optimistic Scenario: {mc_results['p99'].mean():.2f} kWh/m²")
    
    return mc_results, future_features

if __name__ == "__main__":
    try:
        results, features = main()
        print("\n🎉 Execution finished successfully!")
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()