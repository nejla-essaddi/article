# -*- coding: utf-8 -*-
"""
Analyse Complète de Prédiction d'Irradiance Solaire avec LSTM, Informer, Simulation de Monte Carlo et Optimisation d'Hyperparamètres Avancée (Version Corrigée)

Ce script fusionne et améliore plusieurs approches pour la prédiction de GHI (Global Horizontal Irradiance).
"""

# --- 1. Importations et Configuration Initiale ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error # Added MAPE import
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math
import warnings
import os
import time

# Importations pour l'optimisation
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
import optuna

warnings.filterwarnings('ignore')

# --- Configuration Globale ---
CONFIG = {
    'features': ['ALLSKY_KT', 'T2M', 'RH2M'],
    'target': 'GHI',
    'seq_len': 12, 'label_len': 6, 'pred_len': 1,
    'split_ratios': [0.7, 0.15, 0.15],
    'random_state': 42,
    'lstm_params': {
        'lstm_units_1': 64, 'lstm_units_2': 32, 'num_layers': 2,
        'dense_layers': [16], 'dropout': 0.2, 'learning_rate': 0.001,
        'batch_size': 32, 'epochs': 100
    },
    'informer_params': {
        'embed_size': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1,
        'd_ff': 2048, 'dropout': 0.1, 'activation': 'gelu',
        'learning_rate': 0.0001, 'batch_size': 32, 'epochs': 100
    }
}

# --- Configuration de Styles et Semences ---
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(CONFIG['random_state'])
tf.random.set_seed(CONFIG['random_state'])
torch.manual_seed(CONFIG['random_state'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"✅ Configuration initiale chargée | Dispositif: {device}")
print(f"✅ Graines aléatoires fixées à: {CONFIG['random_state']}")
def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE) handling zero true values."""
    # Avoid division by zero: replace 0 in y_true with a small epsilon
    # or handle cases where y_true is 0.
    # For now, let's filter out true zero values for MAPE calculation as per common practice.
    # If y_true is exactly 0, the percentage error is undefined.
    non_zero_true_indices = y_true != 0
    if not np.any(non_zero_true_indices):
        return np.nan # Or a very large number, depending on how you want to represent "infinite error"

    y_true_filtered = y_true[non_zero_true_indices]
    y_pred_filtered = y_pred[non_zero_true_indices]

    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


# --- 2. Classe Principale d'Orchestration ---
class SolarForecastingSuite:
    def __init__(self, file_path):
        self.base_df = self._load_and_prepare_base_data(file_path)
        if self.base_df is None: raise ValueError("Le chargement des données de base a échoué.")
        self.results_summary = []
        self.output_base_dir = "results"
        os.makedirs(self.output_base_dir, exist_ok=True)

    def _load_and_prepare_base_data(self, file_path):
        try:
            df = pd.read_excel(file_path)
            date_col = next((col for col in ['datetime', 'date', 'YEAR', 'timestamp'] if col in df.columns), None)
            if not date_col: raise ValueError("Aucune colonne de date reconnue trouvée.")
            df['datetime'] = pd.to_datetime(df[date_col])
            df = df.set_index('datetime').sort_index()
            required_cols = CONFIG['features'] + [CONFIG['target']]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Colonnes manquantes: {[c for c in required_cols if c not in df.columns]}")
            print(f"✅ Données de base chargées. Plage de dates: {df.index.min()} à {df.index.max()}")
            return df
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données: {e}")
            return None

    def analyze_feature_correlation(self):
        print("\n" + "="*80 + "\n🔍 ANALYSE DE LA CORRÉLATION DES CARACTÉRISTIQUES\n" + "="*80)
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.base_df[CONFIG['features'] + [CONFIG['target']]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrice de Corrélation des Caractéristiques')
        correlation_path = os.path.join(self.output_base_dir, "feature_correlation_matrix.png")
        plt.savefig(correlation_path, bbox_inches='tight')
        plt.show()
        print(f"✅ Matrice de corrélation générée : {correlation_path}")

    def run_complete_analysis(self, model_type='lstm', timeframe='monthly'):
        print(f"\n" + "="*80 + f"\n🚀 DÉMARRAGE DE L'ANALYSE: Modèle={model_type.upper()}, Fréquence={timeframe.upper()}\n" + "="*80)
        resample_map = {'weekly': 'W', 'monthly': 'M'}
        if timeframe not in resample_map: raise ValueError("Le 'timeframe' doit être 'weekly' ou 'monthly'")

        processed_df = self.base_df.resample(resample_map[timeframe]).mean(numeric_only=True).dropna()
        if len(processed_df) < CONFIG['seq_len'] + CONFIG['pred_len']:
            print(f"❌ Données insuffisantes pour '{timeframe}'."); return

        output_dir = os.path.join(self.output_base_dir, f"{model_type.lower()}_{timeframe}")
        os.makedirs(output_dir, exist_ok=True)

        if model_type.lower() == 'lstm': predictor = LSTMPredictor(output_dir)
        elif model_type.lower() == 'informer': predictor = InformerPredictor(output_dir, timeframe)
        else: raise ValueError("Le 'model_type' doit être 'lstm' ou 'informer'")

        datasets = predictor.create_sequences(processed_df)
        if datasets is None: return

        print("\n🤖 Entraînement du modèle...")
        history, metrics, predictions = predictor.train_and_evaluate(datasets)
        if not metrics: print("L'entraînement a échoué, annulation de l'analyse."); return

        self.results_summary.append({'model': model_type.upper(), 'timeframe': timeframe.capitalize(), **metrics})
        predictor.plot_training_history(history)
        predictor.plot_predictions(predictions[1], predictions[0])
        predictor.monte_carlo_simulation(processed_df.copy(), predictions[1], predictions[0])

    def run_hyperparameter_tuning(self, model_type='lstm', timeframe='monthly', n_trials=50):
        print(f"\n" + "="*80 + f"\n🚀 DÉMARRAGE DE L'OPTIMISATION DES HYPERPARAMÈTRES\n   Modèle={model_type.upper()}, Fréquence={timeframe.upper()}\n" + "="*80)
        resample_map = {'weekly': 'W', 'monthly': 'M'}
        if timeframe not in resample_map: raise ValueError("Le 'timeframe' doit être 'weekly' ou 'monthly'")

        processed_df = self.base_df.resample(resample_map[timeframe]).mean(numeric_only=True).dropna()
        if len(processed_df) < CONFIG['seq_len'] + CONFIG['pred_len']:
            print(f"❌ Données insuffisantes pour '{timeframe}'."); return

        output_dir = os.path.join(self.output_base_dir, f"{model_type.lower()}_{timeframe}_tuning")
        os.makedirs(output_dir, exist_ok=True)

        if model_type.lower() == 'lstm':
            predictor = LSTMPredictor(output_dir)
            datasets = predictor.create_sequences(processed_df)
            if datasets: predictor.tune_hyperparameters(datasets)
        elif model_type.lower() == 'informer':
            predictor = InformerPredictor(output_dir, timeframe)
            datasets = predictor.create_sequences(processed_df)
            if datasets: predictor.tune_hyperparameters_optuna(datasets, n_trials=n_trials)
        else:
            print(f"L'optimisation pour le modèle '{model_type}' n'est pas implémentée.")
            return

        print("\n💡 IMPACT SUR LA PERFORMANCE : Vous pouvez maintenant mettre à jour votre CONFIG avec les meilleurs paramètres et ré-exécuter `run_complete_analysis`.")

    def generate_summary_report(self):
        if not self.results_summary: print("\nAucun résultat à résumer."); return
        print("\n" + "="*80 + "\n📈 RÉSUMÉ COMPARATIF DES PERFORMANCES\n" + "="*80)
        summary_df = pd.DataFrame(self.results_summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(self.output_base_dir, "summary_report.csv"), index=False)
        print(f"\n✅ Rapport de synthèse sauvegardé.")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=summary_df, x='timeframe', y='test_rmse', hue='model', ax=axes[0])
        axes[0].set_title('Comparaison du RMSE (Plus bas = Meilleur)'); axes[0].set_ylabel('RMSE (Wh/m²)')
        sns.barplot(data=summary_df, x='timeframe', y='test_r2', hue='model', ax=axes[1])
        axes[1].set_title('Comparaison du R² (Plus haut = Meilleur)'); axes[1].set_ylabel('Score R²')
        plt.suptitle('Comparaison des Modèles par Fréquence Temporelle', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_base_dir, "summary_comparison.png"))
        plt.show()

# --- Classes de Prédicteurs ---

class BasePredictor:
    def __init__(self, output_dir):
        self.scaler = StandardScaler()
        self.model = None
        self.config = CONFIG
        self.all_cols_ordered = self.config['features'] + [self.config['target']]
        self.output_dir = output_dir

    def plot_training_history(self, history):
        loss = history.history['loss'] if not isinstance(history, dict) else history['loss']
        val_loss = history.history['val_loss'] if not isinstance(history, dict) else history['val_loss']
        plt.figure(figsize=(10, 5)); plt.plot(loss, label='Perte d\'Entraînement'); plt.plot(val_loss, label='Perte de Validation')
        plt.title('Évolution de la Perte du Modèle'); plt.xlabel('Époque'); plt.ylabel('Perte (MSE)'); plt.legend()
        plt.savefig(os.path.join(self.output_dir, "training_history.png")); plt.show()

    def plot_predictions(self, y_true, y_pred):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6)); n_points = min(100, len(y_true))
        axes[0].plot(y_true[:n_points], label='Actual Values', color='blue', marker='.', linestyle='-')
        axes[0].plot(y_pred[:n_points], label='Predictions', color='red', marker='x', linestyle='--')
        axes[0].set_title(f'Time Comparison (First {n_points} Points)'); axes[0].legend()
        axes[1].scatter(y_true, y_pred, alpha=0.5, edgecolors='k'); axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Fit')
        axes[1].set_title('Actual vs. Predicted Correlation'); axes[1].grid(True); axes[1].legend()
        fig.suptitle(f"Prediction Analysis - {os.path.basename(self.output_dir)}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(self.output_dir, "predictions_analysis.png")); plt.show()

    def _get_metrics(self, y_true, y_pred):
        """Calculates and returns various regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred) # Call the new MAPE function

        return {
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2,
            'test_mape': mape # Add MAPE to the metrics dictionary
        }

    def _plot_monte_carlo_results(self, results, n_simulations):
        stats = {'mean': np.mean(results), 'median': np.median(results), 'std': np.std(results), 'ci_95': np.percentile(results, [2.5, 97.5])}
        print("\n--- Résultats de la Simulation de Monte Carlo ---"); [print(f"{k.capitalize()}: {v:.2f}") for k, v in stats.items() if not isinstance(v, np.ndarray)]; print(f"CI_95: [{stats['ci_95'][0]:.2f}, {stats['ci_95'][1]:.2f}]")
        fig, ax = plt.subplots(figsize=(12, 6)); sns.histplot(results, bins=100, kde=True, ax=ax, color='skyblue')
        ax.axvline(stats['mean'], color='red', linestyle='--', label=f"Moyenne: {stats['mean']:.2f}"); ax.axvline(stats['median'], color='green', linestyle='-', label=f"Médiane: {stats['median']:.2f}")
        ax.axvspan(stats['ci_95'][0], stats['ci_95'][1], color='red', alpha=0.1, label='Intervalle de Confiance 95%'); ax.legend()
        ax.set_title(f'Distribution des Prédictions de {self.config["target"]} ({n_simulations:,} simulations)')
        plt.savefig(os.path.join(self.output_dir, "monte_carlo_simulation.png")); plt.show()

class LSTMPredictor(BasePredictor):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    @staticmethod
    def _build_model(num_layers=2, lstm_units_1=64, lstm_units_2=32, dense_layers=[16], dropout=0.2, learning_rate=0.001, input_shape=None, pred_len=1):
        model = Sequential([Input(shape=input_shape)])
        for i in range(num_layers):
            model.add(LSTM(lstm_units_1 if i == 0 else lstm_units_2, return_sequences=(i < num_layers - 1)))
            model.add(Dropout(dropout))
        for units in dense_layers:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(pred_len))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        return model

    def create_sequences(self, df):
        data = df[self.all_cols_ordered].apply(pd.to_numeric, errors='coerce').values
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) < self.config['seq_len'] + self.config['pred_len']:
            return None
        data_scaled = self.scaler.fit_transform(data)
        target_idx, feature_indices = self.all_cols_ordered.index(self.config['target']), [self.all_cols_ordered.index(f) for f in self.config['features']]
        X_seq, y_seq = [], []
        for i in range(len(data_scaled) - self.config['seq_len'] - self.config['pred_len'] + 1):
            X_seq.append(data_scaled[i:i + self.config['seq_len'], feature_indices])
            y_seq.append(data_scaled[i + self.config['seq_len']:i + self.config['seq_len'] + self.config['pred_len'], target_idx])
        if not X_seq:
            return None
        X, y = np.array(X_seq), np.array(y_seq)
        train_size, val_size = int(len(X) * self.config['split_ratios'][0]), int(len(X) * self.config['split_ratios'][1])
        return (X[:train_size], y[:train_size]), (X[train_size:train_size + val_size], y[train_size:train_size + val_size]), (X[train_size + val_size:], y[train_size + val_size:])

    def train_and_evaluate(self, datasets):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
        params = self.config['lstm_params']
        model_build_params = {
            'num_layers': params.get('num_layers'),
            'lstm_units_1': params.get('lstm_units_1'),
            'lstm_units_2': params.get('lstm_units_2'),
            'dense_layers': params.get('dense_layers'),
            'dropout': params.get('dropout'),
            'learning_rate': params.get('learning_rate')
        }
        self.model = self._build_model(
            input_shape=X_train.shape[1:],
            pred_len=self.config['pred_len'],
            **model_build_params
        )
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[EarlyStopping(patience=20, restore_best_weights=True), ReduceLROnPlateau(patience=10)],
            verbose=1
        )
        pred_scaled = self.model.predict(X_test, verbose=0)
        target_idx = self.all_cols_ordered.index(self.config['target'])
        temp_pred = np.zeros((len(pred_scaled), len(self.all_cols_ordered)))
        temp_pred[:, target_idx] = pred_scaled.flatten()
        pred_denorm = self.scaler.inverse_transform(temp_pred)[:, target_idx]
        temp_true = np.zeros((len(y_test), len(self.all_cols_ordered)))
        temp_true[:, target_idx] = y_test.flatten()
        true_denorm = self.scaler.inverse_transform(temp_true)[:, target_idx]
        metrics = self._get_metrics(true_denorm, pred_denorm)
        print(f"\n--- Métriques LSTM (Test Set) --- \n {metrics}")
        return history, metrics, (pred_denorm, true_denorm)

    def tune_hyperparameters(self, datasets):
        print("\n" + "="*80 + "\nOPTIMISATION (GRID SEARCH) POUR LSTM\n" + "="*80)
        (X_train, y_train), _, _ = datasets
        param_grid = {
            'model__learning_rate': [0.001, 0.0005],
            'model__lstm_units_1': [32, 64],
            'model__dropout': [0.2, 0.3],
            'batch_size': [32, 64]
        }

        # Correction: Added 'validation_split=0.2' to ensure 'val_loss' is computed
        # for the EarlyStopping callback within each cross-validation fold.
        regressor = KerasRegressor(
            model=self._build_model,
            model__input_shape=X_train.shape[1:],
            model__pred_len=self.config['pred_len'],
            epochs=50,
            validation_split=0.2, # <-- THIS IS THE FIX
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )

        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("\n--- Meilleurs Hyperparamètres Trouvés ---")
        print(f"Meilleur score (RMSE) : {-grid_search.best_score_:.4f}")
        print("Meilleurs Paramètres:")
        # Print parameters in a cleaner format
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")
            
        return grid_search.best_params_

    def monte_carlo_simulation(self, df, y_true, y_pred, n_simulations=10000):
        residuals = y_true - y_pred
        last_prediction = y_pred[-1]
        simulations = last_prediction + np.random.choice(residuals, size=(n_simulations,), replace=True)
        self._plot_monte_carlo_results(np.maximum(0, simulations), n_simulations)

# --- Classes du modèle Informer ---

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.d_keys = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_keys ** -0.5
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        scores = torch.einsum("blhe,bshe->bhls", (queries, keys))
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(torch.softmax(scores * self.scale, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", (attn, values)).contiguous()
        V = V.view(B, L, -1)
        return self.out_projection(V)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.cross_attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm3(x+y)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = nn.Embedding(CONFIG['seq_len'] + CONFIG['pred_len'], d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        pos = torch.arange(0, x.size(1)).long().to(device)
        x = self.value_embedding(x) + self.position_embedding(pos)
        return self.dropout(x)

class InformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.1, activation='gelu'):
        super(InformerModel, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(d_layers)])
        self.projection = nn.Linear(d_model, c_out, bias=True)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)
        return self.projection(dec_out)

class Dataset_Informer(Dataset):
    def __init__(self, data, size):
        self.data_x, self.data_y = data
        self.seq_len, self.label_len, self.pred_len = size
    def __getitem__(self, i):
        s_end = i + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[i:s_end]
        seq_y_lbl = self.data_y[r_begin:r_end]
        seq_y_tkn = self.data_y[r_begin:r_begin + self.label_len]
        dec_inp = np.concatenate([seq_y_tkn, np.zeros((self.pred_len, seq_y_tkn.shape[1]))], axis=0)
        return torch.FloatTensor(seq_x), torch.FloatTensor(dec_inp), torch.FloatTensor(seq_y_lbl)
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class InformerPredictor(BasePredictor):
    def __init__(self, output_dir, timeframe):
        super().__init__(output_dir)
        self.timeframe = timeframe

    def create_sequences(self, df):
        df_copy=df.copy(); df_copy['date']=df_copy.index; df_copy['time_feat'] = df_copy['date'].dt.isocalendar().week if self.timeframe == 'weekly' else df_copy['date'].dt.month
        if df_copy['time_feat'].max()>0: df_copy['time_feat']/=df_copy['time_feat'].max()
        self.all_cols_ordered=(f:=self.config['features']+['time_feat'])+[self.config['target']]
        df_subset=df_copy[self.all_cols_ordered].apply(pd.to_numeric,errors='coerce').dropna();
        if len(df_subset)<self.config['seq_len']+self.config['pred_len']: return None
        data_s=self.scaler.fit_transform(df_subset.values); fi=[self.all_cols_ordered.index(c) for c in f]; ti=self.all_cols_ordered.index(self.config['target'])
        b1,b2=int(len(data_s)*self.config['split_ratios'][0]),int(len(data_s)*(self.config['split_ratios'][0]+self.config['split_ratios'][1]))
        data_x,data_y=data_s[:,fi],data_s[:,[ti]]; size=[self.config['seq_len'],self.config['label_len'],self.config['pred_len']]
        train_ds,val_ds,test_ds=Dataset_Informer((data_x[:b1],data_y[:b1]),size),Dataset_Informer((data_x[b1:b2],data_y[b1:b2]),size),Dataset_Informer((data_x[b2:],data_y[b2:]),size)
        bs=self.config['informer_params']['batch_size']
        return DataLoader(train_ds,bs,True),DataLoader(val_ds,bs,False),DataLoader(test_ds,1,False)

    def train_and_evaluate(self, datasets):
        train_loader, val_loader, test_loader = datasets
        params = self.config['informer_params']
        self.model = InformerModel(
            enc_in=len(self.config['features']) + 1, dec_in=1, c_out=1,
            d_model=params['embed_size'], n_heads=params['n_heads'], e_layers=params['e_layers'],
            d_layers=params['d_layers'], d_ff=params['d_ff'], dropout=params['dropout'],
            activation=params['activation']
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        for epoch in range(params['epochs']):
            self.model.train()
            epoch_loss = 0
            for x, y, y_lbl in train_loader:
                x, y, y_lbl = x.to(device), y.to(device), y_lbl.to(device)
                optimizer.zero_grad()
                out = self.model(x, None, y, None)
                loss = criterion(out[:, -self.config['pred_len']:], y_lbl[:, -self.config['pred_len']:])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y, y_lbl in val_loader:
                    x, y, y_lbl = x.to(device), y.to(device), y_lbl.to(device)
                    out = self.model(x, None, y, None)
                    val_loss += criterion(out[:, -self.config['pred_len']:], y_lbl[:, -self.config['pred_len']:]).item()
            val_losses.append(val_loss / len(val_loader))
            print(f"Epoch {epoch+1}/{params['epochs']} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y, y_lbl in test_loader:
                x, y, y_lbl = x.to(device), y.to(device), y_lbl.to(device)
                preds.append(self.model(x, None, y, None)[:, -self.config['pred_len']:].cpu())
                trues.append(y_lbl[:, -self.config['pred_len']:].cpu())

        preds, trues = np.concatenate(preds).flatten(), np.concatenate(trues).flatten()
        ti, nc = self.all_cols_ordered.index(self.config['target']), len(self.all_cols_ordered)
        pred_pad = np.zeros((len(preds), nc)); pred_pad[:, ti] = preds; pred_denorm = self.scaler.inverse_transform(pred_pad)[:, ti]
        true_pad = np.zeros((len(trues), nc)); true_pad[:, ti] = trues; true_denorm = self.scaler.inverse_transform(true_pad)[:, ti]
        metrics = self._get_metrics(true_denorm, pred_denorm)
        print(f"\n--- Métriques Informer (Test Set) --- \n {metrics}")
        return {'loss': train_losses, 'val_loss': val_losses}, metrics, (pred_denorm, true_denorm)

    def monte_carlo_simulation(self, df, y_true, y_pred, n_simulations=10000):
        residuals=y_true-y_pred; last_pred=y_pred[-1]; sims=last_pred+np.random.choice(residuals,size=(n_simulations,),replace=True)
        self._plot_monte_carlo_results(np.maximum(0,sims), n_simulations)

    def tune_hyperparameters_optuna(self, datasets, n_trials=50):
        print("\n" + "="*80 + "\nOPTIMISATION (OPTUNA) POUR INFORMER\n" + "="*80)
        train_loader, val_loader, _ = datasets
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'embed_size': trial.suggest_categorical('embed_size', [128, 256, 512]),
                'd_ff': trial.suggest_categorical('d_ff', [512, 1024, 2048]),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
                'e_layers': trial.suggest_int('e_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.05, 0.4),
            }
            
            model = InformerModel(
                enc_in=len(self.config['features'])+1, dec_in=1, c_out=1, 
                d_model=params['embed_size'], n_heads=params['n_heads'], e_layers=params['e_layers'], 
                d_layers=1, d_ff=params['d_ff'], dropout=params['dropout'], activation='gelu'
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.MSELoss()
            
            # --- OPTIMIZATION: More epochs and Early Stopping ---
            epochs = 75  # Increased from 30 to allow for proper convergence
            patience = 10 # Stop trial if no improvement for 10 epochs
            patience_counter = 0
            best_val_loss = float('inf')

            for epoch in range(epochs): 
                model.train()
                for x, y, y_lbl in train_loader:
                    x,y,y_lbl = x.to(device),y.to(device),y_lbl.to(device)
                    optimizer.zero_grad()
                    out=model(x,None,y,None)
                    loss=criterion(out[:,-self.config['pred_len']:],y_lbl[:,-self.config['pred_len']:])
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x,y,y_lbl in val_loader:
                        out = model(x.to(device),None,y.to(device),None)
                        val_loss += criterion(out[:,-self.config['pred_len']:],y_lbl.to(device)[:,-self.config['pred_len']:]).item()
                
                current_val_loss = val_loss / len(val_loader)

                # Update best loss and check for early stopping
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                trial.report(current_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                if patience_counter >= patience:
                    # print(f"Trial {trial.number} stopped early at epoch {epoch+1}.") # Optional: for debugging
                    break
                    
            return best_val_loss

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        # Added a timeout to prevent excessively long runs
        study.optimize(objective, n_trials=n_trials, timeout=7200)

        print("\n✅ Étude d'optimisation terminée.")
        print(f"--- Meilleur Essai (valeur={study.best_value:.6f}) ---")
        print("  Meilleurs hyperparamètres trouvés:")
        for k, v in study.best_params.items():
            print(f"    - {k}: {v}")
            
        return study.best_params


# --- 4. Point d'Entrée Principal ---
if __name__ == "__main__":
    # MODIFIEZ CETTE LIGNE pour pointer vers votre fichier de données Excel
    data_file_path = "dataset500.xlsx" 

    if not os.path.exists(data_file_path):
        print(f"❌ '{data_file_path}' introuvable. Assurez-vous que le fichier est dans le même répertoire que le script.")
    else:
        suite = SolarForecastingSuite(data_file_path)
        
        # --- Étape 1 : Analyse de corrélation ---
        suite.analyze_feature_correlation()

        # --- Étape 2 : Optimisation des hyperparamètres ---
        # Décommentez les lignes que vous souhaitez exécuter.

        
        print("\nLancement de l'optimisation pour Informer...")
        suite.run_hyperparameter_tuning(model_type='informer', timeframe='monthly', n_trials=20)


        # --- Étape 3 : Exécution de l'analyse comparative complète ---
        print("\n\n" + "="*80 + "\nEXÉCUTION DE L'ANALYSE COMPARATIVE AVEC LES PARAMÈTRES INITIAUX\n" + "="*80)

        #suite.run_complete_analysis(model_type='informer', timeframe='weekly')
        suite.run_complete_analysis(model_type='informer', timeframe='monthly')

        # print("\nLancement de l'optimisation pour LSTM...")
        suite.run_hyperparameter_tuning(model_type='lstm', timeframe='monthly')
        #suite.run_complete_analysis(model_type='lstm', timeframe='weekly')
        suite.run_complete_analysis(model_type='lstm', timeframe='monthly')

        # --- Étape 4 : Générer le rapport final comparatif ---
        suite.generate_summary_report()