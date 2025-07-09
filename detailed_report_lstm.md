
# Detailed Report: Long-Term GHI Forecast (LSTM)

## Model Configuration
- **Model**: LSTM
- **Exogenous Feature Strategy**: enhanced_monthly_avg
- **Forecast Period**: 30 years (2024-01 to 2053-12)
- **Monte Carlo Simulations**: 10,000

## Historical Statistics
- **Historical Mean**: 5456.26 kWh/m²
- **Historical Std Dev**: 1731.08 kWh/m²
- **Historical Period**: 2004-01 to 2023-12

## Forecast Results
- **Future Mean**: 5292.01 kWh/m²
- **Future Std Dev**: 1760.76 kWh/m²
- **Relative Change**: -3.0%

## Trend Analysis
- **Trend Slope**: -16.5025 kWh/m²/year
- **Correlation (R-value)**: -0.081
- **P-value**: 0.123925
- **Standard Error**: 10.7011

## Confidence Intervals
- **90% Confidence Interval**: [4766.52, 5723.69] kWh/m²
- **Worst Case (1st percentile)**: 4493.79 kWh/m²
- **Best Case (99th percentile)**: 5873.46 kWh/m²

## Seasonal Variability
- **Seasonal Std Dev**: 1829.14 kWh/m²
- **Peak Month**: 7 (7646.29 kWh/m²)
- **Lowest Month**: 12 (2747.75 kWh/m²)

## Predicted Monthly Averages
- **January**: 3082.72 kWh/m²
- **February**: 4036.69 kWh/m²
- **March**: 5228.86 kWh/m²
- **April**: 6280.66 kWh/m²
- **May**: 7085.92 kWh/m²
- **June**: 7597.33 kWh/m²
- **July**: 7646.29 kWh/m²
- **August**: 6953.07 kWh/m²
- **September**: 5569.39 kWh/m²
- **October**: 4141.29 kWh/m²
- **November**: 3134.12 kWh/m²
- **December**: 2747.75 kWh/m²
