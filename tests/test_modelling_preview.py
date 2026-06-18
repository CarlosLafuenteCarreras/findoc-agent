# Script temporal de verificación — NO es parte del proyecto final,
# solo para comprobar que data_loader.py y preprocessing.py funcionan
# correctamente sobre el dataset real antes de avanzar al trainer.

import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO)

from src.modelling.data_loader import load_loan_dataset, get_dataset_summary
from src.modelling.preprocessing import prepare_features, split_train_test, scale_features

DATASET_PATH = "data/datasets/loan_data.csv"

print("=" * 60)
print("PASO 1 — Cargando dataset")
print("=" * 60)
df = load_loan_dataset(DATASET_PATH)
print(df.head(3))

print("\n" + "=" * 60)
print("PASO 2 — Resumen del dataset")
print("=" * 60)
summary = get_dataset_summary(df)
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n" + "=" * 60)
print("PASO 3 — Preparando features")
print("=" * 60)
X, y = prepare_features(df)
print(f"Shape de X: {X.shape}")
print(f"Columnas: {list(X.columns)}")

print("\n" + "=" * 60)
print("PASO 4 — Split train/test estratificado")
print("=" * 60)
X_train, X_test, y_train, y_test = split_train_test(X, y)

print("\n" + "=" * 60)
print("PASO 5 — Escalado de features")
print("=" * 60)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
print(X_train_scaled.head(3))

print("\n" + "=" * 60)
print("TODO CORRECTO — listo para el trainer")
print("=" * 60)