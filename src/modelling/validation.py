
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import logging

logger = logging.getLogger(__name__)


def calculate_gini(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    """
    Gini = 2 * AUC - 1. Es la métrica estándar en banca para medir
    capacidad discriminante de un modelo de PD (a cuánto separa buenos de malos pagadores).
    Rango: 0 (sin poder discriminante) a 1 (discriminación perfecta).
    En la práctica, un modelo de PD aceptable suele rondar 0.3-0.5 de Gini.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def calculate_ks(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov: máxima distancia entre las distribuciones acumuladas
    de buenos y malos pagadores según el score del modelo.
    Es la otra métrica estándar exigida en validación de modelos IRB junto al Gini.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(np.abs(tpr - fpr))
    return ks


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evalúa un modelo de PD ya entrenado con las métricas estándar de banca.

    Args:
        model: modelo ya entrenado (Logistic, RandomForest o XGBoost).
        X_test: features de test.
        y_test: target real de test.

    Returns:
        Dict con gini, ks, auc — listo para que el LLM lo redacte sin tener
        que calcular nada por su cuenta.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]   # probabilidad de la clase "impago"

    auc = roc_auc_score(y_test, y_pred_proba)
    gini = calculate_gini(y_test, y_pred_proba)
    ks = calculate_ks(y_test, y_pred_proba)

    logger.info(f"Evaluación — AUC: {auc:.3f}, Gini: {gini:.3f}, KS: {ks:.3f}")

    return {
        "auc": auc,
        "gini": gini,
        "ks": ks,
    }