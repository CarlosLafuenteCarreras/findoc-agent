
from langchain.tools import tool
from typing import Optional
import pandas as pd

from src.modelling.data_loader import load_loan_dataset, get_dataset_summary
from src.modelling.preprocessing import prepare_features, split_train_test, scale_features
from src.modelling.trainer import train_model, get_feature_importance, explain_model_choice, AVAILABLE_MODELS
from src.modelling.validation import evaluate_model  # lo escribimos después

# Estado compartido entre tools de modelado — mismo patrón que tools.py
_dataset: Optional[pd.DataFrame] = None
_X_train = _X_test = _y_train = _y_test = None
_trained_models: dict = {}   # guarda varios modelos por nombre, para poder comparar


@tool
def prepare_dataset_tool(dataset_path: str = "data/datasets/loan_data.csv") -> str:
    """
    Carga y preprocesa el dataset de préstamos para entrenar un modelo de PD.
    Úsala SIEMPRE como primer paso antes de entrenar cualquier modelo, salvo que
    el dataset ya esté preparado en esta sesión.
    Output: resumen estadístico del dataset (tasa de impago, número de filas, etc.)
    """
    global _dataset, _X_train, _X_test, _y_train, _y_test

    _dataset = load_loan_dataset(dataset_path)
    summary = get_dataset_summary(_dataset)

    X, y = prepare_features(_dataset)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train, X_test, _ = scale_features(X_train, X_test)

    _X_train, _X_test, _y_train, _y_test = X_train, X_test, y_train, y_test

    return (
        f"Dataset preparado: {summary['n_rows']} préstamos, "
        f"tasa de impago {summary['default_rate']:.2%}, "
        f"{summary['n_features']} variables tras encoding. "
        f"Listo para entrenar con cualquiera de: {AVAILABLE_MODELS}"
    )


@tool
def train_pd_model_tool(model_type: str = "logistic") -> str:
    """
    Entrena un modelo de Probabilidad de Default con el algoritmo especificado.
    Requiere haber llamado antes a prepare_dataset_tool en esta misma conversación.
    Úsala cuando el usuario pida entrenar, construir o ajustar un modelo de PD.

    Input: model_type, uno de 'logistic' (interpretable, estándar en banca regulada),
    'random_forest' (equilibrio interpretabilidad/performance), o 'xgboost'
    (mejor performance, menos explicable). Si el usuario no especifica, usa 'logistic'
    por ser el estándar regulatorio. Si pide "el mejor modelo posible", usa 'xgboost'.
    Si pide algo "auditable" o "explicable", usa 'logistic'.

    Output: confirmación del entrenamiento y razonamiento sobre la elección del modelo.
    """
    global _trained_models

    if _X_train is None:
        return "Error: primero hay que preparar el dataset con prepare_dataset_tool."

    model = train_model(_X_train, _y_train, model_type=model_type)
    _trained_models[model_type] = model

    explanation = explain_model_choice(model_type)

    return (
        f"Modelo '{model_type}' entrenado correctamente sobre {len(_X_train)} observaciones.\n\n"
        f"Razonamiento de la elección: {explanation}"
    )


@tool
def evaluate_pd_model_tool(model_type: str = "logistic") -> str:
    """
    Evalúa un modelo ya entrenado calculando Gini, KS y la matriz de confusión.
    Requiere haber entrenado antes ese model_type con train_pd_model_tool.
    Úsala cuando el usuario pregunte qué tal funciona el modelo, su precisión,  o pida las métricas de validación. Explica cada metrica y su significado en el contexto de PD.

    Input: model_type del modelo ya entrenado a evaluar.
    Output: métricas de validación exactas, calculadas con sklearn (no estimadas).
    """
    if model_type not in _trained_models:
        return f"Error: el modelo '{model_type}' no ha sido entrenado todavía."

    model = _trained_models[model_type]
    metrics = evaluate_model(model, _X_test, _y_test)

    return (
        f"Métricas del modelo '{model_type}' sobre el conjunto de test "
        f"({len(_X_test)} observaciones):\n"
        f"- Gini: {metrics['gini']:.3f}\n"
        f"- KS: {metrics['ks']:.3f}\n"
        f"- AUC: {metrics['auc']:.3f}\n"
        f"- Tasa de impago real en test: {_y_test.mean():.2%}"
    )


@tool
def explain_pd_model_tool(model_type: str = "logistic") -> str:
    """
    Explica qué variables son más influyentes en un modelo ya entrenado.
    Úsala cuando el usuario pregunte qué variables importan más, por qué el
    modelo predice lo que predice, o pida interpretar los resultados.

    Input: model_type del modelo ya entrenado a interpretar.
    Output: ranking de variables por importancia con interpretación.
    """
    if model_type not in _trained_models:
        return f"Error: el modelo '{model_type}' no ha sido entrenado todavía."

    model = _trained_models[model_type]
    importance = get_feature_importance(model, _X_train.columns.tolist())

    top_5 = list(importance.items())[:5]
    lines = [f"- {var}: {val:+.4f}" for var, val in top_5]

    return "Las 5 variables más influyentes en el modelo '{}':\n".format(model_type) + "\n".join(lines)


@tool
def compare_pd_models_tool(model_types: str = "logistic,xgboost") -> str:
    """
    Entrena y compara varios modelos a la vez según su rendimiento (Gini, KS).
    Úsala cuando el usuario pida comparar algoritmos o encontrar "el mejor modelo".

    Input: model_types como string separado por comas, ej. 'logistic,random_forest,xgboost'.
    Output: tabla comparativa de métricas para decidir cuál usar.
    """
    if _X_train is None:
        return "Error: primero hay que preparar el dataset con prepare_dataset_tool."

    types = [t.strip() for t in model_types.split(",")]
    results = []

    for t in types:
        model = train_model(_X_train, _y_train, model_type=t)
        _trained_models[t] = model
        metrics = evaluate_model(model, _X_test, _y_test)
        results.append(f"- {t}: Gini={metrics['gini']:.3f}, KS={metrics['ks']:.3f}, AUC={metrics['auc']:.3f}")

    return "Comparación de modelos:\n" + "\n".join(results)