
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["logistic", "random_forest", "xgboost"]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic",
    **hyperparams
):
    """
    Entrena un modelo de PD con el algoritmo especificado.

    Args:
        X_train: features de entrenamiento.
        y_train: target de entrenamiento.
        model_type: uno de 'logistic', 'random_forest', 'xgboost'.
        **hyperparams: hiperparámetros específicos del modelo elegido,
            se pasan directamente al constructor de sklearn/xgboost.

    Returns:
        Modelo entrenado.

    Raises:
        ValueError: si model_type no es reconocido.
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"model_type debe ser uno de {AVAILABLE_MODELS}, recibido: {model_type}")

    if model_type == "logistic":
        defaults = {"max_iter": 1000, "random_state": 42}
        defaults.update(hyperparams)
        model = LogisticRegression(**defaults)

    elif model_type == "random_forest":
        defaults = {"n_estimators": 200, "max_depth": 6, "random_state": 42}
        defaults.update(hyperparams)
        model = RandomForestClassifier(**defaults)

    elif model_type == "xgboost":
        defaults = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "random_state": 42}
        defaults.update(hyperparams)
        model = XGBClassifier(**defaults, eval_metric="logloss")

    model.fit(X_train, y_train)

    logger.info(
        f"Modelo '{model_type}' entrenado — {X_train.shape[1]} features, "
        f"{len(X_train)} observaciones, hiperparámetros: {defaults}"
    )

    return model


def get_feature_importance(model, feature_names: list[str]) -> dict:
    """
    Extrae importancia de variables, adaptándose al tipo de modelo:
    - Logistic Regression → coeficientes (con signo, indican dirección del efecto)
    - Random Forest / XGBoost → feature_importances_ (sin signo, solo magnitud)
    """
    if hasattr(model, "coef_"):
        importance = dict(zip(feature_names, model.coef_[0]))
    elif hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_names, model.feature_importances_))
    else:
        raise ValueError("El modelo no expone coef_ ni feature_importances_")

    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))


def explain_model_choice(model_type: str) -> str:
    """
    Devuelve el razonamiento técnico estándar sobre cuándo conviene cada modelo.
    El LLM puede usar esto como base para justificar la elección al usuario.
    """
    explanations = {
        "logistic": (
            "Modelo interpretable, coeficientes con significado directo. "
            "Es el estándar de facto en modelos IRB regulados porque facilita "
            "la justificación ante el supervisor (CRR Art. 174). Recomendado "
            "cuando la explicabilidad es prioritaria."
        ),
        "random_forest": (
            "Captura relaciones no lineales e interacciones entre variables sin "
            "especificarlas manualmente. Menos interpretable que la regresión "
            "logística, pero más que XGBoost. Buen punto medio entre performance "
            "y explicabilidad."
        ),
        "xgboost": (
            "Generalmente el mejor performance bruto (mayor Gini/KS) gracias al "
            "boosting secuencial. Es una 'caja negra' relativamente opaca — su uso "
            "en modelos IRB de producción requiere justificación adicional de "
            "explicabilidad (ej. SHAP values) ante el supervisor."
        ),
    }
    return explanations.get(model_type, "Modelo no reconocido.")